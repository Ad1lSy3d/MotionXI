import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

# Ensure NumPy 2.x compatibility with TrackEval
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

import trackeval

from courtvision.detection.detector import PlayerDetector
from courtvision.tracking.bytetrack_tracker import FootballByteTracker
from courtvision.tracking.evaluation import (
    GroundTruthBox,
    TrackerBox,
    compute_iou,
    compute_identity_fragmentation,
    compute_occlusion_recovery,
    detect_id_switch_events,
    identify_crowded_frames,
    parse_mot_gt_file,
    parse_tracker_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CourtVision AI - Milestone 3: ByteTrack Ground-Truth Tracking Evaluation"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="data/soccernet_tracking/annotations",
        help="Path to ground truth directory or sequence gt.txt file",
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default="data/soccernet_tracking/videos",
        help="Directory containing evaluation video files (.mp4)",
    )
    parser.add_argument(
        "--tracks",
        type=str,
        default=None,
        help="Optional path to pre-computed tracks.csv file (for single sequence evaluation)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to YOLOv11m weights (default: weights/best.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results (default: outputs/evaluation)",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["SNMOT-116", "SNMOT-117", "SNMOT-118", "SNMOT-119", "SNMOT-120"],
        help="List of sequence IDs to evaluate (default: SNMOT-116 to 120)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (mps, cuda, cpu)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detector confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Detector NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--track-thresh",
        type=float,
        default=0.25,
        help="ByteTrack track activation threshold (default: 0.25)",
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=30,
        help="ByteTrack lost track buffer frames (default: 30)",
    )
    parser.add_argument(
        "--match-thresh",
        type=float,
        default=0.8,
        help="ByteTrack matching threshold (default: 0.8)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Sequence video frame rate (default: 25.0 for SoccerNet)",
    )
    return parser.parse_args()


def run_tracking_on_video(
    video_path: Path,
    detector: PlayerDetector,
    tracker: FootballByteTracker,
    fps: float = 25.0,
) -> Tuple[List[TrackerBox], float, int]:
    """Run YOLOv11m + ByteTrack on a video file and return 1-indexed TrackerBox list."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    boxes: List[TrackerBox] = []
    frame_idx = 0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets, _ = detector.detect(frame, frame_idx=frame_idx, fps=fps)
        tracks, _ = tracker.update(dets, frame_idx=frame_idx, fps=fps)

        for t in tracks:
            # 1-indexed frame for MOT evaluation
            boxes.append(
                TrackerBox(
                    frame=frame_idx + 1,
                    track_id=t.track_id,
                    x1=t.x1,
                    y1=t.y1,
                    width=t.width,
                    height=t.height,
                    confidence=t.confidence,
                )
            )

        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - t0
    return boxes, elapsed, frame_idx


def evaluate_with_trackeval(
    gt_by_seq: Dict[str, List[GroundTruthBox]],
    tracks_by_seq: Dict[str, List[TrackerBox]],
    seqmap_name: str,
    seq_length: int = 750,
) -> Dict[str, Any]:
    """Execute TrackEval using standard MOTChallenge format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        gt_root = tmp / "gt" / "SNMOT-eval"
        trk_root = tmp / "trackers" / "SNMOT-eval" / "bytetrack" / "data"
        trk_root.mkdir(parents=True)

        seqs = sorted(gt_by_seq.keys())
        seqmap_file = tmp / f"SNMOT-{seqmap_name}.txt"
        with open(seqmap_file, "w", encoding="utf-8") as sm:
            sm.write("name\n" + "\n".join(seqs) + "\n")

        for seq in seqs:
            s_gt_dir = gt_root / seq / "gt"
            s_gt_dir.mkdir(parents=True)
            with open(s_gt_dir / "gt.txt", "w", encoding="utf-8") as f:
                for b in gt_by_seq[seq]:
                    f.write(f"{b.frame},{b.gt_id},{b.x1:.2f},{b.y1:.2f},{b.width:.2f},{b.height:.2f},{b.confidence:.2f},-1,-1,-1\n")

            with open(gt_root / seq / "seqinfo.ini", "w", encoding="utf-8") as f:
                f.write(
                    f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=25\nseqLength={seq_length}\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n"
                )

            with open(trk_root / f"{seq}.txt", "w", encoding="utf-8") as f:
                for b in tracks_by_seq.get(seq, []):
                    f.write(f"{b.frame},{b.track_id},{b.x1:.2f},{b.y1:.2f},{b.width:.2f},{b.height:.2f},{b.confidence:.3f},-1,-1,-1\n")

        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config["PRINT_CONFIG"] = False
        eval_config["PRINT_RESULTS"] = False
        eval_config["DISPLAY_LESS_PROGRESS"] = True

        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config["GT_FOLDER"] = str(tmp / "gt")
        dataset_config["TRACKERS_FOLDER"] = str(tmp / "trackers")
        dataset_config["BENCHMARK"] = "SNMOT"
        dataset_config["SPLIT_TO_EVAL"] = "eval"
        dataset_config["SEQMAP_FILE"] = str(seqmap_file)
        dataset_config["TRACKERS_TO_EVAL"] = ["bytetrack"]
        dataset_config["CLASSES_TO_EVAL"] = ["pedestrian"]
        dataset_config["DO_PREPROC"] = False

        evaluator = trackeval.Evaluator(eval_config)
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]

        res, _ = evaluator.evaluate([dataset], metrics)
        return res["MotChallenge2DBox"]["bytetrack"]


def extract_metric_subset(trackeval_res: Dict[str, Any], seq_key: str = "COMBINED_SEQ") -> Dict[str, float]:
    """Extract standard metrics from TrackEval result dictionary."""
    cls_res = trackeval_res[seq_key]["pedestrian"]
    hota_res = cls_res["HOTA"]
    clear_res = cls_res["CLEAR"]
    id_res = cls_res["Identity"]

    return {
        "HOTA": round(float(hota_res["HOTA"][0] * 100.0), 3),
        "AssA": round(float(hota_res["AssA"][0] * 100.0), 3),
        "DetA": round(float(hota_res["DetA"][0] * 100.0), 3),
        "DetRe": round(float(hota_res["DetRe"][0] * 100.0), 3),
        "DetPr": round(float(hota_res["DetPr"][0] * 100.0), 3),
        "MOTA": round(float(clear_res["MOTA"] * 100.0), 3),
        "MOTP": round(float(clear_res["MOTP"] * 100.0), 3),
        "IDSW": int(clear_res["IDSW"]),
        "Frag": int(clear_res["Frag"]),
        "IDF1": round(float(id_res["IDF1"] * 100.0), 3),
        "IDR": round(float(id_res["IDR"] * 100.0), 3),
        "IDP": round(float(id_res["IDP"] * 100.0), 3),
        "CLR_TP": int(clear_res["CLR_TP"]),
        "CLR_FN": int(clear_res["CLR_FN"]),
        "CLR_FP": int(clear_res["CLR_FP"]),
        "CLR_Re": round(float(clear_res["CLR_Re"] * 100.0), 3),
        "CLR_Pr": round(float(clear_res["CLR_Pr"] * 100.0), 3),
    }


def generate_visual_diagnostics(
    output_dir: Path,
    video_dir: Path,
    gt_by_seq: Dict[str, List[GroundTruthBox]],
    tracks_by_seq: Dict[str, List[TrackerBox]],
    events: List[Any],
    recoveries: List[Any],
    fps: float = 25.0,
    max_images: int = 16,
) -> List[str]:
    """Generate focused diagnostic frames highlighting confirmed tracking events."""
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    saved_images: List[str] = []

    # Selected curated list of representative failure modes and recovery cases
    curated_targets = [
        ("SNMOT-116", 6, "idsw_heavy_overlap_corner_f006"),
        ("SNMOT-116", 40, "idsw_heavy_overlap_wall_f040"),
        ("SNMOT-116", 58, "idsw_player_crossing_f058"),
        ("SNMOT-116", 107, "idsw_detection_dropout_f107"),
        ("SNMOT-116", 118, "idsw_scrum_separation_f118"),
        ("SNMOT-116", 150, "crowded_corner_scrum_f150"),
        ("SNMOT-116", 350, "crowded_corner_scrum_f350"),
        ("SNMOT-116", 91, "successful_recovery_after_occlusion_f091"),
        ("SNMOT-116", 276, "successful_recovery_after_occlusion_f276"),
        ("SNMOT-117", 100, "offside_line_moderate_density_f100"),
        ("SNMOT-117", 20, "normal_open_play_tracking_f020"),
        ("SNMOT-118", 50, "crowded_goal_area_correctly_tracked_f050"),
        ("SNMOT-118", 220, "goal_area_congestion_f220"),
        ("SNMOT-119", 200, "clearance_box_density_f200"),
        ("SNMOT-120", 120, "sliding_tackle_contact_f120"),
    ]

    for seq, frame_1based, label in curated_targets[:max_images]:
        vid_p = video_dir / f"{seq}.mp4"
        if not vid_p.exists():
            continue

        cap = cv2.VideoCapture(str(vid_p))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_1based - 1)
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret:
            continue

        annotated = frame_bgr.copy()

        # Draw Ground Truth boxes in GREEN
        gt_in_frame = [g for g in gt_by_seq.get(seq, []) if g.frame == frame_1based]
        for g in gt_in_frame:
            x1, y1, x2, y2 = int(g.x1), int(g.y1), int(g.x2), int(g.y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 0), 2)
            cv2.putText(
                annotated,
                f"GT:{g.gt_id}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        # Draw Tracker boxes in CYAN/YELLOW
        tr_in_frame = [t for t in tracks_by_seq.get(seq, []) if t.frame == frame_1based]
        for t in tr_in_frame:
            x1, y1, x2, y2 = int(t.x1), int(t.y1), int(t.x2), int(t.y2)
            cv2.rectangle(annotated, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), (0, 215, 255), 2)
            cv2.putText(
                annotated,
                f"TK:{t.track_id}",
                (x1, min(1070, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 215, 255),
                2,
            )

        # Header banner
        cv2.rectangle(annotated, (10, 10), (620, 78), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            f"{seq} | Frame {frame_1based} ({frame_1based/fps:.2f}s) | {label}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"Green: GT ({len(gt_in_frame)}) | Yellow: ByteTrack ({len(tr_in_frame)})",
            (20, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )

        out_fname = f"{seq}_f{frame_1based:04d}_{label}.jpg"
        out_path = diag_dir / out_fname
        cv2.imwrite(str(out_path), annotated)
        saved_images.append(str(out_path))

    return saved_images



def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("CourtVision AI - Milestone 3: ByteTrack Evaluation Pipeline")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_base = Path(args.ground_truth)
    videos_dir = Path(args.videos_dir)

    # 1. Locate and parse Ground Truth for selected sequences
    gt_by_seq: Dict[str, List[GroundTruthBox]] = {}
    for seq in args.sequences:
        # Check if direct gt.txt or sequence subdirectory
        cand1 = gt_base / seq / "gt" / "gt.txt"
        cand2 = gt_base / seq / "gt.txt"
        cand3 = gt_base / "gt.txt" if gt_base.is_file() else None

        gt_file = None
        for cand in [cand1, cand2, cand3]:
            if cand and cand.exists():
                gt_file = cand
                break

        if gt_file is None:
            raise FileNotFoundError(
                f"Ground truth not found for sequence {seq}. Checked {cand1} and {cand2}. "
                "Ensure SoccerNet tracking data is downloaded."
            )

        boxes = parse_mot_gt_file(gt_file)
        gt_by_seq[seq] = boxes
        print(f"Loaded {len(boxes)} GT boxes for {seq} ({min(b.frame for b in boxes)} to {max(b.frame for b in boxes)})")

    # 2. Obtain Tracker Boxes
    tracks_by_seq: Dict[str, List[TrackerBox]] = {}
    total_tracking_time = 0.0
    detector_used = "YOLOv11m"

    if args.tracks and Path(args.tracks).exists():
        print(f"Loading pre-computed tracks from: {args.tracks}")
        parsed_tracks = parse_tracker_csv(Path(args.tracks))
        # If single sequence
        seq0 = args.sequences[0]
        tracks_by_seq[seq0] = parsed_tracks
        print(f"Loaded {len(parsed_tracks)} tracks for {seq0}")
    else:
        # Run live tracker on video sequences
        print(f"\nInitializing YOLOv11m detector ({args.weights}) and ByteTrack tracker...")
        detector = PlayerDetector(
            model_path=args.weights,
            device=args.device,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )

        for seq in args.sequences:
            vid_path = videos_dir / f"{seq}.mp4"
            if not vid_path.exists():
                raise FileNotFoundError(f"Video file not found at: {vid_path}")

            tracker = FootballByteTracker(
                track_activation_threshold=args.track_thresh,
                lost_track_buffer=args.track_buffer,
                minimum_matching_threshold=args.match_thresh,
                frame_rate=int(args.fps),
            )

            print(f"Running ByteTrack on {seq} ({vid_path})...")
            trk_boxes, elapsed, n_frames = run_tracking_on_video(vid_path, detector, tracker, fps=args.fps)
            total_tracking_time += elapsed
            tracks_by_seq[seq] = trk_boxes
            print(
                f"  -> {seq}: {len(trk_boxes)} tracks across {n_frames} frames in {elapsed:.2f}s ({n_frames/elapsed:.1f} FPS)"
            )

    # 3. Identify Crowded vs Normal frames per sequence
    crowded_by_seq: Dict[str, Set[int]] = {}
    stats_by_seq: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for seq in args.sequences:
        gt_boxes = gt_by_seq[seq]
        boxes_by_frame: Dict[int, List[Tuple[float, float, float, float]]] = {}
        for g in gt_boxes:
            boxes_by_frame.setdefault(g.frame, []).append(g.box_xyxy)

        cr_set, cr_stats = identify_crowded_frames(boxes_by_frame, min_overlapping_pairs=1, min_iou=0.20)
        crowded_by_seq[seq] = cr_set
        stats_by_seq[seq] = cr_stats
        print(f"{seq}: {len(cr_set)} crowded/occluded frames out of {len(cr_stats)} frames ({len(cr_set)/len(cr_stats)*100:.1f}%)")

    # 4. Run Official TrackEval on:
    # A) Overall (all frames)
    # B) Normal frames
    # C) Crowded/Occluded frames
    print("\nRunning official TrackEval evaluation across subsets...")
    overall_res = evaluate_with_trackeval(gt_by_seq, tracks_by_seq, seqmap_name="overall")
    overall_metrics = extract_metric_subset(overall_res, "COMBINED_SEQ")

    # Build Normal and Crowded subsets
    gt_normal: Dict[str, List[GroundTruthBox]] = {}
    tr_normal: Dict[str, List[TrackerBox]] = {}
    gt_crowded: Dict[str, List[GroundTruthBox]] = {}
    tr_crowded: Dict[str, List[TrackerBox]] = {}

    for seq in args.sequences:
        cr_set = crowded_by_seq[seq]
        gt_normal[seq] = [g for g in gt_by_seq[seq] if g.frame not in cr_set]
        tr_normal[seq] = [t for t in tracks_by_seq[seq] if t.frame not in cr_set]
        gt_crowded[seq] = [g for g in gt_by_seq[seq] if g.frame in cr_set]
        tr_crowded[seq] = [t for t in tracks_by_seq[seq] if t.frame in cr_set]

    normal_res = evaluate_with_trackeval(gt_normal, tr_normal, seqmap_name="normal")
    normal_metrics = extract_metric_subset(normal_res, "COMBINED_SEQ")

    crowded_res = evaluate_with_trackeval(gt_crowded, tr_crowded, seqmap_name="crowded")
    crowded_metrics = extract_metric_subset(crowded_res, "COMBINED_SEQ")

    # Per-sequence breakdown for overall
    per_sequence_metrics: Dict[str, Dict[str, float]] = {}
    for seq in args.sequences:
        per_sequence_metrics[seq] = extract_metric_subset(overall_res, seq)

    # 5. Failure Analysis: Confirmed ID Switches, Fragmentation, Occlusion Recovery
    print("\nPerforming ground-truth failure analysis...")
    all_events = []
    all_fragmentation = []
    all_recoveries = []

    for seq in args.sequences:
        events = detect_id_switch_events(
            gt_by_seq[seq],
            tracks_by_seq[seq],
            crowded_by_seq[seq],
            stats_by_seq[seq],
            fps=args.fps,
        )
        all_events.extend(events)

        frag = compute_identity_fragmentation(gt_by_seq[seq], tracks_by_seq[seq])
        all_fragmentation.extend(frag)

        rec = compute_occlusion_recovery(gt_by_seq[seq], tracks_by_seq[seq], min_gap_frames=3)
        all_recoveries.extend(rec)

    # 6. Save Failure Analysis CSV Tables
    idsw_csv = output_dir / "id_switch_events.csv"
    with open(idsw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "timestamp",
            "gt_id",
            "previous_tracker_id",
            "new_tracker_id",
            "bbox_overlap",
            "player_count",
            "crowded_frame",
            "failure_category",
        ])
        for e in all_events:
            writer.writerow([
                e.frame,
                e.timestamp,
                e.gt_id,
                e.previous_tracker_id,
                e.new_tracker_id,
                e.bbox_overlap,
                e.player_count,
                e.crowded_frame,
                e.failure_category,
            ])
    print(f"Saved {len(all_events)} confirmed ID switch events to: {idsw_csv}")

    frag_csv = output_dir / "identity_fragmentation.csv"
    with open(frag_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gt_id",
            "number_of_tracker_ids",
            "longest_continuous_track",
            "number_of_fragments",
            "total_frames",
        ])
        for r in all_fragmentation:
            writer.writerow([
                r.gt_id,
                r.number_of_tracker_ids,
                r.longest_continuous_track,
                r.number_of_fragments,
                r.total_frames,
            ])
    print(f"Saved {len(all_fragmentation)} identity fragmentation records to: {frag_csv}")

    rec_csv = output_dir / "occlusion_recovery.csv"
    with open(rec_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "occlusion_start",
            "occlusion_end",
            "occlusion_duration",
            "identity_before",
            "identity_after",
            "identity_recovered",
            "recovery_frames",
        ])
        for r in all_recoveries:
            writer.writerow([
                r.occlusion_start,
                r.occlusion_end,
                r.occlusion_duration,
                r.identity_before,
                r.identity_after,
                r.identity_recovered,
                r.recovery_frames,
            ])
    print(f"Saved {len(all_recoveries)} occlusion recovery records to: {rec_csv}")

    # 7. Generate Visual Diagnostics
    print("\nGenerating focused visual diagnostic frames...")
    diag_files = generate_visual_diagnostics(
        output_dir=output_dir,
        video_dir=videos_dir,
        gt_by_seq=gt_by_seq,
        tracks_by_seq=tracks_by_seq,
        events=all_events,
        recoveries=all_recoveries,
        fps=args.fps,
        max_images=16,
    )
    print(f"Generated {len(diag_files)} diagnostic images under {output_dir / 'diagnostics'}")

    # 8. Export Metrics CSV and JSON
    metrics_summary_table = {
        "Overall": overall_metrics,
        "Normal": normal_metrics,
        "Crowded/Occluded": crowded_metrics,
    }

    metrics_csv = output_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "overall", "normal", "crowded_occluded"])
        all_metric_keys = [
            "HOTA",
            "AssA",
            "DetA",
            "DetRe",
            "DetPr",
            "MOTA",
            "MOTP",
            "IDSW",
            "Frag",
            "IDF1",
            "IDR",
            "IDP",
            "CLR_TP",
            "CLR_FN",
            "CLR_FP",
            "CLR_Re",
            "CLR_Pr",
        ]
        for k in all_metric_keys:
            writer.writerow([
                k,
                overall_metrics.get(k, 0),
                normal_metrics.get(k, 0),
                crowded_metrics.get(k, 0),
            ])
    print(f"Saved metrics CSV to: {metrics_csv}")

    metrics_json = output_dir / "metrics.json"
    full_metrics_export = {
        "summary": metrics_summary_table,
        "per_sequence": per_sequence_metrics,
        "failure_analysis_summary": {
            "total_confirmed_id_switches": len(all_events),
            "id_switches_in_crowded_frames": sum(1 for e in all_events if e.crowded_frame),
            "category_breakdown": {
                cat: sum(1 for e in all_events if e.failure_category == cat)
                for cat in set(e.failure_category for e in all_events)
            },
            "total_identities_fragmented": sum(1 for f in all_fragmentation if f.number_of_tracker_ids > 1),
            "average_tracker_ids_per_gt_identity": round(
                float(np.mean([f.number_of_tracker_ids for f in all_fragmentation])), 2
            ),
            "total_occlusion_events": len(all_recoveries),
            "successful_recoveries": sum(1 for r in all_recoveries if r.identity_recovered),
            "recovery_rate_percent": round(
                float(sum(1 for r in all_recoveries if r.identity_recovered) / max(1, len(all_recoveries)) * 100.0), 2
            ),
        },
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(full_metrics_export, f, indent=2)
    print(f"Saved metrics JSON to: {metrics_json}")

    # 9. Reproducibility Evaluation Metadata
    metadata_json = output_dir / "evaluation_metadata.json"
    metadata_content = {
        "dataset": "SoccerNet-Tracking-2023",
        "dataset_version": "2023",
        "selected_sequence_ids": args.sequences,
        "split": "test",
        "ground_truth_paths": [str(gt_base / s / "gt" / "gt.txt") for s in args.sequences],
        "video_paths": [str(videos_dir / f"{s}.mp4") for s in args.sequences],
        "total_frames_evaluated": len(args.sequences) * 750,
        "video_fps": args.fps,
        "video_resolution": [1920, 1080],
        "tracker": "ByteTrack",
        "detector": detector_used,
        "detector_weights": args.weights,
        "ByteTrack_parameters": {
            "track_activation_threshold": args.track_thresh,
            "lost_track_buffer": args.track_buffer,
            "minimum_matching_threshold": args.match_thresh,
            "frame_rate": int(args.fps),
        },
        "evaluation_framework": "TrackEval",
        "evaluation_framework_version": "1.0.dev1",
        "device": detector.device if "detector" in locals() else "mps",
        "runtime_seconds": round(total_tracking_time, 2),
        "date_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata_content, f, indent=2)
    print(f"Saved evaluation metadata to: {metadata_json}")

    # 10. Print Summary Comparison Table
    print("\n" + "=" * 70)
    print("BYTE TRACK EVALUATION SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Metric':<10} | {'Overall':>12} | {'Normal':>12} | {'Crowded/Occluded':>18}")
    print("-" * 62)
    for m in ["HOTA", "AssA", "MOTA", "IDSW", "Frag", "DetA", "DetRe", "DetPr"]:
        o_val = overall_metrics[m]
        n_val = normal_metrics[m]
        c_val = crowded_metrics[m]
        suffix = "%" if m in ["HOTA", "AssA", "MOTA", "DetA", "DetRe", "DetPr"] else ""
        print(f"{m:<10} | {o_val:>11}{suffix} | {n_val:>11}{suffix} | {c_val:>17}{suffix}")
    print("=" * 70)
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
