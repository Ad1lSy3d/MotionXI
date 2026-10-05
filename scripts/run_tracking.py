import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set, Tuple

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from courtvision.detection.detector import PlayerDetector
from courtvision.projection.homography import PitchHomography
from courtvision.tracking.bytetrack_tracker import FootballByteTracker
from courtvision.tracking.track_types import TrackedPlayer
from courtvision.utils.video import (
    VideoMetadata,
    calculate_pairwise_ious,
    create_video_writer,
    draw_tracks,
    get_video_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CourtVision AI - Milestone 2: YOLOv11m + ByteTrack Baseline Pipeline"
    )
    parser.add_argument(
        "--video",
        "--input",
        dest="video",
        type=str,
        default="data/test_match.mp4",
        help="Path to input video file (default: data/test_match.mp4)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to YOLOv11m trained weights (default: weights/best.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/tracking",
        help="Directory to save tracking output files (default: outputs/tracking)",
    )
    parser.add_argument(
        "--homography",
        type=str,
        default="data/homography_matrix.json",
        help="Path to calibrated 3x3 homography matrix (default: data/homography_matrix.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for PyTorch inference (mps, cuda, cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Detector NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
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
        "--trail-length",
        type=int,
        default=15,
        help="Visualization trail history length in frames (default: 15)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip generating annotated video output",
    )
    return parser.parse_args()


def detect_id_switch_candidates(
    frame_idx: int,
    timestamp: float,
    current_tracks: List[TrackedPlayer],
    active_track_ids: Set[int],
    prev_active_track_ids: Set[int],
    track_last_known_positions: Dict[int, Tuple[int, float, float]],
    overlapping_pairs: List[Tuple[int, int, float]],
) -> List[Dict[str, Any]]:
    """
    Heuristic detection of suspicious tracking behaviors / candidate identity switches.
    These are purely candidate indicators for baseline diagnostic analysis.
    """
    candidates = []

    # 1. Heavy spatial overlap between tracks (IoU > 0.40)
    for i, j, iou_val in overlapping_pairs:
        if iou_val >= 0.40:
            t1 = current_tracks[i].track_id
            t2 = current_tracks[j].track_id
            candidates.append({
                "frame": frame_idx,
                "timestamp": round(timestamp, 3),
                "candidate_type": "high_overlap_cluster",
                "track_id_1": t1,
                "track_id_2": t2,
                "distance_px": round(float(np.hypot(
                    current_tracks[i].foot_x - current_tracks[j].foot_x,
                    current_tracks[i].foot_y - current_tracks[j].foot_y
                )), 1),
                "iou": round(float(iou_val), 3),
                "heuristic_candidate": True,
                "description": f"Tracks {t1} and {t2} heavily overlapping (IoU={iou_val:.2f})",
            })

    # 2. Track disappearance followed by close spatial rebirth of a new track
    terminated_ids = prev_active_track_ids - active_track_ids
    for t_id in terminated_ids:
        # Record last position when track was last active
        for trk in current_tracks:
            pass  # t_id is not in current_tracks

    newly_born_ids = active_track_ids - prev_active_track_ids
    for new_id in newly_born_ids:
        new_trk = next((t for t in current_tracks if t.track_id == new_id), None)
        if new_trk is None:
            continue
        # Compare with tracks that disappeared recently (within 1-10 frames)
        for dead_id, (last_frame, last_fx, last_fy) in list(track_last_known_positions.items()):
            if dead_id != new_id and 1 <= (frame_idx - last_frame) <= 10:
                dist = float(np.hypot(new_trk.foot_x - last_fx, new_trk.foot_y - last_fy))
                if dist < 45.0:  # within 45 pixels
                    candidates.append({
                        "frame": frame_idx,
                        "timestamp": round(timestamp, 3),
                        "candidate_type": "spatial_substitution_rebirth",
                        "track_id_1": dead_id,
                        "track_id_2": new_id,
                        "distance_px": round(dist, 1),
                        "iou": 0.0,
                        "heuristic_candidate": True,
                        "description": (
                            f"Track {dead_id} ended at frame {last_frame}, new Track {new_id} "
                            f"appeared {frame_idx - last_frame} frames later at distance {dist:.1f}px"
                        ),
                    })

    # Update last known positions
    for trk in current_tracks:
        track_last_known_positions[trk.track_id] = (frame_idx, trk.foot_x, trk.foot_y)

    return candidates


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    homography_path = Path(args.homography)
    diag_dir = output_dir / "diagnostics"

    output_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("COURTVISION AI: MILESTONE 2 BYTETRACK BASELINE")
    print("==================================================")
    print(f"Input Video:      {video_path}")
    print(f"Model Checkpoint: {weights_path}")
    print(f"Output Directory: {output_dir}")

    # 1. Video Metadata
    metadata: VideoMetadata = get_video_metadata(video_path)
    print(f"Video Properties: {metadata.width}x{metadata.height} @ {metadata.fps:.2f} FPS")
    print(f"Container Frames: {metadata.container_frame_count}")

    # 2. Homography Initialization & Safety Validation
    homography: PitchHomography = None
    if homography_path.exists():
        homography = PitchHomography(homography_path)
        print(f"Homography:       Loaded 3x3 matrix from {homography_path}")
        print("                  Calibration Scope: penalty_box_region (18-yard box)")
        print("                  Pitch Dimensions:  105m x 68m")
        print("                  Full Pitch Safe:   FALSE (Boundary validation enforced)")
    else:
        print(f"Homography:       [WARNING] {homography_path} not found. Pitch projection disabled.")

    # 3. Detector Initialization
    detector = PlayerDetector(
        model_path=str(weights_path),
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
    )
    print(f"Detector Device:  {detector.device.upper()}")

    # 4. ByteTrack Baseline Initialization
    tracker = FootballByteTracker(
        track_activation_threshold=args.track_thresh,
        lost_track_buffer=args.track_buffer,
        minimum_matching_threshold=args.match_thresh,
        frame_rate=int(round(metadata.fps)),
        homography=homography,
        trail_length=args.trail_length,
    )
    print(f"Tracker:          ByteTrack (activation={args.track_thresh}, buffer={args.track_buffer}f, match={args.match_thresh})")

    # 5. Video Writer Initialization
    annotated_video_path = output_dir / "bytetrack_annotated.mp4"
    video_writer = None
    if not args.no_video:
        video_writer = create_video_writer(
            output_path=annotated_video_path,
            fps=metadata.fps,
            width=metadata.width,
            height=metadata.height,
        )
        print(f"Video Output:     {annotated_video_path}")

    # 6. Stream Execution Loop
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    all_tracks: List[Dict[str, Any]] = []
    crowded_frames: List[Dict[str, Any]] = []
    id_switch_candidates: List[Dict[str, Any]] = []

    detector_times_ms: List[float] = []
    tracker_times_ms: List[float] = []
    frame_processing_times_sec: List[float] = []

    saved_diag_frames: Dict[int, np.ndarray] = {}
    prev_active_track_ids: Set[int] = set()
    track_last_known_positions: Dict[int, Tuple[int, float, float]] = {}

    print("\nRunning tracking pipeline across video frames...")
    pipeline_start_time = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                # Reached authoritative end of decoded stream
                break

            timestamp = frame_idx / metadata.fps if metadata.fps > 0 else 0.0

            # 1. Detector Inference
            detections, infer_ms = detector.detect(
                frame=frame,
                frame_idx=frame_idx,
                fps=metadata.fps,
            )
            detector_times_ms.append(infer_ms)

            # 2. ByteTrack Association
            tracks, track_ms = tracker.update(
                detections=detections,
                frame_idx=frame_idx,
                fps=metadata.fps,
            )
            tracker_times_ms.append(track_ms)

            # 3. Store Track Records
            current_active_ids = set()
            for t in tracks:
                all_tracks.append(t.to_dict())
                current_active_ids.add(t.track_id)

            # 4. Crowded-Frame Analysis
            boxes = np.array([[t.x1, t.y1, t.x2, t.y2] for t in tracks], dtype=np.float32)
            max_iou, overlap_count, overlapping_pairs = calculate_pairwise_ious(boxes, min_iou_threshold=0.1)
            crowded_frames.append({
                "frame": frame_idx,
                "timestamp": round(timestamp, 3),
                "player_count": len(tracks),
                "max_pairwise_iou": max_iou,
                "overlapping_pairs": overlap_count,
            })

            # 5. Heuristic ID-Switch Candidate Detection
            id_switch_events = detect_id_switch_candidates(
                frame_idx=frame_idx,
                timestamp=timestamp,
                current_tracks=tracks,
                active_track_ids=current_active_ids,
                prev_active_track_ids=prev_active_track_ids,
                track_last_known_positions=track_last_known_positions,
                overlapping_pairs=overlapping_pairs,
            )
            id_switch_candidates.extend(id_switch_events)
            prev_active_track_ids = current_active_ids

            # 6. Render Video Visualization
            total_unique_so_far = len(tracker.track_stats)
            annotated_frame = draw_tracks(
                frame=frame,
                tracks=tracks,
                trail_history=tracker.trail_history,
                frame_idx=frame_idx,
                total_frames=metadata.container_frame_count,
                fps=metadata.fps,
                infer_ms=infer_ms,
                track_ms=track_ms,
                total_unique_ids=total_unique_so_far,
                device_name=detector.device,
            )

            # Cache frame for potential diagnostic snapshot
            if len(tracks) >= 12 and (max_iou > 0.35 or overlap_count >= 3):
                saved_diag_frames[frame_idx] = annotated_frame.copy()

            if video_writer is not None:
                video_writer.write(annotated_frame)

            frame_elapsed = time.perf_counter() - frame_start
            frame_processing_times_sec.append(frame_elapsed)

            frame_idx += 1
            if frame_idx % 60 == 0 or frame_idx == metadata.container_frame_count:
                fps_instant = 1.0 / frame_elapsed if frame_elapsed > 0 else 0.0
                print(
                    f"  Frame {frame_idx:03d} | Active: {len(tracks):2d} | "
                    f"Total IDs: {total_unique_so_far:2d} | "
                    f"Infer: {infer_ms:4.1f}ms | Track: {track_ms:4.2f}ms | "
                    f"Rate: {fps_instant:4.1f} FPS"
                )

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()

    pipeline_total_time = time.perf_counter() - pipeline_start_time
    effective_fps = frame_idx / pipeline_total_time if pipeline_total_time > 0 else 0.0
    avg_frame_time_ms = (pipeline_total_time / frame_idx * 1000.0) if frame_idx > 0 else 0.0
    avg_infer_ms = float(np.mean(detector_times_ms)) if detector_times_ms else 0.0
    avg_track_ms = float(np.mean(tracker_times_ms)) if tracker_times_ms else 0.0

    # Authoritative decoded metadata
    metadata = metadata.with_decoded_count(frame_idx)

    print("\nTracking processing complete.")

    # 7. Save Tracks CSV
    tracks_csv_path = output_dir / "tracks.csv"
    if all_tracks:
        fieldnames = list(all_tracks[0].keys())
        with open(tracks_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_tracks)
    print(f"Saved tracks CSV:               {tracks_csv_path}")

    # 8. Save Tracks JSON
    tracks_json_path = output_dir / "tracks.json"
    with open(tracks_json_path, "w") as f:
        json.dump(all_tracks, f, indent=2)
    print(f"Saved tracks JSON:              {tracks_json_path}")

    # 9. Save Track Summary CSV
    summary_csv_path = output_dir / "track_summary.csv"
    summary_records = tracker.get_track_summary_records()
    if summary_records:
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
            writer.writeheader()
            writer.writerows(summary_records)
    print(f"Saved track summary CSV:        {summary_csv_path}")

    # 10. Save Crowded Frames CSV (sorted by crowding severity)
    crowded_csv_path = output_dir / "crowded_frames.csv"
    crowded_sorted = sorted(
        crowded_frames,
        key=lambda x: (x["overlapping_pairs"], x["max_pairwise_iou"], x["player_count"]),
        reverse=True,
    )
    if crowded_sorted:
        with open(crowded_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(crowded_sorted[0].keys()))
            writer.writeheader()
            writer.writerows(crowded_sorted)
    print(f"Saved crowded frames CSV:       {crowded_csv_path}")

    # 11. Save Top 10 Crowded Diagnostic Images
    top_10_crowded = crowded_sorted[:10]
    saved_img_count = 0
    for cf in top_10_crowded:
        cf_frame = cf["frame"]
        if cf_frame in saved_diag_frames:
            diag_img_path = diag_dir / f"frame_{cf_frame:03d}_crowded.png"
            cv2.imwrite(str(diag_img_path), saved_diag_frames[cf_frame])
            saved_img_count += 1
    print(f"Saved {saved_img_count} diagnostic images to:    {diag_dir}")

    # 12. Save ID Switch Candidates CSV
    id_switch_csv_path = output_dir / "id_switch_candidates.csv"
    if id_switch_candidates:
        with open(id_switch_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(id_switch_candidates[0].keys()))
            writer.writeheader()
            writer.writerows(id_switch_candidates)
    else:
        with open(id_switch_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame", "timestamp", "candidate_type", "track_id_1",
                "track_id_2", "distance_px", "iou", "heuristic_candidate", "description"
            ])
            writer.writeheader()
    print(f"Saved ID switch candidates CSV: {id_switch_csv_path}")

    # 13. Compile and Save Tracking Metadata
    track_stats = tracker.get_track_statistics()
    metadata_path = output_dir / "tracking_metadata.json"

    tracking_metadata: Dict[str, Any] = {
        "pipeline_version": "0.2.0",
        "milestone": "Milestone 2: ByteTrack Baseline",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detector": {
            "model": "YOLOv11m",
            "weights": str(weights_path),
            "classes": {"0": "player"},
            "conf_threshold": detector.conf_threshold,
            "iou_threshold": detector.iou_threshold,
            "imgsz": detector.imgsz,
        },
        "tracker": tracker.get_config(),
        "video": metadata.to_dict(),
        "homography": homography.get_metadata() if homography is not None else {
            "enabled": False,
            "calibration_scope": "none",
            "full_pitch_calibration": False,
        },
        "hardware": {
            "device": detector.device,
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "performance": {
            "frames_processed": frame_idx,
            "total_processing_time_sec": round(pipeline_total_time, 3),
            "effective_pipeline_fps": round(effective_fps, 2),
            "avg_detector_ms": round(avg_infer_ms, 2),
            "avg_tracker_ms": round(avg_track_ms, 2),
            "avg_end_to_end_frame_ms": round(avg_frame_time_ms, 2),
        },
        "tracking_statistics": track_stats,
        "crowded_frame_diagnostics": {
            "crowded_frames_analyzed": len(crowded_frames),
            "frames_with_overlapping_players": sum(1 for c in crowded_frames if c["overlapping_pairs"] > 0),
            "top_crowded_frames": top_10_crowded,
        },
        "id_switch_diagnostics": {
            "heuristic_candidate_events_count": len(id_switch_candidates),
            "note": (
                "These are heuristic indicator events only (high overlap clusters and close spatial rebirths), "
                "not ground-truth benchmarked ID switches."
            ),
        },
        "outputs": {
            "annotated_video": str(annotated_video_path) if not args.no_video else None,
            "tracks_csv": str(tracks_csv_path),
            "tracks_json": str(tracks_json_path),
            "track_summary_csv": str(summary_csv_path),
            "crowded_frames_csv": str(crowded_csv_path),
            "id_switch_candidates_csv": str(id_switch_csv_path),
            "diagnostics_dir": str(diag_dir),
        },
    }

    with open(metadata_path, "w") as f:
        json.dump(tracking_metadata, f, indent=2)
    print(f"Saved tracking metadata JSON:   {metadata_path}")

    print("\n==================================================")
    print("BYTETRACK BASELINE RUN SUMMARY")
    print("==================================================")
    print(f"Decoded Frames:         {frame_idx} (Container reported: {metadata.container_frame_count})")
    print(f"Total Unique Tracks:    {track_stats['total_unique_track_ids']}")
    print(f"Avg Track Length:       {track_stats['average_track_length_frames']} frames")
    print(f"Median Track Length:    {track_stats['median_track_length_frames']} frames")
    print(f"Max Track Length:       {track_stats['max_track_length_frames']} frames")
    print(f"Avg Active Tracks/Frame:{track_stats['average_active_tracks_per_frame']}")
    print(f"Max Active Tracks/Frame:{track_stats['maximum_active_tracks_per_frame']}")
    print(f"Total Track Records:    {len(all_tracks)}")
    print(f"Heuristic IDSw Events:  {len(id_switch_candidates)}")
    print("--------------------------------------------------")
    print(f"Detector Inference Avg: {avg_infer_ms:.2f} ms/frame")
    print(f"Tracker Association Avg:{avg_track_ms:.2f} ms/frame")
    print(f"End-to-End Pipeline Avg:{avg_frame_time_ms:.2f} ms/frame")
    print(f"Effective Throughput:   {effective_fps:.2f} FPS")
    print("==================================================")


if __name__ == "__main__":
    main()
