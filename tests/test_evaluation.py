import json
from pathlib import Path
import tempfile
import pytest

from courtvision.tracking.evaluation import (
    GroundTruthBox,
    TrackerBox,
    compute_iou,
    compute_identity_fragmentation,
    compute_occlusion_recovery,
    detect_id_switch_events,
    identify_crowded_frames,
    match_gt_to_tracker,
    parse_mot_gt_file,
    parse_tracker_csv,
)


def test_mot_gt_parsing(tmp_path: Path):
    gt_file = tmp_path / "gt.txt"
    gt_file.write_text(
        "1,1,100.0,200.0,50.0,150.0,1,1,1.0\n"
        "1,2,300.0,400.0,60.0,180.0,1,1,1.0\n"
        "2,1,102.0,201.0,50.0,150.0,1,1,1.0\n",
        encoding="utf-8",
    )

    boxes = parse_mot_gt_file(gt_file)
    assert len(boxes) == 3
    assert boxes[0].frame == 1
    assert boxes[0].gt_id == 1
    assert boxes[0].x1 == 100.0
    assert boxes[0].y1 == 200.0
    assert boxes[0].width == 50.0
    assert boxes[0].height == 150.0
    assert boxes[0].x2 == 150.0
    assert boxes[0].y2 == 350.0
    assert boxes[0].box_xyxy == (100.0, 200.0, 150.0, 350.0)


def test_video_gt_frame_alignment_and_tracker_csv_parsing(tmp_path: Path):
    # CourtVision format: 0-indexed frame -> converts to 1-indexed MOT frame
    cv_csv = tmp_path / "tracks_courtvision.csv"
    cv_csv.write_text(
        "frame,track_id,x1,y1,x2,y2,width,height,confidence\n"
        "0,10,100,200,150,350,50,150,0.95\n"
        "1,10,105,202,155,352,50,150,0.92\n",
        encoding="utf-8",
    )
    cv_boxes = parse_tracker_csv(cv_csv)
    assert len(cv_boxes) == 2
    # Verify frame 0 was incremented to frame 1 for MOT alignment
    assert cv_boxes[0].frame == 1
    assert cv_boxes[1].frame == 2
    assert cv_boxes[0].track_id == 10

    # Standard MOT format: already 1-indexed
    mot_csv = tmp_path / "tracks_mot.csv"
    mot_csv.write_text(
        "1,10,100,200,50,150,0.95,-1,-1,-1\n"
        "2,10,105,202,50,150,0.92,-1,-1,-1\n",
        encoding="utf-8",
    )
    mot_boxes = parse_tracker_csv(mot_csv)
    assert len(mot_boxes) == 2
    assert mot_boxes[0].frame == 1
    assert mot_boxes[1].frame == 2


def test_evaluator_bipartite_matching():
    gt_boxes = [
        GroundTruthBox(frame=1, gt_id=1, x1=100.0, y1=100.0, width=50.0, height=100.0),
        GroundTruthBox(frame=1, gt_id=2, x1=400.0, y1=400.0, width=50.0, height=100.0),
        GroundTruthBox(frame=1, gt_id=3, x1=800.0, y1=800.0, width=50.0, height=100.0),
    ]
    # Tracker detected gt 1 and 2 accurately, but missed 3 and hallucinated a false positive
    tracker_boxes = [
        TrackerBox(frame=1, track_id=101, x1=102.0, y1=101.0, width=50.0, height=100.0),
        TrackerBox(frame=1, track_id=102, x1=398.0, y1=402.0, width=50.0, height=100.0),
        TrackerBox(frame=1, track_id=999, x1=1200.0, y1=1200.0, width=50.0, height=100.0),
    ]

    matches, unmatched_gt, unmatched_tr = match_gt_to_tracker(gt_boxes, tracker_boxes, iou_threshold=0.5)

    assert matches[1] == 101
    assert matches[2] == 102
    assert 3 in unmatched_gt
    assert 999 in unmatched_tr


def test_crowded_frame_subset_selection():
    # Frame 1: Disjoint players -> Normal
    # Frame 2: Overlapping players (IoU > 0.20) -> Crowded
    # Frame 3: High player count (>= 18) -> Crowded
    boxes_by_frame = {
        1: [
            (100.0, 100.0, 150.0, 200.0),
            (300.0, 300.0, 350.0, 400.0),
        ],
        2: [
            (100.0, 100.0, 150.0, 200.0),
            (110.0, 110.0, 160.0, 210.0),  # Heavy overlap
        ],
        3: [(float(i * 40), 100.0, float(i * 40 + 30), 200.0) for i in range(19)],
    }

    crowded_frames, stats = identify_crowded_frames(boxes_by_frame, min_overlapping_pairs=1, min_iou=0.20)

    assert 1 not in crowded_frames
    assert 2 in crowded_frames
    assert 3 in crowded_frames
    assert stats[1]["is_crowded"] is False
    assert stats[2]["is_crowded"] is True
    assert stats[2]["overlapping_pairs"] >= 1
    assert stats[2]["max_pairwise_iou"] > 0.20


def test_synthetic_identity_swap():
    # GT identity 1 tracked as ID 10 in frames 1-2, then incorrectly swapped to ID 20 in frames 3-5
    gt_boxes = [
        GroundTruthBox(frame=f, gt_id=1, x1=100.0, y1=100.0, width=50.0, height=100.0)
        for f in range(1, 6)
    ]
    tracker_boxes = [
        TrackerBox(frame=1, track_id=10, x1=100.0, y1=100.0, width=50.0, height=100.0),
        TrackerBox(frame=2, track_id=10, x1=100.0, y1=100.0, width=50.0, height=100.0),
        TrackerBox(frame=3, track_id=20, x1=100.0, y1=100.0, width=50.0, height=100.0),
        TrackerBox(frame=4, track_id=20, x1=100.0, y1=100.0, width=50.0, height=100.0),
        TrackerBox(frame=5, track_id=20, x1=100.0, y1=100.0, width=50.0, height=100.0),
    ]

    frame_stats = {
        f: {"player_count": 2, "max_pairwise_iou": 0.40, "is_crowded": True}
        for f in range(1, 6)
    }
    crowded_frames = {1, 2, 3, 4, 5}

    events = detect_id_switch_events(gt_boxes, tracker_boxes, crowded_frames, frame_stats, fps=25.0)

    assert len(events) == 1
    ev = events[0]
    assert ev.frame == 3
    assert ev.gt_id == 1
    assert ev.previous_tracker_id == 10
    assert ev.new_tracker_id == 20
    assert ev.crowded_frame is True
    assert ev.bbox_overlap == 0.40


def test_synthetic_track_fragmentation():
    # GT identity 1 active for 10 frames
    # Tracked by ID 10 for frames 1-3
    # Missed in frames 4-5
    # Tracked by ID 20 for frames 6-8
    # Tracked by ID 30 for frames 9-10
    gt_boxes = [
        GroundTruthBox(frame=f, gt_id=1, x1=100.0, y1=100.0, width=50.0, height=100.0)
        for f in range(1, 11)
    ]
    tracker_boxes = [
        TrackerBox(frame=1, track_id=10, x1=100.0, y1=100.0, width=50.0, height=100.0),
        TrackerBox(frame=2, track_id=10, x1=100.0, y1=100.0, width=50.0, height=100.0),
        TrackerBox(frame=3, track_id=10, x1=100.0, y1=100.0, width=50.0, height=100.0),
        # frames 4, 5 missing
        TrackerBox(frame=6, track_id=20, x1=100.0, y1=100.0, width=50.0, height=100.0),
        TrackerBox(frame=7, track_id=20, x1=100.0, y1=100.0, width=50.0, height=100.0),
        TrackerBox(frame=8, track_id=20, x1=100.0, y1=100.0, width=50.0, height=100.0),
        TrackerBox(frame=9, track_id=30, x1=100.0, y1=100.0, width=50.0, height=100.0),
        TrackerBox(frame=10, track_id=30, x1=100.0, y1=100.0, width=50.0, height=100.0),
    ]

    frag_records = compute_identity_fragmentation(gt_boxes, tracker_boxes)

    assert len(frag_records) == 1
    rec = frag_records[0]
    assert rec.gt_id == 1
    assert rec.number_of_tracker_ids == 3
    assert rec.number_of_fragments == 3
    assert rec.longest_continuous_track == 3
    assert rec.total_frames == 10


def test_occlusion_recovery_metric():
    gt_boxes = (
        [GroundTruthBox(frame=f, gt_id=1, x1=100.0, y1=100.0, width=50.0, height=100.0) for f in range(1, 15)]
        + [GroundTruthBox(frame=f, gt_id=2, x1=300.0, y1=300.0, width=50.0, height=100.0) for f in range(1, 15)]
    )

    # GT 1: tracked with ID 5 (frames 1-3), gap frames 4-8 (duration 5), recovered with ID 5 (frames 9-12)
    # GT 2: tracked with ID 8 (frames 1-3), gap frames 4-8 (duration 5), failed to recover -> ID 99 (frames 9-12)
    tracker_boxes = (
        [TrackerBox(frame=f, track_id=5, x1=100.0, y1=100.0, width=50.0, height=100.0) for f in (1, 2, 3)]
        + [TrackerBox(frame=f, track_id=5, x1=100.0, y1=100.0, width=50.0, height=100.0) for f in (9, 10, 11, 12)]
        + [TrackerBox(frame=f, track_id=8, x1=300.0, y1=300.0, width=50.0, height=100.0) for f in (1, 2, 3)]
        + [TrackerBox(frame=f, track_id=99, x1=300.0, y1=300.0, width=50.0, height=100.0) for f in (9, 10, 11, 12)]
    )

    records = compute_occlusion_recovery(gt_boxes, tracker_boxes, min_gap_frames=3)

    assert len(records) == 2
    r_rec = next(r for r in records if r.identity_before == 5)
    assert r_rec.identity_after == 5
    assert r_rec.identity_recovered is True
    assert r_rec.occlusion_duration == 5

    r_fail = next(r for r in records if r.identity_before == 8)
    assert r_fail.identity_after == 99
    assert r_fail.identity_recovered is False
    assert r_fail.occlusion_duration == 5


def test_metric_output_schema():
    metrics_path = Path("outputs/evaluation/metrics.json")
    metadata_path = Path("outputs/evaluation/evaluation_metadata.json")

    assert metrics_path.exists(), "outputs/evaluation/metrics.json must exist"
    assert metadata_path.exists(), "outputs/evaluation/evaluation_metadata.json must exist"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Check subset keys under summary
    summary = metrics.get("summary", metrics)
    for subset in ["Overall", "Normal", "Crowded/Occluded"]:
        assert subset in summary, f"Subset '{subset}' missing from metrics.json"
        sub_metrics = summary[subset]
        for metric_name in ["HOTA", "AssA", "MOTA", "IDSW", "Frag", "DetA", "DetRe", "DetPr"]:
            assert metric_name in sub_metrics, f"Metric '{metric_name}' missing from subset '{subset}'"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    required_meta = [
        "dataset",
        "dataset_version",
        "selected_sequence_ids",
        "split",
        "tracker",
        "detector",
        "evaluation_framework",
    ]
    for key in required_meta:
        assert key in metadata, f"Metadata key '{key}' missing from evaluation_metadata.json"
