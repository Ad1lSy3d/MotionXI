import numpy as np
import pytest
from courtvision.detection.detector import Detection
from courtvision.projection.homography import PitchHomography
from courtvision.tracking.bytetrack_tracker import FootballByteTracker
from courtvision.tracking.track_types import TrackState, TrackedPlayer


def test_tracker_initialization():
    tracker = FootballByteTracker(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )
    config = tracker.get_config()
    assert config["tracker_name"] == "ByteTrack"
    assert config["track_activation_threshold"] == 0.25
    assert config["lost_track_buffer"] == 30
    assert config["frame_rate"] == 30


def test_track_id_persistence_across_consecutive_frames():
    homography = PitchHomography("data/homography_matrix.json")
    tracker = FootballByteTracker(homography=homography)

    # Frame 0: Single player at (100, 200, 160, 320)
    det0 = [
        Detection.create(
            frame=0,
            timestamp=0.0,
            class_id=0,
            class_name="player",
            confidence=0.92,
            x1=100.0,
            y1=200.0,
            x2=160.0,
            y2=320.0,
        )
    ]
    tracks0, _ = tracker.update(det0, frame_idx=0, fps=30.0)
    assert len(tracks0) == 1
    t0_id = tracks0[0].track_id

    # Frame 1: Same player moved slightly to (103, 201, 163, 321)
    det1 = [
        Detection.create(
            frame=1,
            timestamp=0.0333,
            class_id=0,
            class_name="player",
            confidence=0.91,
            x1=103.0,
            y1=201.0,
            x2=163.0,
            y2=321.0,
        )
    ]
    tracks1, _ = tracker.update(det1, frame_idx=1, fps=30.0)
    assert len(tracks1) == 1
    t1_id = tracks1[0].track_id

    # Track ID MUST persist
    assert t0_id == t1_id
    assert tracks1[0].frame == 1
    assert tracks1[0].foot_x == pytest.approx((103.0 + 163.0) / 2.0)
    assert tracks1[0].foot_y == pytest.approx(321.0)


def test_tracked_player_schema_and_serialization():
    tp = TrackedPlayer.create(
        frame=10,
        timestamp=0.333,
        track_id=4,
        class_id=0,
        class_name="player",
        confidence=0.89,
        x1=50.0,
        y1=100.0,
        x2=110.0,
        y2=250.0,
        pitch_x=12.5,
        pitch_y=34.0,
        projection_valid=True,
        in_calibrated_zone=True,
        state=TrackState.ACTIVE.value,
    )
    d = tp.to_dict()

    required_fields = [
        "frame",
        "timestamp",
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "foot_x",
        "foot_y",
        "pitch_x",
        "pitch_y",
        "projection_valid",
        "in_calibrated_zone",
        "state",
    ]
    for field in required_fields:
        assert field in d, f"Missing required track field: {field}"

    assert d["track_id"] == 4
    assert d["width"] == 60.0
    assert d["height"] == 150.0
    assert d["foot_x"] == 80.0
    assert d["foot_y"] == 250.0
    assert d["projection_valid"] is True
    assert d["state"] == "active"


def test_track_statistics_and_summary_generation():
    tracker = FootballByteTracker()

    # Feed 3 frames with 2 tracks
    for f in range(3):
        dets = [
            Detection.create(f, f / 30.0, 0, "player", 0.9, 100 + f, 200, 150 + f, 300),
            Detection.create(f, f / 30.0, 0, "player", 0.85, 400 + f, 200, 450 + f, 300),
        ]
        tracker.update(dets, frame_idx=f, fps=30.0)

    stats = tracker.get_track_statistics()
    assert stats["total_unique_track_ids"] == 2
    assert stats["average_track_length_frames"] == 3.0
    assert stats["max_track_length_frames"] == 3
    assert stats["average_active_tracks_per_frame"] == 2.0

    summary = tracker.get_track_summary_records()
    assert len(summary) == 2
    assert summary[0]["track_length"] == 3
    assert summary[0]["first_frame"] == 0
    assert summary[0]["last_frame"] == 2


def test_trail_history_buffer():
    tracker = FootballByteTracker(trail_length=5)
    for f in range(8):
        dets = [
            Detection.create(f, f / 30.0, 0, "player", 0.9, 100 + f * 5, 200, 150 + f * 5, 300),
        ]
        tracks, _ = tracker.update(dets, frame_idx=f, fps=30.0)

    t_id = tracks[0].track_id
    trail = tracker.trail_history[t_id]
    # Max len should be 5
    assert len(trail) == 5
    # Last point should match frame 7 foot position
    expected_last_foot_x = (100 + 7 * 5 + 150 + 7 * 5) / 2.0
    assert trail[-1][0] == pytest.approx(expected_last_foot_x)
