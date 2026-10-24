import pytest

from courtvision.detection.detector import Detection
from courtvision.tracking.kmt_tracker import (
    KinematicMemoryTracker,
    KinematicTrackState,
)


def make_detection(x1, y1=100.0, x2=None, y2=200.0):
    if x2 is None:
        x2 = x1 + 40.0
    return Detection.create(
        frame=0,
        timestamp=0.0,
        class_id=0,
        class_name="player",
        confidence=0.9,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def test_overlapping_detections_activate_cluster_state():
    tracker = KinematicMemoryTracker(cluster_iou_threshold=0.35)

    tracks, _ = tracker.update(
        [
            make_detection(0.0, y1=0.0, x2=100.0, y2=100.0),
            make_detection(40.0, y1=0.0, x2=140.0, y2=100.0),
        ],
        frame_idx=0,
    )

    assert len(tracks) == 2
    assert {track.state for track in tracks} == {
        KinematicTrackState.IN_CLUSTER.value
    }
    assert all(track.in_cluster for track in tracker.tracks.values())


def test_cluster_freezes_cached_motion_estimates():
    tracker = KinematicMemoryTracker()
    first, _ = tracker.update([make_detection(100.0)], frame_idx=0)
    track_id = first[0].track_id
    tracker.update([make_detection(110.0)], frame_idx=1)

    tracker.update(
        [
            make_detection(112.0),
            make_detection(120.0),
        ],
        frame_idx=2,
    )

    track = tracker.tracks[track_id]
    assert track.state == KinematicTrackState.IN_CLUSTER
    assert track.velocity == pytest.approx((10.0, 0.0))
    assert track.cached_velocity == pytest.approx((10.0, 0.0))
    assert track.bbox == pytest.approx((110.0, 100.0, 150.0, 200.0))


def test_ghost_position_uses_cached_velocity():
    tracker = KinematicMemoryTracker(min_ghost_frames=6, max_ghost_frames=25)
    first, _ = tracker.update([make_detection(100.0)], frame_idx=0)
    track_id = first[0].track_id

    tracker.update([make_detection(110.0)], frame_idx=1)
    for frame_idx in range(2, 7):
        tracks, _ = tracker.update([], frame_idx=frame_idx)
        assert not tracks

    tracks, _ = tracker.update([], frame_idx=7)

    assert len(tracks) == 1
    assert tracks[0].track_id == track_id
    assert tracks[0].state == KinematicTrackState.GHOST.value
    assert tracks[0].foot_x == pytest.approx(190.0)


def test_speed_gate_rejects_impossible_reappearance():
    tracker = KinematicMemoryTracker(max_implied_speed_px_per_frame=120.0)
    first, _ = tracker.update([make_detection(100.0)], frame_idx=0)
    original_track_id = first[0].track_id

    tracks, _ = tracker.update([make_detection(1100.0)], frame_idx=7)

    detected_track = next(track for track in tracks if track.state != "ghost")
    ghost_track = next(track for track in tracks if track.state == "ghost")
    assert detected_track.track_id != original_track_id
    assert ghost_track.track_id == original_track_id


def test_accepts_detection_dictionary_and_preserves_track_id():
    tracker = KinematicMemoryTracker()
    first, _ = tracker.update(
        [{"bbox": [100.0, 100.0, 140.0, 200.0], "confidence": 0.8}],
        frame_idx=0,
    )
    second, _ = tracker.update(
        [{"x1": 102.0, "y1": 100.0, "x2": 142.0, "y2": 200.0}],
        frame_idx=1,
    )

    assert second[0].track_id == first[0].track_id
    assert second[0].foot_x == pytest.approx(122.0)
