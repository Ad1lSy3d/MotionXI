import numpy as np
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


def _make_textured_frame(shift_x: float = 0.0, shift_y: float = 0.0) -> np.ndarray:
    """Generate synthetic high-contrast textured frame with non-periodic features and known translation."""
    import cv2

    rng = np.random.RandomState(42)
    base = rng.randint(40, 220, (400, 400, 3), dtype=np.uint8)
    # Add non-periodic distinct geometrical anchors
    cv2.circle(base, (90, 110), 18, (255, 0, 0), -1)
    cv2.rectangle(base, (180, 140), (260, 230), (0, 255, 0), -1)
    cv2.circle(base, (290, 280), 25, (0, 0, 255), -1)
    cv2.line(base, (50, 320), (350, 320), (255, 255, 0), 4)

    if shift_x == 0.0 and shift_y == 0.0:
        return base
    M = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
    return cv2.warpAffine(base, M, (400, 400), borderMode=cv2.BORDER_REFLECT)


def test_camera_motion_transform_direction():
    """Verify that estimated affine transform correctly maps points in the direction of scene motion."""
    from courtvision.tracking.camera_motion import CameraMotionCompensator

    cmc = CameraMotionCompensator()
    f0 = _make_textured_frame(shift_x=0.0, shift_y=0.0)
    f1 = _make_textured_frame(shift_x=15.0, shift_y=-5.0)

    assert cmc.estimate_motion(f0) is None  # Initial frame establishes reference
    M = cmc.estimate_motion(f1)
    assert M is not None

    # Translation components should match (+15, -5)
    assert M[0, 2] == pytest.approx(15.0, abs=1.0)
    assert M[1, 2] == pytest.approx(-5.0, abs=1.0)

    # Point at (100, 100) should transform to ~(115, 95)
    pt_x, pt_y = CameraMotionCompensator.transform_point(100.0, 100.0, M)
    assert pt_x == pytest.approx(115.0, abs=1.0)
    assert pt_y == pytest.approx(95.0, abs=1.0)

    # Bounding box should shift by the same translation
    bbox = (100.0, 100.0, 140.0, 200.0)
    new_bbox = CameraMotionCompensator.transform_bbox(bbox, M)
    assert new_bbox[0] == pytest.approx(115.0, abs=1.0)
    assert new_bbox[1] == pytest.approx(95.0, abs=1.0)
    assert new_bbox[2] == pytest.approx(155.0, abs=1.0)
    assert new_bbox[3] == pytest.approx(195.0, abs=1.0)


def test_point_prediction_under_synthetic_camera_translation():
    """Verify that KMT predicts track location incorporating camera motion when enabled."""
    tracker = KinematicMemoryTracker(enable_camera_motion=True)

    f0 = _make_textured_frame(shift_x=0.0, shift_y=0.0)
    f1 = _make_textured_frame(shift_x=15.0, shift_y=0.0)

    # Frame 0: detection at [100, 100, 140, 200], foot at (120, 200)
    first_tracks, _ = tracker.update(
        [{"bbox": [100.0, 100.0, 140.0, 200.0], "confidence": 0.9}],
        frame_idx=0,
        frame=f0,
    )
    assert len(first_tracks) == 1
    track_id = first_tracks[0].track_id

    # Frame 1: missing detection (player occluded), camera translated by +15px
    tracker.update([], frame_idx=1, frame=f1)

    track = tracker.tracks[track_id]
    # Foot point and bbox should be shifted by camera motion (+15px)
    assert track.foot_x == pytest.approx(135.0, abs=1.5)
    assert track.bbox[0] == pytest.approx(115.0, abs=1.5)


def test_unreliable_motion_fallback():
    """Verify that blank/uniform frames or extreme warps trigger safe identity fallback."""
    import numpy as np
    from courtvision.tracking.camera_motion import CameraMotionCompensator

    cmc = CameraMotionCompensator(max_translation_px=50.0)
    # 1. Blank uniform frames (no corners detected)
    blank0 = np.full((400, 400, 3), 128, dtype=np.uint8)
    blank1 = np.full((400, 400, 3), 128, dtype=np.uint8)
    assert cmc.estimate_motion(blank0) is None
    assert cmc.estimate_motion(blank1) is None
    stats = cmc.get_statistics()
    assert stats["failed_frames"] >= 1

    # 2. None frame
    assert cmc.estimate_motion(None) is None

    # 3. KMT graceful fallback under blank frames
    tracker = KinematicMemoryTracker(enable_camera_motion=True)
    first_tracks, _ = tracker.update(
        [{"bbox": [100.0, 100.0, 140.0, 200.0], "confidence": 0.9}],
        frame_idx=0,
        frame=blank0,
    )
    track_id = first_tracks[0].track_id
    tracker.update([], frame_idx=1, frame=blank1)
    # Track foot_x remains intact at 120.0 (no crash, no wild jumps)
    assert tracker.tracks[track_id].foot_x == pytest.approx(120.0)


def test_unchanged_behavior_when_compensation_disabled():
    """Verify that when enable_camera_motion=False, behavior is strictly identical to baseline."""
    tracker_disabled = KinematicMemoryTracker(enable_camera_motion=False)
    tracker_enabled = KinematicMemoryTracker(enable_camera_motion=True)

    f0 = _make_textured_frame(shift_x=0)
    f1 = _make_textured_frame(shift_x=25)

    # When enable_camera_motion=False, passing frame does not invoke compensator
    assert tracker_disabled.cmc is None
    assert tracker_enabled.cmc is not None

    t0_dis, _ = tracker_disabled.update(
        [{"bbox": [100.0, 100.0, 140.0, 200.0], "confidence": 0.9}],
        frame_idx=0,
        frame=f0,
    )
    # Tracker disabled receives f1 but ignores it
    tracker_disabled.update([], frame_idx=1, frame=f1)

    # Foot position for disabled tracker should not have shifted by 25px
    assert tracker_disabled.tracks[t0_dis[0].track_id].foot_x == pytest.approx(120.0)

