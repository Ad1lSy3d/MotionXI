"""
tests/test_analytics.py
~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for courtvision.analytics.metrics.

Coverage
--------
1.  Empty input returns empty dict.
2.  Single observation (< 2 valid samples) returns metrics with no speed/distance.
3.  Two valid observations: distance, speed, time_span computed correctly.
4.  Invalid projections are excluded; only valid calibrated-zone observations used.
5.  Spatial deadband suppresses jitter below min_displacement_m.
6.  Timestamp fallback: non-monotonic timestamps -> frame/fps derivation.
7.  Sprint detection: single bout counted when threshold and duration met.
8.  Sprint detection: partial bout below min_duration_s is NOT counted.
9.  Multiple players: metrics are independent per track_id.
10. Occlusion gap detection: gaps between valid observations counted correctly.
11. to_dict() is JSON-serialisable and excludes private _speed_samples field.
12. metrics_to_json_list() returns deterministically ordered list.
"""

import math
import json
import pytest

from courtvision.analytics.metrics import (
    DEFAULT_MIN_DISPLACEMENT_M,
    DEFAULT_SPRINT_MIN_DURATION_S,
    DEFAULT_SPRINT_THRESHOLD_KMH,
    PlayerMetrics,
    compute_player_metrics,
    metrics_to_json_list,
)
from courtvision.tracking.track_types import TrackedPlayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs(
    frame: int,
    track_id: int,
    *,
    pitch_x: float = 10.0,
    pitch_y: float = 30.0,
    projection_valid: bool = True,
    in_calibrated_zone: bool = True,
    timestamp: float = -1.0,
    fps: float = 25.0,
) -> TrackedPlayer:
    """Convenience factory for test observations.

    timestamp defaults to frame / fps when not specified.
    """
    t = timestamp if timestamp >= 0 else frame / fps
    return TrackedPlayer.create(
        frame=frame,
        timestamp=t,
        track_id=track_id,
        class_id=0,
        class_name="player",
        confidence=0.9,
        x1=0.0,
        y1=0.0,
        x2=50.0,
        y2=100.0,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        projection_valid=projection_valid,
        in_calibrated_zone=in_calibrated_zone,
    )


# ---------------------------------------------------------------------------
# Test 1: Empty input
# ---------------------------------------------------------------------------


def test_empty_observations_returns_empty_dict():
    result = compute_player_metrics([])
    assert result == {}


# ---------------------------------------------------------------------------
# Test 2: Single valid observation (< 2 samples)
# ---------------------------------------------------------------------------


def test_single_valid_observation_has_no_speed_or_distance():
    obs = [_obs(frame=0, track_id=1)]
    result = compute_player_metrics(obs, fps=25.0)

    assert 1 in result
    m = result[1]
    assert m.distance_km == 0.0
    assert m.avg_speed_kmh is None
    assert m.top_speed_kmh is None
    assert m.sprint_count == 0
    assert m.valid_samples == 1
    assert m.total_samples == 1
    assert m.time_span_s is None


# ---------------------------------------------------------------------------
# Test 3: Two valid observations -- exact distance and speed
# ---------------------------------------------------------------------------


def test_two_valid_observations_correct_distance_and_speed():
    """
    Player moves 10 m in 1 second -> 10 m/s = 36 km/h.
    Distance = 10 m = 0.010 km.
    """
    fps = 25.0
    obs = [
        _obs(frame=0, track_id=1, pitch_x=0.0, pitch_y=0.0, fps=fps),
        _obs(frame=25, track_id=1, pitch_x=10.0, pitch_y=0.0, fps=fps),
    ]
    result = compute_player_metrics(obs, fps=fps)

    m = result[1]
    assert m.distance_km == pytest.approx(0.010, abs=1e-6)
    assert m.avg_speed_kmh == pytest.approx(36.0, abs=1e-4)
    assert m.top_speed_kmh == pytest.approx(36.0, abs=1e-4)
    assert m.time_span_s == pytest.approx(1.0, abs=1e-6)
    assert m.valid_samples == 2
    assert m.total_samples == 2
    assert m.coverage_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 4: Invalid projections excluded
# ---------------------------------------------------------------------------


def test_invalid_observations_are_excluded_from_metrics():
    """
    Frame 0: invalid (should be ignored).
    Frame 1: valid at (0, 0).
    Frame 2: invalid (should be ignored).
    Frame 3: valid at (5, 0) -> 5 m / (2/25 s) = 62.5 m/s = 225 km/h.
    """
    fps = 25.0
    obs = [
        _obs(frame=0, track_id=7, pitch_x=99.0, pitch_y=99.0, projection_valid=False, in_calibrated_zone=False, fps=fps),
        _obs(frame=1, track_id=7, pitch_x=0.0, pitch_y=0.0, fps=fps),
        _obs(frame=2, track_id=7, pitch_x=99.0, pitch_y=99.0, projection_valid=False, fps=fps),
        _obs(frame=3, track_id=7, pitch_x=5.0, pitch_y=0.0, fps=fps),
    ]
    result = compute_player_metrics(obs, fps=fps)

    m = result[7]
    assert m.valid_samples == 2
    assert m.total_samples == 4
    assert m.coverage_ratio == pytest.approx(0.5)
    # Distance = 5 m
    assert m.distance_km == pytest.approx(0.005, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 5: Spatial deadband
# ---------------------------------------------------------------------------


def test_deadband_suppresses_sub_threshold_displacement():
    """
    Player is almost stationary: moves 0.03 m (< 0.05 m default deadband).
    Expect distance = 0.0 km (deadband applied).
    """
    fps = 25.0
    obs = [
        _obs(frame=0, track_id=2, pitch_x=0.0, pitch_y=0.0, fps=fps),
        _obs(frame=25, track_id=2, pitch_x=0.03, pitch_y=0.0, fps=fps),  # 3 cm step
    ]
    result = compute_player_metrics(obs, fps=fps, min_displacement_m=0.05)

    m = result[2]
    assert m.distance_km == pytest.approx(0.0, abs=1e-9)
    assert m.avg_speed_kmh == pytest.approx(0.0, abs=1e-6)
    assert m.top_speed_kmh == pytest.approx(0.0, abs=1e-6)


def test_deadband_does_not_suppress_above_threshold_displacement():
    """Displacement > deadband should be accumulated normally."""
    fps = 25.0
    obs = [
        _obs(frame=0, track_id=3, pitch_x=0.0, pitch_y=0.0, fps=fps),
        _obs(frame=25, track_id=3, pitch_x=0.10, pitch_y=0.0, fps=fps),  # 10 cm step
    ]
    result = compute_player_metrics(obs, fps=fps, min_displacement_m=0.05)

    m = result[3]
    assert m.distance_km == pytest.approx(0.0001, abs=1e-7)


# ---------------------------------------------------------------------------
# Test 6: Timestamp fallback
# ---------------------------------------------------------------------------


def test_timestamp_fallback_used_when_timestamps_non_monotonic():
    """
    Provide observations with timestamps that are NOT monotonically increasing.
    The function should fall back to frame / fps for time derivation.
    fps=10, so frame 0 -> t=0.0 s, frame 10 -> t=1.0 s.
    Movement = 10 m in 1 s -> 36 km/h.
    """
    fps = 10.0
    bad_ts = 99.0  # non-monotonic (frame 0 gets a later timestamp than frame 10)
    obs = [
        TrackedPlayer.create(
            frame=0, timestamp=bad_ts, track_id=5,
            class_id=0, class_name="player", confidence=0.9,
            x1=0.0, y1=0.0, x2=50.0, y2=100.0,
            pitch_x=0.0, pitch_y=0.0,
            projection_valid=True, in_calibrated_zone=True,
        ),
        TrackedPlayer.create(
            frame=10, timestamp=0.0, track_id=5,  # ts goes BACKWARDS -> non-monotonic
            class_id=0, class_name="player", confidence=0.9,
            x1=0.0, y1=0.0, x2=50.0, y2=100.0,
            pitch_x=10.0, pitch_y=0.0,
            projection_valid=True, in_calibrated_zone=True,
        ),
    ]
    result = compute_player_metrics(obs, fps=fps)

    m = result[5]
    # After fallback: dt = 10 / 10 = 1.0 s, dist = 10 m
    assert m.distance_km == pytest.approx(0.010, abs=1e-6)
    assert m.avg_speed_kmh == pytest.approx(36.0, abs=1e-3)
    assert m.time_span_s == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 7: Sprint detection -- full sprint counted
# ---------------------------------------------------------------------------


def test_sprint_counted_when_speed_exceeds_threshold_for_required_duration():
    """
    Sprint at 30 km/h for 2 seconds (> 1 s minimum) -> 1 sprint.
    30 km/h = 30/3.6 m/s = 8.333... m/s.
    In 2 s: displacement = 16.667 m.
    Use fps=25 for simple frame arithmetic.
    """
    fps = 25.0
    speed_ms = 30.0 / 3.6  # 8.333 m/s
    dt = 2.0  # seconds
    dist_m = speed_ms * dt  # 16.667 m

    obs = [
        _obs(frame=0, track_id=9, pitch_x=0.0, pitch_y=0.0, fps=fps),
        _obs(frame=int(fps * dt), track_id=9, pitch_x=dist_m, pitch_y=0.0, fps=fps),
    ]
    result = compute_player_metrics(
        obs, fps=fps, sprint_threshold_kmh=25.0, sprint_min_duration_s=1.0
    )

    m = result[9]
    assert m.sprint_count == 1
    assert m.top_speed_kmh == pytest.approx(30.0, abs=0.02)


# ---------------------------------------------------------------------------
# Test 8: Sprint NOT counted when below minimum duration
# ---------------------------------------------------------------------------


def test_sprint_not_counted_when_below_minimum_duration():
    """
    Speed = 30 km/h but sustained for only 0.5 s (< 1 s minimum).
    Sprint should NOT be counted.
    """
    fps = 25.0
    speed_ms = 30.0 / 3.6
    dt = 0.5  # seconds -- less than sprint_min_duration_s
    dist_m = speed_ms * dt

    obs = [
        _obs(frame=0, track_id=10, pitch_x=0.0, pitch_y=0.0, fps=fps),
        _obs(frame=int(fps * dt), track_id=10, pitch_x=dist_m, pitch_y=0.0, fps=fps),
    ]
    result = compute_player_metrics(
        obs, fps=fps, sprint_threshold_kmh=25.0, sprint_min_duration_s=1.0
    )

    m = result[10]
    assert m.sprint_count == 0


# ---------------------------------------------------------------------------
# Test 9: Multiple players are independent
# ---------------------------------------------------------------------------


def test_multiple_players_metrics_are_independent():
    """
    Two players with different tracks should produce independent metrics.
    """
    fps = 25.0
    obs = [
        # Player A: 5 m in 1 s
        _obs(frame=0, track_id=1, pitch_x=0.0, pitch_y=0.0, fps=fps),
        _obs(frame=25, track_id=1, pitch_x=5.0, pitch_y=0.0, fps=fps),
        # Player B: 20 m in 1 s
        _obs(frame=0, track_id=2, pitch_x=0.0, pitch_y=0.0, fps=fps),
        _obs(frame=25, track_id=2, pitch_x=20.0, pitch_y=0.0, fps=fps),
    ]
    result = compute_player_metrics(obs, fps=fps)

    assert 1 in result
    assert 2 in result
    assert result[1].distance_km == pytest.approx(0.005, abs=1e-6)
    assert result[2].distance_km == pytest.approx(0.020, abs=1e-6)
    assert result[1].avg_speed_kmh != result[2].avg_speed_kmh


# ---------------------------------------------------------------------------
# Test 10: Occlusion gap detection
# ---------------------------------------------------------------------------


def test_occlusion_gap_counted_between_valid_observations():
    """
    Sequence: valid, invalid, invalid, valid -> 1 gap.
    """
    fps = 25.0
    obs = [
        _obs(frame=0, track_id=11, pitch_x=0.0, pitch_y=0.0, fps=fps),
        _obs(frame=1, track_id=11, projection_valid=False, in_calibrated_zone=False, fps=fps),
        _obs(frame=2, track_id=11, projection_valid=False, fps=fps),
        _obs(frame=3, track_id=11, pitch_x=1.0, pitch_y=0.0, fps=fps),
    ]
    result = compute_player_metrics(obs, fps=fps)

    m = result[11]
    assert m.occlusion_gaps == 1


def test_trailing_invalid_observations_not_counted_as_gap():
    """
    Sequence: valid, valid, invalid -> 0 gaps (trailing invalids don't count).
    """
    fps = 25.0
    obs = [
        _obs(frame=0, track_id=12, pitch_x=0.0, pitch_y=0.0, fps=fps),
        _obs(frame=1, track_id=12, pitch_x=1.0, pitch_y=0.0, fps=fps),
        _obs(frame=2, track_id=12, projection_valid=False, fps=fps),
    ]
    result = compute_player_metrics(obs, fps=fps)

    m = result[12]
    assert m.occlusion_gaps == 0


def test_multiple_occlusion_gaps_counted():
    """
    valid, invalid, valid, invalid, valid -> 2 gaps.
    """
    fps = 25.0
    obs = [
        _obs(frame=0, track_id=13, pitch_x=0.0, pitch_y=0.0, fps=fps),
        _obs(frame=1, track_id=13, projection_valid=False, fps=fps),
        _obs(frame=2, track_id=13, pitch_x=1.0, pitch_y=0.0, fps=fps),
        _obs(frame=3, track_id=13, projection_valid=False, fps=fps),
        _obs(frame=4, track_id=13, pitch_x=2.0, pitch_y=0.0, fps=fps),
    ]
    result = compute_player_metrics(obs, fps=fps)

    m = result[13]
    assert m.occlusion_gaps == 2


# ---------------------------------------------------------------------------
# Test 11: to_dict() JSON-serialisable and excludes private fields
# ---------------------------------------------------------------------------


def test_to_dict_is_json_serialisable_and_excludes_private():
    fps = 25.0
    obs = [
        _obs(frame=0, track_id=20, pitch_x=0.0, pitch_y=0.0, fps=fps),
        _obs(frame=25, track_id=20, pitch_x=5.0, pitch_y=0.0, fps=fps),
    ]
    result = compute_player_metrics(obs, fps=fps)
    d = result[20].to_dict()

    # Must not raise
    json_str = json.dumps(d)
    assert json_str  # non-empty

    # Private field must be absent
    assert "_speed_samples" not in d

    # Required public fields
    for key in [
        "track_id",
        "distance_km",
        "avg_speed_kmh",
        "top_speed_kmh",
        "sprint_count",
        "sprint_threshold_kmh",
        "sprint_min_duration_s",
        "valid_samples",
        "total_samples",
        "coverage_ratio",
        "occlusion_gaps",
        "time_span_s",
    ]:
        assert key in d, f"Missing key in to_dict(): {key}"


# ---------------------------------------------------------------------------
# Test 12: metrics_to_json_list() deterministic ordering
# ---------------------------------------------------------------------------


def test_metrics_to_json_list_sorted_by_track_id():
    fps = 25.0
    obs = [
        _obs(frame=0, track_id=30, fps=fps),
        _obs(frame=25, track_id=30, pitch_x=1.0, fps=fps),
        _obs(frame=0, track_id=10, fps=fps),
        _obs(frame=25, track_id=10, pitch_x=2.0, fps=fps),
        _obs(frame=0, track_id=20, fps=fps),
        _obs(frame=25, track_id=20, pitch_x=3.0, fps=fps),
    ]
    result = compute_player_metrics(obs, fps=fps)
    as_list = metrics_to_json_list(result)

    assert len(as_list) == 3
    ids = [entry["track_id"] for entry in as_list]
    assert ids == sorted(ids)
    assert ids == [10, 20, 30]
