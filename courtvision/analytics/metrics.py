"""
courtvision.analytics.metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Per-player physical analytics derived from a sequence of :class:`TrackedPlayer`
observations produced by the tracking pipeline.

Design constraints
------------------
* Only observations where *both* ``projection_valid`` and ``in_calibrated_zone``
  are ``True`` are used for metric calculations.  Uncalibrated image coordinates
  are never interpreted as real-world measurements.
* Time is derived from the ``timestamp`` field (seconds) when available and
  consistent; a frame-index + FPS fallback is used when timestamps are absent,
  zero, or non-monotonically increasing.
* A spatial noise deadband suppresses sub-centimetre jitter:  displacement steps
  smaller than ``min_displacement_m`` (default 0.05 m) are counted as zero.
* Occlusion gaps -- runs of consecutive *invalid* observations -- are detected
  and counted but do NOT contribute distance or speed samples.  No position
  imputation is performed.
* Sprint detection requires speed >= ``sprint_threshold_kmh`` sustained for at
  least ``sprint_min_duration_s`` seconds (default 1.0 s).
* All public results are JSON-serialisable (plain Python scalars / dicts / lists).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from courtvision.tracking.track_types import TrackedPlayer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SPRINT_THRESHOLD_KMH: float = 25.0
DEFAULT_SPRINT_MIN_DURATION_S: float = 1.0
DEFAULT_MIN_DISPLACEMENT_M: float = 0.05  # deadband: ignore jitter < 5 cm
_MS_TO_KMH: float = 3.6


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PlayerMetrics:
    """Physical analytics summary for a single persistent track ID.

    All distances and speeds are in metric units (km and km/h) and are computed
    exclusively from pitch-projected, calibrated-zone observations.

    Attributes
    ----------
    track_id:
        Persistent integer identifier shared across all frames for this player.
    distance_km:
        Total straight-line distance covered in kilometres, accumulated from
        consecutive valid pitch-coordinate steps (deadband applied).
    avg_speed_kmh:
        Mean of all instantaneous speed samples (km/h).  ``None`` if fewer than
        two valid consecutive observations with positive dt exist.
    top_speed_kmh:
        Maximum instantaneous speed sample (km/h).  ``None`` if same condition.
    sprint_count:
        Number of discrete sprint bouts where speed exceeded
        ``sprint_threshold_kmh`` for at least ``sprint_min_duration_s`` seconds.
    sprint_threshold_kmh:
        Threshold used for sprint detection (km/h).
    sprint_min_duration_s:
        Minimum sustained duration used for sprint detection (seconds).
    valid_samples:
        Number of valid pitch-projected observations used.
    total_samples:
        Total number of observations in the track (valid + invalid).
    coverage_ratio:
        Fraction of total observations that were projection-valid and
        in-calibrated-zone.  Values close to 1.0 indicate high-quality data.
    occlusion_gaps:
        Count of contiguous runs of invalid observations between two valid ones.
    time_span_s:
        Elapsed time in seconds between the first and last valid observation.
        ``None`` if fewer than two valid observations exist.
    """

    track_id: int
    distance_km: float = 0.0
    avg_speed_kmh: Optional[float] = None
    top_speed_kmh: Optional[float] = None
    sprint_count: int = 0
    sprint_threshold_kmh: float = DEFAULT_SPRINT_THRESHOLD_KMH
    sprint_min_duration_s: float = DEFAULT_SPRINT_MIN_DURATION_S
    valid_samples: int = 0
    total_samples: int = 0
    coverage_ratio: float = 0.0
    occlusion_gaps: int = 0
    time_span_s: Optional[float] = None
    # Private: cached speed samples (excluded from to_dict)
    _speed_samples: List[float] = field(default_factory=list, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary (excludes private fields)."""
        d = asdict(self)
        d.pop("_speed_samples", None)
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_timestamps(obs: List[TrackedPlayer], fps: float) -> List[float]:
    """Return a list of timestamps (seconds) aligned with *obs*.

    Strategy
    --------
    1. Use ``observation.timestamp`` directly if the sequence is strictly
       monotonically non-decreasing and the total span is > 0.
    2. Otherwise fall back to ``frame_index / fps``.
    """
    if not obs:
        return []

    raw = [o.timestamp for o in obs]
    span = raw[-1] - raw[0]
    monotonic = all(b >= a for a, b in zip(raw, raw[1:]))

    if monotonic and span > 0:
        return raw

    if fps <= 0:
        fps = 25.0
    return [float(o.frame) / fps for o in obs]


def _is_valid(obs: TrackedPlayer) -> bool:
    """True iff the observation carries a usable pitch projection."""
    return (
        obs.projection_valid
        and obs.in_calibrated_zone
        and obs.pitch_x is not None
        and obs.pitch_y is not None
        and math.isfinite(obs.pitch_x)
        and math.isfinite(obs.pitch_y)
    )


def _euclidean_m(x0: float, y0: float, x1: float, y1: float) -> float:
    """Euclidean distance in metres on the flat pitch plane."""
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def _count_sprints(
    speed_samples: List[Tuple[float, float, float]],
    threshold_kmh: float,
    min_duration_s: float,
) -> int:
    """Count discrete sprint bouts in a time-ordered speed series.

    Each sample carries the *interval duration* (dt) it represents so that a
    single long interval above threshold is correctly counted as a sprint even
    when only one speed sample exists.

    Parameters
    ----------
    speed_samples:
        List of ``(t_start, dt, speed_kmh)`` triples in chronological order,
        where *dt* is the duration of the interval this sample represents.
    threshold_kmh:
        Speed threshold to enter/exit a sprint.
    min_duration_s:
        Minimum cumulative duration (seconds) above threshold required to count
        a sprint bout.
    """
    sprints = 0
    in_sprint = False
    bout_duration: float = 0.0

    for _t, dt, spd in speed_samples:
        if spd >= threshold_kmh:
            in_sprint = True
            bout_duration += dt
        else:
            if in_sprint:
                if bout_duration >= min_duration_s:
                    sprints += 1
                in_sprint = False
                bout_duration = 0.0

    # Close any open sprint at end of sequence
    if in_sprint and bout_duration >= min_duration_s:
        sprints += 1

    return sprints


def _count_occlusion_gaps(track_obs: List[TrackedPlayer]) -> int:
    """Count runs of invalid observations sandwiched between two valid ones."""
    gaps = 0
    in_gap = False
    seen_valid = False

    for obs in track_obs:
        if _is_valid(obs):
            if in_gap:
                gaps += 1
            in_gap = False
            seen_valid = True
        else:
            if seen_valid:
                in_gap = True

    return gaps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_player_metrics(
    observations: Sequence[TrackedPlayer],
    fps: float = 25.0,
    sprint_threshold_kmh: float = DEFAULT_SPRINT_THRESHOLD_KMH,
    sprint_min_duration_s: float = DEFAULT_SPRINT_MIN_DURATION_S,
    min_displacement_m: float = DEFAULT_MIN_DISPLACEMENT_M,
) -> Dict[int, PlayerMetrics]:
    """Compute per-player physical analytics from a flat observation sequence.

    The function groups observations by ``track_id``, sorts each group by frame
    index (ascending), resolves timestamps, and iterates over consecutive pairs
    of valid observations to accumulate distance and speed.

    Parameters
    ----------
    observations:
        Flat sequence of :class:`~courtvision.tracking.track_types.TrackedPlayer`
        objects.  May contain multiple players and frames in any order.
    fps:
        Source video frame rate.  Used as timestamp fallback.  Default 25.
    sprint_threshold_kmh:
        Minimum speed (km/h) to register a sprint.  Default 25 km/h.
    sprint_min_duration_s:
        Minimum continuous duration (seconds) above threshold.  Default 1.0 s.
    min_displacement_m:
        Spatial deadband (metres) -- steps below this value are treated as zero.
        Default 0.05 m.

    Returns
    -------
    dict[int, PlayerMetrics]
        Mapping from ``track_id`` to a :class:`PlayerMetrics` instance.
        Returns an empty dict if *observations* is empty.
    """
    if not observations:
        return {}

    # ------------------------------------------------------------------
    # 1. Group by track_id, sort by frame then timestamp
    # ------------------------------------------------------------------
    grouped: Dict[int, List[TrackedPlayer]] = {}
    for obs in observations:
        grouped.setdefault(obs.track_id, []).append(obs)

    for tid in grouped:
        grouped[tid].sort(key=lambda o: (o.frame, o.timestamp))

    # ------------------------------------------------------------------
    # 2. Compute metrics per track
    # ------------------------------------------------------------------
    results: Dict[int, PlayerMetrics] = {}

    for track_id, track_obs in grouped.items():
        metrics = PlayerMetrics(
            track_id=track_id,
            sprint_threshold_kmh=sprint_threshold_kmh,
            sprint_min_duration_s=sprint_min_duration_s,
        )
        metrics.total_samples = len(track_obs)

        # Resolve timestamps for the full track sequence
        timestamps = _resolve_timestamps(track_obs, fps)

        # Collect valid observations with their resolved timestamps
        valid_with_ts: List[Tuple[float, float, float]] = []  # (t, px, py)
        for obs, t in zip(track_obs, timestamps):
            if _is_valid(obs):
                valid_with_ts.append((t, float(obs.pitch_x), float(obs.pitch_y)))  # type: ignore[arg-type]

        metrics.valid_samples = len(valid_with_ts)
        metrics.coverage_ratio = (
            metrics.valid_samples / metrics.total_samples
            if metrics.total_samples > 0
            else 0.0
        )
        metrics.occlusion_gaps = _count_occlusion_gaps(track_obs)

        if metrics.valid_samples < 2:
            results[track_id] = metrics
            continue

        # Time span
        metrics.time_span_s = valid_with_ts[-1][0] - valid_with_ts[0][0]

        # ------------------------------------------------------------------
        # 3. Accumulate distance and build speed series
        # ------------------------------------------------------------------
        total_distance_m = 0.0
        # (t_start, dt, speed_kmh) -- dt is the interval length for sprint timing
        speed_series: List[Tuple[float, float, float]] = []

        for (t0, px0, py0), (t1, px1, py1) in zip(valid_with_ts, valid_with_ts[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue  # skip duplicate timestamps

            dist_m = _euclidean_m(px0, py0, px1, py1)

            # Apply deadband
            if dist_m < min_displacement_m:
                dist_m = 0.0

            total_distance_m += dist_m

            speed_ms = dist_m / dt
            speed_kmh = speed_ms * _MS_TO_KMH
            speed_series.append((t0, dt, speed_kmh))

        metrics.distance_km = total_distance_m / 1000.0

        if speed_series:
            speeds = [s for _, _, s in speed_series]
            metrics.avg_speed_kmh = sum(speeds) / len(speeds)
            metrics.top_speed_kmh = max(speeds)
            metrics.sprint_count = _count_sprints(
                speed_series,
                sprint_threshold_kmh,
                sprint_min_duration_s,
            )
            metrics._speed_samples = speeds

        results[track_id] = metrics

    return results


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def metrics_to_json_list(metrics_map: Dict[int, PlayerMetrics]) -> List[Dict[str, Any]]:
    """Convert a ``compute_player_metrics`` result to a JSON-serialisable list.

    Sorted by ``track_id`` for deterministic output.
    """
    return [metrics_map[tid].to_dict() for tid in sorted(metrics_map)]
