"""
courtvision.projection.homography
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Project 2D camera pixel coordinates (u, v) into 2D metric pitch coordinates (X, Y).

Coordinate convention (105 m × 68 m FIFA pitch)
------------------------------------------------
Origin (0, 0) at the left goal-line / near touchline intersection.
  X axis  — along pitch length, 0 → 105 m (left goal → right goal)
  Y axis  — along pitch width,  0 → 68 m  (near touchline → far touchline)

Loading formats
---------------
``PitchHomography`` transparently loads two JSON formats:

1. **Legacy (bare matrix)**: a JSON array-of-arrays ``[[...],[...],[...]]``.
   Treated as a penalty-box calibration.  calibrated_zone defaults to the
   18-yard box bounds.

2. **Rich (v1)**: written by ``HomographyCalibrator.save()``.  Contains key
   ``"format": "courtvision_homography_v1"`` along with the matrix under ``"H"``,
   calibration scope, coverage metrics, and landmark correspondences.  The
   ``calibrated_zone`` is derived from inlier landmark coverage; for full-pitch
   calibrations the entire pitch is considered the calibrated zone.

Limitations
-----------
A single static homography assumes a flat pitch and a static camera.  Camera pan,
tilt, zoom, or cuts will introduce errors not captured here.  Use
``in_calibrated_zone`` to flag observations that are outside the region
validated during calibration.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class PitchHomography:
    """Project camera pixel coordinates to metric pitch coordinates.

    Parameters
    ----------
    matrix_source:
        Path to a JSON file, a 3×3 numpy array, or a nested list.
        Two JSON formats are accepted:
        - Legacy: bare ``[[...],[...],[...]]`` array
        - Rich v1: ``{"format": "courtvision_homography_v1", "H": [[...]], ...}``
    pitch_length:
        Pitch length in meters (default 105.0).
    pitch_width:
        Pitch width in meters (default 68.0).
    box_depth:
        Depth of the penalty box (default 16.5 m, used for legacy mode
        ``is_within_calibrated_zone``).
    box_y_min, box_y_max:
        Width extents of the penalty box for legacy calibrated-zone checks.
    """

    def __init__(
        self,
        matrix_source: Union[str, Path, np.ndarray, List[List[float]]],
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        box_depth: float = 16.5,
        box_y_min: float = 13.84,
        box_y_max: float = 54.16,
    ) -> None:
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)
        self.box_depth = float(box_depth)
        self.box_y_min = float(box_y_min)
        self.box_y_max = float(box_y_max)

        # Populated during loading; override defaults for rich calibrations.
        self._calibration_metadata: Dict[str, Any] = {}
        self._is_full_pitch: bool = False
        self._calibration_scope: str = "penalty_box_region"
        self._coverage_score: float = 0.0
        self._validation: Dict[str, Any] = {}
        self._correspondences: List[Dict[str, Any]] = []
        self._camera_motion_warning: str = ""

        if isinstance(matrix_source, (str, Path)):
            self.source_path = str(matrix_source)
            self.H, rich_meta = self._load_json(Path(matrix_source))
            if rich_meta is not None:
                self._apply_rich_metadata(rich_meta)
        elif isinstance(matrix_source, (list, tuple)):
            self.source_path = "in_memory"
            self.H = np.array(matrix_source, dtype=np.float64)
        elif isinstance(matrix_source, np.ndarray):
            self.source_path = "in_memory"
            self.H = matrix_source.astype(np.float64)
        else:
            raise TypeError(f"Unsupported homography source type: {type(matrix_source)}")

        if self.H.shape != (3, 3):
            raise ValueError(f"Homography matrix must have shape (3, 3), got {self.H.shape}")

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Load from a JSON file.  Returns (H, rich_metadata_dict_or_None)."""
        if not path.exists():
            raise FileNotFoundError(f"Homography file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # Legacy format: bare matrix
            return np.array(data, dtype=np.float64), None

        if isinstance(data, dict):
            fmt = data.get("format", "")
            if fmt == "courtvision_homography_v1":
                H = np.array(data["H"], dtype=np.float64)
                return H, data
            # Unknown dict format — try the "H" key, then fall back
            if "H" in data:
                return np.array(data["H"], dtype=np.float64), None
            raise ValueError(
                f"Unrecognized JSON homography format in {path}. "
                "Expected a bare matrix array or a courtvision_homography_v1 dict."
            )

        raise ValueError(f"Unexpected JSON content in {path}: {type(data)}")

    def _apply_rich_metadata(self, meta: Dict[str, Any]) -> None:
        """Configure calibrated-zone bounds from a rich calibration JSON."""
        self._calibration_metadata = meta
        self._is_full_pitch = bool(meta.get("full_pitch_calibration", False))
        self._calibration_scope = meta.get("calibration_scope", "partial")
        self._camera_motion_warning = meta.get("camera_motion_warning", "")
        self._correspondences = meta.get("correspondences", [])

        v = meta.get("validation", {})
        self._validation = v
        self._coverage_score = float(v.get("coverage_score", 0.0))

        if self._is_full_pitch:
            # For a validated full-pitch calibration the entire pitch is
            # the calibrated zone.
            self.box_depth = self.pitch_length
            self.box_y_min = 0.0
            self.box_y_max = self.pitch_width
        else:
            # Use inlier landmark bounding box as calibrated zone, with 5 m margin.
            pitch_pts = [
                (c["pitch_x"], c["pitch_y"])
                for c in self._correspondences
                if not (inlier_ok := True)  # include all saved correspondences
            ]
            inlier_mask = v.get("inlier_mask", [])
            pitch_pts = [
                (c["pitch_x"], c["pitch_y"])
                for c, ok in zip(self._correspondences, inlier_mask)
                if ok
            ]
            if pitch_pts:
                xs = [p[0] for p in pitch_pts]
                ys = [p[1] for p in pitch_pts]
                # Calibrated zone = inlier landmark bounding box + 5 m margin,
                # clipped to pitch boundaries.
                margin = 5.0
                self.box_depth = min(max(xs) + margin, self.pitch_length)
                self.box_y_min = max(min(ys) - margin, 0.0)
                self.box_y_max = min(max(ys) + margin, self.pitch_width)
            # else: keep the constructor defaults (penalty-box region)

    # ------------------------------------------------------------------
    # Public projection API  (unchanged signatures)
    # ------------------------------------------------------------------

    def project_point(
        self,
        pixel_x: float,
        pixel_y: float,
        margin_meters: float = 0.0,
    ) -> Tuple[float, float, bool]:
        """Transform a single pixel point to metric pitch coordinates.

        Returns
        -------
        (pitch_x, pitch_y, is_valid_pitch)
        ``is_valid_pitch`` is True when the projected point lies within the
        105 m × 68 m pitch boundary (plus optional margin).
        """
        vec = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
        proj = self.H @ vec

        if abs(proj[2]) < 1e-7:
            return float("nan"), float("nan"), False

        pitch_x = float(proj[0] / proj[2])
        pitch_y = float(proj[1] / proj[2])

        is_valid = self.is_within_pitch(pitch_x, pitch_y, margin_meters=margin_meters)
        return pitch_x, pitch_y, is_valid

    def project_bbox_foot(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        margin_meters: float = 0.0,
    ) -> Tuple[float, float, float, float, bool, bool]:
        """Extract bottom-center foot anchor and project to pitch coordinates.

        Returns
        -------
        (foot_x, foot_y, pitch_x, pitch_y, is_valid_pitch, is_in_calibrated_zone)
        """
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)
        pitch_x, pitch_y, is_valid = self.project_point(foot_x, foot_y, margin_meters=margin_meters)
        in_calibrated_zone = self.is_within_calibrated_zone(pitch_x, pitch_y)
        return foot_x, foot_y, pitch_x, pitch_y, is_valid, in_calibrated_zone

    def is_within_pitch(
        self,
        pitch_x: float,
        pitch_y: float,
        margin_meters: float = 0.0,
    ) -> bool:
        """True when (pitch_x, pitch_y) lies inside [0, pitch_length] × [0, pitch_width]."""
        if np.isnan(pitch_x) or np.isnan(pitch_y):
            return False
        x_ok = -margin_meters <= pitch_x <= (self.pitch_length + margin_meters)
        y_ok = -margin_meters <= pitch_y <= (self.pitch_width + margin_meters)
        return bool(x_ok and y_ok)

    def is_within_calibrated_zone(
        self,
        pitch_x: float,
        pitch_y: float,
        margin_meters: float = 5.0,
    ) -> bool:
        """True when the projected point is within the calibrated region.

        For legacy penalty-box calibrations this checks proximity to the 18-yard box.
        For rich calibrations derived from landmark coverage this checks proximity
        to the inlier bounding box.  For validated full-pitch calibrations the
        entire pitch is considered calibrated (margin applies to pitch edges).
        """
        if np.isnan(pitch_x) or np.isnan(pitch_y):
            return False
        x_ok = -margin_meters <= pitch_x <= (self.box_depth + margin_meters)
        y_ok = (self.box_y_min - margin_meters) <= pitch_y <= (self.box_y_max + margin_meters)
        return bool(x_ok and y_ok)

    def project_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """Batch project an array of shape (N, 2) to (N, 2).

        Points with a near-zero perspective scale (degenerate projections)
        are set to NaN.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected points array of shape (N, 2), got {points.shape}")

        n = len(points)
        homogeneous = np.hstack([points, np.ones((n, 1), dtype=np.float64)])
        projected = (self.H @ homogeneous.T).T

        scales = projected[:, 2]
        valid_mask = np.abs(scales) > 1e-7

        result = np.full((n, 2), np.nan, dtype=np.float64)
        result[valid_mask, 0] = projected[valid_mask, 0] / scales[valid_mask]
        result[valid_mask, 1] = projected[valid_mask, 1] / scales[valid_mask]
        return result

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata describing calibration scope and limitations.

        For rich calibrations this includes validation metrics and per-quadrant
        coverage.  For legacy calibrations a minimal summary is returned.
        """
        if self._calibration_scope != "penalty_box_region" and self._calibration_metadata:
            # Rich calibration: return saved metadata summary
            meta: Dict[str, Any] = {
                "source": self.source_path,
                "calibration_scope": self._calibration_scope,
                "full_pitch_calibration": self._is_full_pitch,
                "pitch_dimensions_m": [self.pitch_length, self.pitch_width],
                "calibrated_zone_m": {
                    "x_range": [0.0, self.box_depth],
                    "y_range": [self.box_y_min, self.box_y_max],
                },
                "coverage_score": self._coverage_score,
                "validation": self._validation,
            }
            if self._camera_motion_warning:
                meta["camera_motion_warning"] = self._camera_motion_warning
            return meta

        # Legacy / default metadata
        return {
            "source": self.source_path,
            "calibration_scope": "penalty_box_region",
            "full_pitch_calibration": False,
            "pitch_dimensions_m": [self.pitch_length, self.pitch_width],
            "calibrated_zone_m": {
                "x_range": [0.0, self.box_depth],
                "y_range": [self.box_y_min, self.box_y_max],
            },
            "warning": (
                "H is calibrated solely from 4 points of the 18-yard penalty box. "
                "Points outside the penalty-box region suffer from perspective divergence "
                "and must not be treated as trustworthy full-pitch coordinates."
            ),
        }
