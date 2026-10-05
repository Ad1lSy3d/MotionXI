import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class PitchHomography:
    """
    Project 2D camera pixel coordinates (u, v) into 2D metric pitch coordinates (X, Y) in meters.

    Calibration Context:
    The homography matrix H is calibrated using the standard FIFA 18-yard penalty box:
    - Pitch dimensions: 105m (length, X) x 68m (width, Y)
    - Penalty box depth: X in [0.0, 16.5]
    - Penalty box width: Y in [13.84, 54.16] (centered on 68m width)

    IMPORTANT RESEARCH & PERSPECTIVE LIMITATION:
    Because H was calculated from a 4-point planar calibration of a single 18-yard box,
    projecting pixel coordinates outside this calibrated quadrant (e.g. midfield, far touchline,
    or opposite half) is subject to significant planar extrapolation error and perspective divergence.
    This class enforces pitch boundary validation to discard or flag out-of-bounds projections.
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
        self.source_path = str(matrix_source) if isinstance(matrix_source, (str, Path)) else "in_memory"
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)
        self.box_depth = float(box_depth)
        self.box_y_min = float(box_y_min)
        self.box_y_max = float(box_y_max)

        self.H = self._load_matrix(matrix_source)
        if self.H.shape != (3, 3):
            raise ValueError(f"Homography matrix must have shape (3, 3), got {self.H.shape}")

    @staticmethod
    def _load_matrix(
        source: Union[str, Path, np.ndarray, List[List[float]]]
    ) -> np.ndarray:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Homography file not found at: {path}")
            with open(path, "r") as f:
                data = json.load(f)
            return np.array(data, dtype=np.float64)
        elif isinstance(source, (list, tuple)):
            return np.array(source, dtype=np.float64)
        elif isinstance(source, np.ndarray):
            return source.astype(np.float64)
        else:
            raise TypeError(f"Unsupported homography source type: {type(source)}")

    def project_point(
        self,
        pixel_x: float,
        pixel_y: float,
        margin_meters: float = 0.0,
    ) -> Tuple[float, float, bool]:
        """
        Transform a single pixel point (e.g. player foot contact) to metric pitch coordinates.

        Returns:
            Tuple of (pitch_x, pitch_y, is_valid_pitch)
            where is_valid_pitch indicates the point falls within the 105m x 68m pitch (+ margin).
        """
        vec = np.array([pixel_x, pixel_y, 1.0], dtype=np.float64)
        proj = self.H @ vec

        # Prevent division by zero or negative perspective scale
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
        """
        Extract the bottom-center foot contact anchor and project to pitch coordinates.

        Returns:
            Tuple of:
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
        """Check if pitch coordinate lies within the configured boundary [0, pitch_length] x [0, pitch_width]."""
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
        """Check if coordinate is near the calibrated 18-yard box where homography error is minimal."""
        if np.isnan(pitch_x) or np.isnan(pitch_y):
            return False
        x_ok = -margin_meters <= pitch_x <= (self.box_depth + margin_meters)
        y_ok = (self.box_y_min - margin_meters) <= pitch_y <= (self.box_y_max + margin_meters)
        return bool(x_ok and y_ok)

    def project_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Batch project an array of shape (N, 2) to (N, 2).
        Points with singular scale are set to NaN.
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

    def get_metadata(self) -> Dict[str, Any]:
        """Return standardized homography metadata describing scope and limitations."""
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
