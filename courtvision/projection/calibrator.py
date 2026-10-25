"""
courtvision.projection.calibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Full-pitch homography calibration for CourtVision AI.

Coordinate Convention
---------------------
This module uses the standard 105 m × 68 m FIFA pitch coordinate frame:

    Origin (0, 0) ← Left goal-line / near touchline intersection

         Y (width, 0 → 68 m, near touchline → far touchline)
         |
    0,68 +-----------------------------------+ 105,68   ← far touchline
         |                                   |
         |   penalty   |       |   penalty   |
         |   box left  |       |   box right |
         |             |       |             |
    0,0  +-----------------------------------+ 105,0    ← near touchline
         Origin                              X (length, 0 → 105 m) →

Axis meanings:
  X  — pitch length, 0 at left goal line, 105 at right goal line
  Y  — pitch width, 0 at near touchline, 68 at far touchline

Named landmarks use this convention (see PITCH_LANDMARKS).

Calibration workflow
--------------------
1. Instantiate ``HomographyCalibrator``.
2. Add correspondences via ``add_correspondence(pixel_x, pixel_y, pitch_x, pitch_y)``.
   Use clearly identified landmarks across the pitch.
3. Call ``estimate()`` to compute the homography using OpenCV RANSAC + refinement.
4. Inspect the returned ``CalibrationResult`` for inlier count, error residuals,
   coverage metrics, calibrated-zone bounds, and warnings.
5. Save via ``save(path)`` to a metadata-rich JSON file.
6. Load via ``PitchHomography(path)`` (backward-compatible with both legacy bare-matrix
   and rich JSON formats).

Fundamental limitations & Error Interpretation
----------------------------------------------
* Image-space reprojection error (px) measures residual error between the
  clicked image pixel and the back-projected pitch landmark via H_inv.
* Forward-projection error (m) measures the metric deviation between the true
  landmark coordinates on the pitch plane and the projected image point via H.
* Low residuals on training landmarks DO NOT prove high real-world accuracy across
  the pitch. Sparse landmarks, lens distortion, rolling shutter, elevation angle,
  and camera panning/zooming induce localized projection errors that training
  residuals cannot detect.
* A single static homography assumes a flat pitch plane and a stationary camera.
  Footage with panning, zooming, tilting, or broadcast camera cuts requires
  per-shot recalibration or dynamic motion compensation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Standard pitch landmark catalogue
# ---------------------------------------------------------------------------

PITCH_LANDMARKS: Dict[str, Tuple[float, float]] = {
    # Left penalty area (goal at X=0)
    "left_goal_near_post":        (0.0,   17.5),
    "left_goal_line_near_pa":     (0.0,   13.84),
    "left_goal_line_far_pa":      (0.0,   54.16),
    "left_pa_near_corner":        (16.5,  13.84),
    "left_pa_far_corner":         (16.5,  54.16),
    "left_six_yard_near":         (5.5,   24.84),
    "left_six_yard_far":          (5.5,   43.16),
    "left_penalty_spot":          (11.0,  34.0),
    "left_goal_line_near_touch":  (0.0,   0.0),
    "left_goal_line_far_touch":   (0.0,   68.0),
    # Right penalty area (goal at X=105)
    "right_goal_line_near_pa":    (105.0, 13.84),
    "right_goal_line_far_pa":     (105.0, 54.16),
    "right_pa_near_corner":       (88.5,  13.84),
    "right_pa_far_corner":        (88.5,  54.16),
    "right_six_yard_near":        (99.5,  24.84),
    "right_six_yard_far":         (99.5,  43.16),
    "right_penalty_spot":         (94.0,  34.0),
    "right_goal_line_near_touch": (105.0, 0.0),
    "right_goal_line_far_touch":  (105.0, 68.0),
    # Halfway / midfield
    "halfway_near_touch":         (52.5,  0.0),
    "halfway_far_touch":          (52.5,  68.0),
    "center_spot":                (52.5,  34.0),
    # Touchline corners (whole pitch)
    "corner_near_left":           (0.0,   0.0),
    "corner_far_left":            (0.0,   68.0),
    "corner_near_right":          (105.0, 0.0),
    "corner_far_right":           (105.0, 68.0),
}

# Minimum correspondences required for mathematical homography estimation
MIN_CORRESPONDENCES: int = 4
# Recommended for reliable full-pitch coverage
RECOMMENDED_CORRESPONDENCES: int = 10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LandmarkCorrespondence:
    """A single image↔pitch coordinate correspondence."""
    pixel_x: float
    pixel_y: float
    pitch_x: float
    pitch_y: float
    label: Optional[str] = None


@dataclass
class CalibrationResult:
    """Result from ``HomographyCalibrator.estimate()``.

    Contains separated image-space reprojection errors (px) and forward
    pitch-plane errors (m), broken down by inliers and overall set.
    """
    H: np.ndarray                              # 3×3 homography matrix (image → pitch)
    inlier_count: int                          # RANSAC inliers count
    total_correspondences: int                 # Total correspondences provided
    rms_reprojection_px: float                 # Inlier RMS image-space reprojection error (pixels)
    max_reprojection_px: float                 # Inlier max image-space reprojection error (pixels)
    rms_reprojection_m: float                  # Inlier RMS forward-projection error (metres)
    max_reprojection_m: float                  # Inlier max forward-projection error (metres)
    overall_rms_reprojection_px: float         # Overall RMS image-space reprojection error across all points (pixels)
    overall_rms_reprojection_m: float          # Overall RMS forward-projection error across all points (metres)
    per_point_errors_px: List[float]           # Per-point image-space reprojection errors (pixels)
    per_point_errors_m: List[float]            # Per-point forward-projection errors (metres)
    inlier_mask: List[bool]                    # Which correspondences are RANSAC inliers
    coverage_score: float                      # Convex-hull area of inliers divided by pitch area [0, 1]
    quadrant_coverage: Dict[str, bool]         # Coverage indicators for pitch regions
    calibrated_zone_m: Dict[str, List[float]]  # Documented valid pitch bounding box {"x_range": [x_min, x_max], "y_range": [y_min, y_max]}
    warnings: List[str]                        # Explicit warnings regarding coverage and accuracy limitations
    calibration_scope: str                     # 'full_pitch', 'partial', or 'penalty_box_region'
    is_valid: bool                             # True if H is finite, non-degenerate, and invertible

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dictionary with documented metric descriptions."""
        return {
            "inlier_count": self.inlier_count,
            "total_correspondences": self.total_correspondences,
            "rms_reprojection_px": self.rms_reprojection_px,
            "max_reprojection_px": self.max_reprojection_px,
            "rms_reprojection_m": self.rms_reprojection_m,
            "max_reprojection_m": self.max_reprojection_m,
            "overall_rms_reprojection_px": self.overall_rms_reprojection_px,
            "overall_rms_reprojection_m": self.overall_rms_reprojection_m,
            "per_point_errors_px": self.per_point_errors_px,
            "per_point_errors_m": self.per_point_errors_m,
            "inlier_mask": self.inlier_mask,
            "coverage_score": self.coverage_score,
            "quadrant_coverage": self.quadrant_coverage,
            "calibrated_zone_m": self.calibrated_zone_m,
            "warnings": self.warnings,
            "calibration_scope": self.calibration_scope,
            "is_valid": self.is_valid,
            "metric_definitions": {
                "rms_reprojection_px": "Inlier root-mean-square error in image pixels: ||H_inv * X_pitch - x_pixel||.",
                "rms_reprojection_m": "Inlier root-mean-square error in pitch metres: ||H * x_pixel - X_pitch||.",
                "overall_rms_reprojection_px": "RMS image pixel error across all input correspondences (including RANSAC outliers).",
                "overall_rms_reprojection_m": "RMS pitch metre error across all input correspondences (including RANSAC outliers).",
                "coverage_score": "Ratio of inlier landmarks convex hull area to total pitch area (105 m x 68 m = 7140 m2).",
                "limitation_note": (
                    "Low training-landmark residuals do not guarantee projection accuracy across uncalibrated "
                    "or sparse pitch regions. Planar perspective divergence increases with distance from landmarks."
                ),
            },
        }


# ---------------------------------------------------------------------------
# Reprojection error utilities
# ---------------------------------------------------------------------------


def _compute_reprojection_errors_px(
    H: np.ndarray,
    correspondences: List[LandmarkCorrespondence],
) -> Tuple[List[float], float, float]:
    """Compute per-point image-space reprojection errors.

    Maps pitch landmarks back to image coordinates using H_inv and calculates
    Euclidean pixel distance from the observed pixel: ||H_inv * X_pitch - x_pixel||.

    Returns
    -------
    (per_point_px, rms_px, max_px)
    """
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        n = len(correspondences)
        return [float("inf")] * n, float("inf"), float("inf")

    errors_px: List[float] = []
    for c in correspondences:
        pvec = np.array([c.pitch_x, c.pitch_y, 1.0], dtype=np.float64)
        back = H_inv @ pvec
        if abs(back[2]) < 1e-9:
            errors_px.append(float("inf"))
            continue
        repx = back[0] / back[2]
        repy = back[1] / back[2]
        err_px = math.sqrt((repx - c.pixel_x) ** 2 + (repy - c.pixel_y) ** 2)
        errors_px.append(err_px)

    finite = [e for e in errors_px if math.isfinite(e)]
    if not finite:
        return errors_px, float("inf"), float("inf")
    rms = math.sqrt(sum(e ** 2 for e in finite) / len(finite))
    return errors_px, rms, max(finite)


def _compute_forward_errors_m(
    H: np.ndarray,
    correspondences: List[LandmarkCorrespondence],
) -> Tuple[List[float], float, float]:
    """Compute per-point forward pitch-space projection errors in metres.

    Maps clicked image pixels to the pitch plane using H and calculates
    metric Euclidean distance from the ground-truth pitch landmark:
    ||H * x_pixel - X_pitch||.

    Returns
    -------
    (per_point_m, rms_m, max_m)
    """
    errors_m: List[float] = []
    for c in correspondences:
        vec = np.array([c.pixel_x, c.pixel_y, 1.0], dtype=np.float64)
        proj = H @ vec
        if abs(proj[2]) < 1e-9:
            errors_m.append(float("inf"))
            continue
        px_proj = proj[0] / proj[2]
        py_proj = proj[1] / proj[2]
        err_m = math.sqrt((px_proj - c.pitch_x) ** 2 + (py_proj - c.pitch_y) ** 2)
        errors_m.append(err_m)

    finite = [e for e in errors_m if math.isfinite(e)]
    if not finite:
        return errors_m, float("inf"), float("inf")
    rms = math.sqrt(sum(e ** 2 for e in finite) / len(finite))
    return errors_m, rms, max(finite)


def compute_reprojection_errors(
    H: np.ndarray,
    pixel_points: np.ndarray,
    pitch_points: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Public helper to compute image-space reprojection errors for point arrays.

    Parameters
    ----------
    H:
        3×3 homography matrix (image → pitch).
    pixel_points:
        Shape (N, 2) array of image pixel coordinates.
    pitch_points:
        Shape (N, 2) array of metric pitch coordinates.

    Returns
    -------
    (per_point_errors_px, rms_px, max_px)
    """
    if pixel_points.shape[0] == 0:
        return np.array([], dtype=np.float64), float("inf"), float("inf")

    corr = [
        LandmarkCorrespondence(
            pixel_x=float(p[0]),
            pixel_y=float(p[1]),
            pitch_x=float(q[0]),
            pitch_y=float(q[1]),
        )
        for p, q in zip(pixel_points, pitch_points)
    ]
    per_pt, rms, max_err = _compute_reprojection_errors_px(H, corr)
    return np.array(per_pt, dtype=np.float64), rms, max_err


# ---------------------------------------------------------------------------
# Coverage & Scope Classification
# ---------------------------------------------------------------------------


def _compute_coverage(
    pitch_points: List[Tuple[float, float]],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> Tuple[float, Dict[str, bool], Dict[str, float]]:
    """Score spatial pitch coverage based on inlier landmark geometry.

    Returns
    -------
    (coverage_score, quadrant_coverage, span_dict)
    """
    pts = [
        (x, y) for x, y in pitch_points
        if 0.0 <= x <= pitch_length and 0.0 <= y <= pitch_width
    ]
    total_area = pitch_length * pitch_width  # 7140 m²

    quadrants: Dict[str, bool] = {
        "left_third":   any(x <= pitch_length / 3 for x, _ in pts),
        "middle_third": any(pitch_length / 3 < x <= 2 * pitch_length / 3 for x, _ in pts),
        "right_third":  any(x > 2 * pitch_length / 3 for x, _ in pts),
        "near_half":    any(y <= pitch_width / 2 for _, y in pts),
        "far_half":     any(y > pitch_width / 2 for _, y in pts),
    }

    if not pts:
        spans = {
            "x_min": 0.0, "x_max": 0.0, "x_span": 0.0,
            "y_min": 0.0, "y_max": 0.0, "y_span": 0.0,
        }
        return 0.0, quadrants, spans

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    spans = {
        "x_min": x_min, "x_max": x_max, "x_span": x_max - x_min,
        "y_min": y_min, "y_max": y_max, "y_span": y_max - y_min,
    }

    if len(pts) < 3:
        return 0.0, quadrants, spans

    try:
        pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts_arr)
        hull_area = float(cv2.contourArea(hull))
        score = min(hull_area / total_area, 1.0)
    except Exception:
        score = 0.0

    return score, quadrants, spans


def _classify_scope(
    coverage_score: float,
    quadrant_coverage: Dict[str, bool],
    spans: Dict[str, float],
    n_inliers: int,
    inlier_rms_px: float,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> str:
    """Classify calibration scope conservatively.

    Criteria for 'full_pitch':
    1. At least 10 inliers (n_inliers >= 10).
    2. Landmarks present in all three pitch thirds (left, middle, right).
    3. Landmarks present on both width halves (near and far touchlines).
    4. Pitch length span >= 70% of pitch length (spans['x_span'] >= 0.70 * pitch_length).
    5. Pitch width span >= 60% of pitch width (spans['y_span'] >= 0.60 * pitch_width).
    6. Extremities reach deep into both halves (x_min <= 20.0 and x_max >= 85.0).
    7. Convex hull coverage score >= 0.50 (at least 50% of pitch enclosed).
    8. Inlier RMS reprojection error <= 4.0 px.

    If these conditions are not all met, the calibration is classified as 'partial'
    (or 'penalty_box_region' if points are confined to a single penalty box area).
    """
    all_thirds = (
        quadrant_coverage.get("left_third", False)
        and quadrant_coverage.get("middle_third", False)
        and quadrant_coverage.get("right_third", False)
    )
    both_width_halves = (
        quadrant_coverage.get("near_half", False)
        and quadrant_coverage.get("far_half", False)
    )
    length_span_ok = spans["x_span"] >= (0.70 * pitch_length)
    width_span_ok = spans["y_span"] >= (0.60 * pitch_width)
    extremities_ok = (spans["x_min"] <= 20.0) and (spans["x_max"] >= (pitch_length - 20.0))
    area_ok = coverage_score >= 0.50
    inliers_ok = n_inliers >= RECOMMENDED_CORRESPONDENCES
    error_ok = inlier_rms_px <= 4.0

    if (
        all_thirds
        and both_width_halves
        and length_span_ok
        and width_span_ok
        and extremities_ok
        and area_ok
        and inliers_ok
        and error_ok
    ):
        return "full_pitch"

    # Check if confined to a single penalty box
    is_left_pa = (spans["x_max"] <= 25.0) and (10.0 <= spans["y_min"] <= spans["y_max"] <= 58.0)
    is_right_pa = (spans["x_min"] >= (pitch_length - 25.0)) and (10.0 <= spans["y_min"] <= spans["y_max"] <= 58.0)
    if is_left_pa or is_right_pa:
        return "penalty_box_region"

    return "partial"


# ---------------------------------------------------------------------------
# Main calibrator class
# ---------------------------------------------------------------------------


class HomographyCalibrator:
    """Full-pitch homography calibrator.

    Usage
    -----
    ::

        cal = HomographyCalibrator()
        cal.add_correspondence(820, 655, 16.5, 54.16, label="left_pa_far_corner")
        # ... add >= 10 well-distributed correspondences across the pitch ...
        result = cal.estimate()
        print(f"Scope: {result.calibration_scope}, Inlier RMS: {result.rms_reprojection_px:.2f} px")
        cal.save("data/homography_full_pitch.json")

    Parameters
    ----------
    pitch_length:
        Pitch length in meters (default 105.0).
    pitch_width:
        Pitch width in meters (default 68.0).
    ransac_threshold:
        RANSAC reprojection threshold in pixels (default 3.0 px).
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        ransac_threshold: float = 3.0,
    ) -> None:
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)
        self.ransac_threshold = float(ransac_threshold)
        self._correspondences: List[LandmarkCorrespondence] = []
        self._last_result: Optional[CalibrationResult] = None

    def add_correspondence(
        self,
        pixel_x: float,
        pixel_y: float,
        pitch_x: float,
        pitch_y: float,
        label: Optional[str] = None,
    ) -> None:
        """Add an image pixel ↔ metric pitch coordinate correspondence."""
        self._correspondences.append(
            LandmarkCorrespondence(
                pixel_x=float(pixel_x),
                pixel_y=float(pixel_y),
                pitch_x=float(pitch_x),
                pitch_y=float(pitch_y),
                label=label,
            )
        )

    def clear_correspondences(self) -> None:
        """Remove all stored correspondences."""
        self._correspondences.clear()
        self._last_result = None

    @property
    def correspondences(self) -> List[LandmarkCorrespondence]:
        return list(self._correspondences)

    def estimate(self) -> CalibrationResult:
        """Estimate homography matrix from correspondences using RANSAC.

        Computes:
        - Inlier and overall image-space reprojection errors (pixels).
        - Inlier and overall forward-projection errors (metres).
        - Pitch coverage score and regional span analysis.
        - Calibrated pitch bounding box (with min and max X and Y bounds).
        - Conservative calibration scope classification.
        - Comprehensive reliability warnings.
        """
        n = len(self._correspondences)
        warnings: List[str] = []

        if n < MIN_CORRESPONDENCES:
            dummy_H = np.eye(3, dtype=np.float64)
            result = CalibrationResult(
                H=dummy_H,
                inlier_count=0,
                total_correspondences=n,
                rms_reprojection_px=float("inf"),
                max_reprojection_px=float("inf"),
                rms_reprojection_m=float("inf"),
                max_reprojection_m=float("inf"),
                overall_rms_reprojection_px=float("inf"),
                overall_rms_reprojection_m=float("inf"),
                per_point_errors_px=[],
                per_point_errors_m=[],
                inlier_mask=[],
                coverage_score=0.0,
                quadrant_coverage={
                    "left_third": False, "middle_third": False, "right_third": False,
                    "near_half": False, "far_half": False,
                },
                calibrated_zone_m={"x_range": [0.0, 0.0], "y_range": [0.0, 0.0]},
                warnings=[
                    f"Insufficient correspondences: {n} provided, minimum required is {MIN_CORRESPONDENCES}."
                ],
                calibration_scope="penalty_box_region",
                is_valid=False,
            )
            self._last_result = result
            return result

        src_pts = np.array(
            [[c.pixel_x, c.pixel_y] for c in self._correspondences], dtype=np.float32
        )
        dst_pts = np.array(
            [[c.pitch_x, c.pitch_y] for c in self._correspondences], dtype=np.float32
        )

        # RANSAC homography estimation
        H_raw, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold
        )

        if H_raw is None or mask is None:
            warnings.append(
                "cv2.findHomography returned None — correspondences may be degenerate "
                "(collinear, duplicate pixels, or numerically unstable)."
            )
            dummy_H = np.eye(3, dtype=np.float64)
            result = CalibrationResult(
                H=dummy_H,
                inlier_count=0,
                total_correspondences=n,
                rms_reprojection_px=float("inf"),
                max_reprojection_px=float("inf"),
                rms_reprojection_m=float("inf"),
                max_reprojection_m=float("inf"),
                overall_rms_reprojection_px=float("inf"),
                overall_rms_reprojection_m=float("inf"),
                per_point_errors_px=[float("inf")] * n,
                per_point_errors_m=[float("inf")] * n,
                inlier_mask=[False] * n,
                coverage_score=0.0,
                quadrant_coverage={
                    "left_third": False, "middle_third": False, "right_third": False,
                    "near_half": False, "far_half": False,
                },
                calibrated_zone_m={"x_range": [0.0, 0.0], "y_range": [0.0, 0.0]},
                warnings=warnings,
                calibration_scope="penalty_box_region",
                is_valid=False,
            )
            self._last_result = result
            return result

        inlier_mask = [bool(m[0]) for m in mask]
        inlier_count = sum(inlier_mask)
        H = H_raw.astype(np.float64)

        # Validate non-degeneracy
        is_valid = True
        if not np.all(np.isfinite(H)):
            warnings.append("Estimated homography contains non-finite values (NaN or Inf).")
            is_valid = False

        det = float(np.linalg.det(H))
        if abs(det) < 1e-10:
            warnings.append(
                f"Homography is near-singular (det={det:.2e}). "
                "Correspondences may be collinear or insufficient."
            )
            is_valid = False

        if is_valid:
            pitch_corners = np.array(
                [[0, 0], [self.pitch_length, 0],
                 [self.pitch_length, self.pitch_width], [0, self.pitch_width]],
                dtype=np.float64
            )
            try:
                H_inv = np.linalg.inv(H)
                corners_h = np.hstack([pitch_corners, np.ones((4, 1))])
                back = (H_inv @ corners_h.T).T
                scales = back[:, 2]
                if np.any(np.abs(scales) < 1e-7) or not np.all(np.isfinite(back)):
                    warnings.append(
                        "Pitch corners back-project to degenerate pixel positions. "
                        "Check landmark selection and coordinate conventions."
                    )
                    is_valid = False
                else:
                    pix_back = back[:, :2] / scales[:, None]
                    if np.any(pix_back < -2000) or np.any(pix_back > 1e5):
                        warnings.append(
                            "Pitch corner back-projections are implausibly outside image boundaries. "
                            "Verify landmark pixel accuracy."
                        )
            except np.linalg.LinAlgError:
                warnings.append("Could not invert homography matrix for validation.")
                is_valid = False

        # Compute per-point errors
        per_pt_errors_px, overall_rms_px, overall_max_px = _compute_reprojection_errors_px(
            H, self._correspondences
        )
        per_pt_errors_m, overall_rms_m, overall_max_m = _compute_forward_errors_m(
            H, self._correspondences
        )

        # Inlier-only error metrics
        inlier_errors_px = [
            e for e, ok in zip(per_pt_errors_px, inlier_mask)
            if ok and math.isfinite(e)
        ]
        inlier_errors_m = [
            e for e, ok in zip(per_pt_errors_m, inlier_mask)
            if ok and math.isfinite(e)
        ]

        if inlier_errors_px:
            inlier_rms_px = math.sqrt(sum(e ** 2 for e in inlier_errors_px) / len(inlier_errors_px))
            inlier_max_px = max(inlier_errors_px)
        else:
            inlier_rms_px = float("inf")
            inlier_max_px = float("inf")

        if inlier_errors_m:
            inlier_rms_m = math.sqrt(sum(e ** 2 for e in inlier_errors_m) / len(inlier_errors_m))
            inlier_max_m = max(inlier_errors_m)
        else:
            inlier_rms_m = float("inf")
            inlier_max_m = float("inf")

        # Spatial coverage calculation
        inlier_pitch_pts = [
            (self._correspondences[i].pitch_x, self._correspondences[i].pitch_y)
            for i, ok in enumerate(inlier_mask)
            if ok
        ]
        coverage_score, quadrant_coverage, spans = _compute_coverage(
            inlier_pitch_pts, self.pitch_length, self.pitch_width
        )

        # Compute calibrated zone bounds with documented 5.0m margin
        margin_m = 5.0
        if inlier_pitch_pts:
            cal_x_min = max(0.0, spans["x_min"] - margin_m)
            cal_x_max = min(self.pitch_length, spans["x_max"] + margin_m)
            cal_y_min = max(0.0, spans["y_min"] - margin_m)
            cal_y_max = min(self.pitch_width, spans["y_max"] + margin_m)
        else:
            cal_x_min, cal_x_max = 0.0, 16.5
            cal_y_min, cal_y_max = 13.84, 54.16

        # Classify scope
        calibration_scope = _classify_scope(
            coverage_score=coverage_score,
            quadrant_coverage=quadrant_coverage,
            spans=spans,
            n_inliers=inlier_count,
            inlier_rms_px=inlier_rms_px,
            pitch_length=self.pitch_length,
            pitch_width=self.pitch_width,
        )

        if calibration_scope == "full_pitch":
            cal_x_min = 0.0
            cal_x_max = self.pitch_length
            cal_y_min = 0.0
            cal_y_max = self.pitch_width

        calibrated_zone_m = {
            "x_range": [round(cal_x_min, 2), round(cal_x_max, 2)],
            "y_range": [round(cal_y_min, 2), round(cal_y_max, 2)],
        }

        # Build informative, honest warnings
        if inlier_rms_px > 3.0:
            warnings.append(
                f"Elevated inlier RMS reprojection error: {inlier_rms_px:.2f} px "
                f"({inlier_rms_m:.2f} m). Check landmark accuracy."
            )

        if calibration_scope == "full_pitch":
            warnings.append(
                "Geometric landmark coverage does not guarantee real-world projection accuracy across "
                "the entire pitch. Extrapolation between sparse landmarks, lens distortion, rolling shutter, "
                "and camera elevation can introduce significant local metric distortion. "
                "Independent pitch-landmark validation is strongly recommended."
            )
        else:
            warnings.append(
                f"Calibration is partial (calibrated zone X: [{cal_x_min:.1f}, {cal_x_max:.1f}] m, "
                f"Y: [{cal_y_min:.1f}, {cal_y_max:.1f}] m). Projections outside this zone suffer from planar "
                "extrapolation divergence and should not be treated as reliable metric coordinates."
            )

        if n < RECOMMENDED_CORRESPONDENCES:
            warnings.append(
                f"Only {n} correspondences provided. >= {RECOMMENDED_CORRESPONDENCES} well-distributed "
                "correspondences across both touchlines and both halves are required for full-pitch validation."
            )

        if inlier_count < MIN_CORRESPONDENCES:
            warnings.append(
                f"Only {inlier_count} RANSAC inliers from {n} correspondences. Calibration quality is unreliable."
            )

        if not quadrant_coverage.get("middle_third", False):
            warnings.append("No inlier landmarks in the middle third of the pitch; midfield accuracy is unvalidated.")

        if not quadrant_coverage.get("right_third", False):
            warnings.append("No inlier landmarks in the right third of the pitch; far-half accuracy is extrapolated.")

        if spans["y_span"] < (0.60 * self.pitch_width):
            warnings.append(
                f"Narrow pitch width span ({spans['y_span']:.1f} m < {0.60 * self.pitch_width:.1f} m); "
                "touchline projection accuracy is unconstrained."
            )

        result = CalibrationResult(
            H=H,
            inlier_count=inlier_count,
            total_correspondences=n,
            rms_reprojection_px=inlier_rms_px,
            max_reprojection_px=inlier_max_px,
            rms_reprojection_m=inlier_rms_m,
            max_reprojection_m=inlier_max_m,
            overall_rms_reprojection_px=overall_rms_px,
            overall_rms_reprojection_m=overall_rms_m,
            per_point_errors_px=per_pt_errors_px,
            per_point_errors_m=per_pt_errors_m,
            inlier_mask=inlier_mask,
            coverage_score=coverage_score,
            quadrant_coverage=quadrant_coverage,
            calibrated_zone_m=calibrated_zone_m,
            warnings=warnings,
            calibration_scope=calibration_scope,
            is_valid=is_valid,
        )
        self._last_result = result
        return result

    def save(
        self,
        path: str | Path,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save calibration to a rich, self-documenting JSON file."""
        if self._last_result is None:
            raise RuntimeError("Call estimate() before save(). No calibration result available.")

        result = self._last_result
        corr_list = [
            {
                "pixel_x": c.pixel_x,
                "pixel_y": c.pixel_y,
                "pitch_x": c.pitch_x,
                "pitch_y": c.pitch_y,
                "label": c.label,
            }
            for c in self._correspondences
        ]

        payload: Dict[str, Any] = {
            "format": "courtvision_homography_v1",
            "H": result.H.tolist(),
            "pitch_length_m": self.pitch_length,
            "pitch_width_m": self.pitch_width,
            "calibration_scope": result.calibration_scope,
            "full_pitch_calibration": result.calibration_scope == "full_pitch",
            "calibrated_zone_m": result.calibrated_zone_m,
            "correspondences": corr_list,
            "validation": result.to_dict(),
            "coordinate_convention": (
                "Origin (0,0) at left goal-line / near touchline intersection. "
                "X axis along pitch length (0→105 m), Y axis along pitch width (0→68 m)."
            ),
            "camera_motion_limitation": (
                "This homography assumes a static camera and a planar pitch. Camera panning, "
                "tilting, zooming, or broadcast shot cuts invalidate this matrix and require recalibration."
            ),
        }
        if extra_metadata:
            payload.update(extra_metadata)

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "HomographyCalibrator":
        """Restore calibrator and correspondence data from a saved rich JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "format" not in data or data["format"] != "courtvision_homography_v1":
            raise ValueError(
                f"File at {path} is not a valid courtvision_homography_v1 file."
            )

        cal = cls(
            pitch_length=float(data.get("pitch_length_m", 105.0)),
            pitch_width=float(data.get("pitch_width_m", 68.0)),
        )

        for c in data.get("correspondences", []):
            cal.add_correspondence(
                pixel_x=c["pixel_x"],
                pixel_y=c["pixel_y"],
                pitch_x=c["pitch_x"],
                pitch_y=c["pitch_y"],
                label=c.get("label"),
            )

        H = np.array(data["H"], dtype=np.float64)
        v = data.get("validation", {})
        cal._last_result = CalibrationResult(
            H=H,
            inlier_count=v.get("inlier_count", 0),
            total_correspondences=v.get("total_correspondences", 0),
            rms_reprojection_px=v.get("rms_reprojection_px", float("inf")),
            max_reprojection_px=v.get("max_reprojection_px", float("inf")),
            rms_reprojection_m=v.get("rms_reprojection_m", float("inf")),
            max_reprojection_m=v.get("max_reprojection_m", float("inf")),
            overall_rms_reprojection_px=v.get("overall_rms_reprojection_px", float("inf")),
            overall_rms_reprojection_m=v.get("overall_rms_reprojection_m", float("inf")),
            per_point_errors_px=v.get("per_point_errors_px", []),
            per_point_errors_m=v.get("per_point_errors_m", []),
            inlier_mask=v.get("inlier_mask", []),
            coverage_score=v.get("coverage_score", 0.0),
            quadrant_coverage=v.get("quadrant_coverage", {}),
            calibrated_zone_m=data.get("calibrated_zone_m") or v.get("calibrated_zone_m", {"x_range": [0.0, 0.0], "y_range": [0.0, 0.0]}),
            warnings=v.get("warnings", []),
            calibration_scope=data.get("calibration_scope", v.get("calibration_scope", "partial")),
            is_valid=v.get("is_valid", False),
        )
        return cal
