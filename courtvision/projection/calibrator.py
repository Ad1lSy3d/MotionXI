"""
courtvision.projection.calibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Full-pitch homography calibration for CourtVision AI.

Coordinate Convention
---------------------
This module uses the following 105 m × 68 m FIFA pitch coordinate frame:

    Origin (0, 0) ← Near goal line, near touchline (left end, bottom edge)

         Y (width, 0 → 68 m)
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
   Use clearly identified landmarks in the video frame.
3. Call ``estimate()`` to compute the homography using RANSAC + refinement.
4. Inspect the returned ``CalibrationResult`` for reprojection errors, coverage,
   inlier count, and warnings.
5. Save via ``save(path)`` to a metadata-rich JSON file.
6. Load via ``PitchHomography(path)`` (backward-compatible with both legacy and
   rich JSON formats).

Fundamental limitations
-----------------------
* A single static homography assumes a flat planar pitch and a static camera.
  Any camera pan, tilt, zoom, or cut will introduce errors not caught by this
  calibration. A single homography cannot compensate for dynamic camera motion.
* Projection accuracy degrades for landmarks far from the calibrated region.
* Minimum 4 non-collinear correspondences are required; ≥ 8 well-distributed
  correspondences are recommended for reliable full-pitch coverage.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Standard pitch landmark catalogue
# ---------------------------------------------------------------------------

PITCH_LANDMARKS: Dict[str, Tuple[float, float]] = {
    # Left penalty area (goal at X=0)
    "left_goal_near_post":        (0.0,   0.0 + 13.84 + 3.66),   # (0, 17.5) centre offset
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

# Minimum landmarks required for a viable estimate
MIN_CORRESPONDENCES: int = 4
# Recommended for reliable full-pitch coverage
RECOMMENDED_CORRESPONDENCES: int = 8


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

    All reprojection errors are in image pixels (image-space RMS).
    Pitch-space errors (in metres) are also reported where available.
    """
    H: np.ndarray                     # 3×3 homography matrix (image → pitch)
    inlier_count: int                 # RANSAC inliers
    total_correspondences: int        # Total correspondences provided
    rms_reprojection_px: float        # RMS reprojection error in image pixels
    max_reprojection_px: float        # Max reprojection error in image pixels
    rms_reprojection_m: float         # RMS error projecting pitch back to image and measuring pitch deviation
    per_point_errors_px: List[float]  # Per-point reprojection errors in pixels
    inlier_mask: List[bool]           # Which correspondences are inliers
    coverage_score: float             # [0, 1] geometric coverage across pitch (convex-hull area fraction)
    quadrant_coverage: Dict[str, bool]  # Which pitch thirds are covered by inlier points
    warnings: List[str]               # Human-readable warnings
    calibration_scope: str            # 'full_pitch', 'partial', 'penalty_box_region'
    is_valid: bool                    # True if H is finite and non-degenerate

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable summary."""
        return {
            "inlier_count": self.inlier_count,
            "total_correspondences": self.total_correspondences,
            "rms_reprojection_px": self.rms_reprojection_px,
            "max_reprojection_px": self.max_reprojection_px,
            "rms_reprojection_m": self.rms_reprojection_m,
            "per_point_errors_px": self.per_point_errors_px,
            "inlier_mask": self.inlier_mask,
            "coverage_score": self.coverage_score,
            "quadrant_coverage": self.quadrant_coverage,
            "warnings": self.warnings,
            "calibration_scope": self.calibration_scope,
            "is_valid": self.is_valid,
        }


# ---------------------------------------------------------------------------
# Reprojection error utilities
# ---------------------------------------------------------------------------


def _compute_reprojection_errors_px(
    H: np.ndarray,
    correspondences: List[LandmarkCorrespondence],
) -> Tuple[List[float], float, float]:
    """Compute per-point image-space reprojection errors.

    We project pitch coordinates back to image space using H_inv and measure
    pixel distance from the original clicked pixel.  This captures round-trip
    accuracy in the observable (pixel) space.

    Returns
    -------
    (per_point_px, rms_px, max_px)
    """
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        n = len(correspondences)
        inf_list = [float("inf")] * n
        return inf_list, float("inf"), float("inf")

    errors_px: List[float] = []
    for c in correspondences:
        # Forward: pixel → pitch
        vec = np.array([c.pixel_x, c.pixel_y, 1.0], dtype=np.float64)
        proj = H @ vec
        if abs(proj[2]) < 1e-9:
            errors_px.append(float("inf"))
            continue
        px_proj = proj[0] / proj[2]
        py_proj = proj[1] / proj[2]
        err_m = math.sqrt((px_proj - c.pitch_x) ** 2 + (py_proj - c.pitch_y) ** 2)

        # Back-project: pitch → pixel using H_inv
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


def compute_reprojection_errors(
    H: np.ndarray,
    pixel_points: np.ndarray,
    pitch_points: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Compute image-space reprojection errors for arrays of corresponding points.

    Convenient public function for use in tests and external callers.

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
# Coverage scoring
# ---------------------------------------------------------------------------


def _compute_coverage(
    pitch_points: List[Tuple[float, float]],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> Tuple[float, Dict[str, bool]]:
    """Score how well the calibration points cover the pitch.

    Coverage score: convex-hull area of inlier pitch points divided by pitch area.
    Maximum theoretical score is 1.0 when points span the full pitch.

    Quadrant coverage: True when ≥1 inlier point falls in each pitch third.
    """
    pts = [(x, y) for x, y in pitch_points
           if 0.0 <= x <= pitch_length and 0.0 <= y <= pitch_width]

    total_area = pitch_length * pitch_width  # 7140 m²

    quadrants: Dict[str, bool] = {
        "left_third":   any(x <= pitch_length / 3 for x, _ in pts),
        "middle_third": any(pitch_length / 3 < x <= 2 * pitch_length / 3 for x, _ in pts),
        "right_third":  any(x > 2 * pitch_length / 3 for x, _ in pts),
    }

    if len(pts) < 3:
        # Cannot compute a convex hull with fewer than 3 points
        return min(len(pts) / RECOMMENDED_CORRESPONDENCES, 1.0) * 0.1, quadrants

    try:
        pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts_arr)
        hull_area = float(cv2.contourArea(hull))
        score = min(hull_area / total_area, 1.0)
    except Exception:
        score = 0.0

    return score, quadrants


def _classify_scope(
    coverage_score: float,
    quadrant_coverage: Dict[str, bool],
    n_inliers: int,
) -> str:
    """Return a calibration scope label."""
    all_thirds = all(quadrant_coverage.values())
    if all_thirds and coverage_score >= 0.30 and n_inliers >= 8:
        return "full_pitch"
    if coverage_score >= 0.05 or n_inliers >= 4:
        return "partial"
    return "penalty_box_region"


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
        cal.add_correspondence(43, 408, 16.5, 13.84, label="left_pa_near_corner")
        # ... add ≥ 8 well-distributed correspondences ...
        result = cal.estimate()
        print(result.rms_reprojection_px, result.warnings)
        cal.save("data/homography_full_pitch.json")

    Parameters
    ----------
    pitch_length:
        Pitch length in meters (default 105.0).
    pitch_width:
        Pitch width in meters (default 68.0).
    ransac_threshold:
        RANSAC reprojection threshold in pixels.  Points with reprojection error
        above this in pixel space are treated as outliers.
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
        """Add an image↔pitch correspondence.

        Parameters
        ----------
        pixel_x, pixel_y:
            Image-space pixel coordinates of the landmark.
        pitch_x, pitch_y:
            Metric pitch coordinates (meters) in the 105×68 coordinate system.
        label:
            Optional human-readable name for the landmark.
        """
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
        """Estimate the homography from stored correspondences.

        Uses OpenCV RANSAC (``cv2.RANSAC``) to robustly reject outlier
        correspondences.  A DLT refinement is performed over inliers.

        Returns
        -------
        CalibrationResult
            Inspect ``is_valid``, ``warnings``, and ``rms_reprojection_px``.
        """
        n = len(self._correspondences)
        warnings: List[str] = []

        if n < MIN_CORRESPONDENCES:
            # Cannot even attempt estimation
            dummy_H = np.eye(3, dtype=np.float64)
            result = CalibrationResult(
                H=dummy_H,
                inlier_count=0,
                total_correspondences=n,
                rms_reprojection_px=float("inf"),
                max_reprojection_px=float("inf"),
                rms_reprojection_m=float("inf"),
                per_point_errors_px=[],
                inlier_mask=[],
                coverage_score=0.0,
                quadrant_coverage={
                    "left_third": False, "middle_third": False, "right_third": False
                },
                warnings=[
                    f"Insufficient correspondences: {n} provided, "
                    f"minimum required is {MIN_CORRESPONDENCES}."
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
                "(collinear, duplicated, or numerically unstable)."
            )
            dummy_H = np.eye(3, dtype=np.float64)
            result = CalibrationResult(
                H=dummy_H,
                inlier_count=0,
                total_correspondences=n,
                rms_reprojection_px=float("inf"),
                max_reprojection_px=float("inf"),
                rms_reprojection_m=float("inf"),
                per_point_errors_px=[float("inf")] * n,
                inlier_mask=[False] * n,
                coverage_score=0.0,
                quadrant_coverage={
                    "left_third": False, "middle_third": False, "right_third": False
                },
                warnings=warnings,
                calibration_scope="penalty_box_region",
                is_valid=False,
            )
            self._last_result = result
            return result

        inlier_mask = [bool(m[0]) for m in mask]
        inlier_count = sum(inlier_mask)
        H = H_raw.astype(np.float64)

        # --- Validate H is finite and non-degenerate ---
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
            # Check that the pitch corners back-project to plausible pixel coordinates
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
                        "Check landmark selection and coordinate convention."
                    )
                    is_valid = False
                else:
                    pix_back = back[:, :2] / scales[:, None]
                    # Sanity: all corners should land at positive pixel coordinates
                    if np.any(pix_back < -500) or np.any(pix_back > 1e5):
                        warnings.append(
                            "Pitch corner back-projections are implausibly large or negative. "
                            "Verify coordinate convention and landmark accuracy."
                        )
            except np.linalg.LinAlgError:
                warnings.append("Could not invert homography matrix for validation.")
                is_valid = False

        # --- Reprojection errors ---
        per_pt_errors, rms_px, max_px = _compute_reprojection_errors_px(H, self._correspondences)
        rms_reprojection_m = 0.0
        if rms_px < float("inf"):
            # Also report pitch-space error for intuitive understanding
            # (forward-projected errors at the calibration landmarks)
            pitch_errs = []
            for c in self._correspondences:
                vec = np.array([c.pixel_x, c.pixel_y, 1.0], dtype=np.float64)
                proj = H @ vec
                if abs(proj[2]) > 1e-9:
                    px_proj = proj[0] / proj[2]
                    py_proj = proj[1] / proj[2]
                    err_m = math.sqrt((px_proj - c.pitch_x) ** 2 + (py_proj - c.pitch_y) ** 2)
                    pitch_errs.append(err_m)
            if pitch_errs:
                rms_reprojection_m = math.sqrt(
                    sum(e ** 2 for e in pitch_errs) / len(pitch_errs)
                )

        if rms_px > 5.0:
            warnings.append(
                f"High RMS reprojection error: {rms_px:.2f} px. "
                "Consider verifying landmark positions or adding more distributed points."
            )

        # --- Coverage ---
        inlier_pitch_pts = [
            (self._correspondences[i].pitch_x, self._correspondences[i].pitch_y)
            for i, ok in enumerate(inlier_mask)
            if ok
        ]
        coverage_score, quadrant_coverage = _compute_coverage(
            inlier_pitch_pts, self.pitch_length, self.pitch_width
        )

        if n < RECOMMENDED_CORRESPONDENCES:
            warnings.append(
                f"Only {n} correspondences provided. "
                f"≥ {RECOMMENDED_CORRESPONDENCES} well-distributed correspondences "
                "are recommended for reliable full-pitch coverage."
            )
        if inlier_count < MIN_CORRESPONDENCES:
            warnings.append(
                f"Only {inlier_count} RANSAC inliers from {n} correspondences. "
                "Calibration quality may be low."
            )
        if not quadrant_coverage.get("middle_third", False):
            warnings.append(
                "No inlier points in the middle third of the pitch. "
                "Projection accuracy near midfield is not validated."
            )
        if not quadrant_coverage.get("right_third", False):
            warnings.append(
                "No inlier points in the right third of the pitch. "
                "Projection accuracy in this region is extrapolated."
            )

        calibration_scope = _classify_scope(coverage_score, quadrant_coverage, inlier_count)

        result = CalibrationResult(
            H=H,
            inlier_count=inlier_count,
            total_correspondences=n,
            rms_reprojection_px=rms_px,
            max_reprojection_px=max_px,
            rms_reprojection_m=rms_reprojection_m,
            per_point_errors_px=per_pt_errors,
            inlier_mask=inlier_mask,
            coverage_score=coverage_score,
            quadrant_coverage=quadrant_coverage,
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
        """Save calibration to a metadata-rich JSON file.

        The JSON contains both the 3×3 homography matrix (key ``"H"``) and
        validation/coverage metadata so that ``PitchHomography`` can distinguish
        between the legacy format (bare matrix) and the rich format.

        Parameters
        ----------
        path:
            Output file path.
        extra_metadata:
            Additional key/value pairs to merge into the metadata block.

        Returns
        -------
        Path to the saved file.
        """
        if self._last_result is None:
            raise RuntimeError(
                "Call estimate() before save(). No calibration result available."
            )

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
            "correspondences": corr_list,
            "validation": result.to_dict(),
            "coordinate_convention": (
                "Origin (0,0) at left goal-line/near touchline intersection. "
                "X axis along pitch length (0→105 m), Y axis along pitch width (0→68 m)."
            ),
            "camera_motion_warning": (
                "This homography assumes a static camera and flat pitch plane. "
                "Camera pan, tilt, zoom, or cuts will introduce errors not captured "
                "by this calibration."
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
        """Restore a calibrator from a previously saved rich JSON file.

        Note: the loaded calibrator's ``_last_result`` is a reconstructed
        ``CalibrationResult`` populated from the saved metadata.  Correspondences
        are also restored so the user can add more points and re-estimate.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "format" not in data or data["format"] != "courtvision_homography_v1":
            raise ValueError(
                f"File at {path} does not appear to be a courtvision_homography_v1 file. "
                "Use PitchHomography for the legacy matrix-only format."
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

        # Reconstruct last result from saved validation block
        H = np.array(data["H"], dtype=np.float64)
        v = data.get("validation", {})
        cal._last_result = CalibrationResult(
            H=H,
            inlier_count=v.get("inlier_count", 0),
            total_correspondences=v.get("total_correspondences", 0),
            rms_reprojection_px=v.get("rms_reprojection_px", float("inf")),
            max_reprojection_px=v.get("max_reprojection_px", float("inf")),
            rms_reprojection_m=v.get("rms_reprojection_m", float("inf")),
            per_point_errors_px=v.get("per_point_errors_px", []),
            inlier_mask=v.get("inlier_mask", []),
            coverage_score=v.get("coverage_score", 0.0),
            quadrant_coverage=v.get("quadrant_coverage", {}),
            warnings=v.get("warnings", []),
            calibration_scope=v.get("calibration_scope", "partial"),
            is_valid=v.get("is_valid", False),
        )
        return cal
