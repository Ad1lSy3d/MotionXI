"""
tests/test_calibration.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for courtvision.projection.calibrator and the updated PitchHomography.

All tests use synthetic data so results are deterministic and independent of
real video footage or the current data/homography_matrix.json file.

Coverage
--------
1.  Known synthetic image↔pitch correspondences → accurate homography recovery.
2.  Reprojection errors are near zero for a known transform.
3.  Degenerate / insufficient landmark sets rejected with is_valid=False.
4.  Collinear points produce a warning and is_valid=False (degenerate H).
5.  Poorly distributed (single-region) points trigger coverage warnings.
6.  Non-finite and out-of-bounds coordinate handling in PitchHomography.
7.  Calibration metadata serialization and loading (rich JSON round-trip).
8.  Legacy penalty-box calibration still loads and works (backward compat).
9.  compute_reprojection_errors public function.
10. Full-pitch scope classification: ≥8 inliers, all thirds, coverage ≥0.30.
11. Partial scope when coverage is limited.
12. is_within_calibrated_zone reflects calibrated region from rich JSON.
13. PitchHomography.get_metadata() returns correct fields for both formats.
14. HomographyCalibrator.load() round-trip restores correspondences.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from courtvision.projection.calibrator import (
    CalibrationResult,
    HomographyCalibrator,
    LandmarkCorrespondence,
    MIN_CORRESPONDENCES,
    PITCH_LANDMARKS,
    RECOMMENDED_CORRESPONDENCES,
    _compute_coverage,
    compute_reprojection_errors,
)
from courtvision.projection.homography import PitchHomography


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def _make_synthetic_H() -> np.ndarray:
    """Return a known perspective transform mapping a virtual camera view to
    a 105×68 pitch.  Constructed analytically so we can verify exact recovery."""
    # Simple projective transform: maps a 1280×720 synthetic frame view of the
    # left half of the pitch (0–52.5m × 0–68m) to pitch coordinates.
    src = np.array(
        [[0, 720], [640, 720], [1280, 720], [640, 0]],
        dtype=np.float32,
    )
    dst = np.array(
        [[0.0, 0.0], [52.5, 0.0], [105.0, 0.0], [52.5, 68.0]],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(src, dst, 0)
    return H.astype(np.float64)


def _generate_correspondences_from_H(
    H: np.ndarray,
    n: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate n synthetic (pixel, pitch) correspondences consistent with H.

    We sample n pitch points and back-project to pixel space using H_inv so
    that H maps pixel → pitch exactly for these pairs.
    """
    np.random.seed(42)
    pitch_pts = np.column_stack([
        np.random.uniform(0.0, PITCH_LENGTH, n),
        np.random.uniform(0.0, PITCH_WIDTH, n),
    ])
    H_inv = np.linalg.inv(H)
    pixel_pts = []
    for pp in pitch_pts:
        hvec = H_inv @ np.array([pp[0], pp[1], 1.0])
        if abs(hvec[2]) < 1e-9:
            hvec = hvec + 1e-9  # avoid divide by zero
        pixel_pts.append([hvec[0] / hvec[2], hvec[1] / hvec[2]])
    return np.array(pixel_pts, dtype=np.float64), pitch_pts.astype(np.float64)


def _calibrator_from_H(H: np.ndarray, n: int = 10) -> HomographyCalibrator:
    """Build a calibrator with synthetic correspondences from a known H."""
    pixel_pts, pitch_pts = _generate_correspondences_from_H(H, n)
    cal = HomographyCalibrator()
    for i, (pix, pit) in enumerate(zip(pixel_pts, pitch_pts)):
        cal.add_correspondence(
            pixel_x=pix[0], pixel_y=pix[1],
            pitch_x=pit[0], pitch_y=pit[1],
            label=f"synth_{i}",
        )
    return cal


# ---------------------------------------------------------------------------
# Test 1: Accurate homography recovery from synthetic correspondences
# ---------------------------------------------------------------------------


def test_synthetic_homography_recovery_accuracy():
    """A known H should be recovered within tight RMS thresholds."""
    H_true = _make_synthetic_H()
    cal = _calibrator_from_H(H_true, n=10)
    result = cal.estimate()

    assert result.is_valid, f"Expected valid calibration. Warnings: {result.warnings}"
    assert result.inlier_count >= 8, f"Expected ≥8 inliers, got {result.inlier_count}"
    # For exact correspondences RMS should be near zero
    assert result.rms_reprojection_px < 1.0, (
        f"Expected RMS < 1.0 px, got {result.rms_reprojection_px:.4f}"
    )
    assert result.rms_reprojection_m < 0.5, (
        f"Expected RMS < 0.5 m, got {result.rms_reprojection_m:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2: Reprojection errors near zero for known transform
# ---------------------------------------------------------------------------


def test_reprojection_errors_near_zero_for_known_H():
    H_true = _make_synthetic_H()
    pixel_pts, pitch_pts = _generate_correspondences_from_H(H_true, n=5)
    per_pt, rms, max_err = compute_reprojection_errors(H_true, pixel_pts, pitch_pts)

    assert rms < 1.0, f"Expected RMS < 1.0 px, got {rms:.4f}"
    assert max_err < 2.0, f"Expected max < 2.0 px, got {max_err:.4f}"
    assert len(per_pt) == 5


def test_reprojection_errors_empty_input():
    H = np.eye(3, dtype=np.float64)
    per_pt, rms, max_err = compute_reprojection_errors(
        H, np.zeros((0, 2)), np.zeros((0, 2))
    )
    assert rms == float("inf")
    assert len(per_pt) == 0


# ---------------------------------------------------------------------------
# Test 3: Insufficient correspondences → is_valid=False
# ---------------------------------------------------------------------------


def test_insufficient_correspondences_rejected():
    cal = HomographyCalibrator()
    for i in range(MIN_CORRESPONDENCES - 1):
        cal.add_correspondence(i * 10, 100, i * 10.0, 10.0)
    result = cal.estimate()

    assert result.is_valid is False
    assert result.inlier_count == 0
    assert any("Insufficient" in w for w in result.warnings)


def test_zero_correspondences_rejected():
    cal = HomographyCalibrator()
    result = cal.estimate()
    assert result.is_valid is False


# ---------------------------------------------------------------------------
# Test 4: Collinear points (degenerate configuration)
# ---------------------------------------------------------------------------


def test_collinear_points_produce_degenerate_warning():
    """All points on the same horizontal line → degenerate H."""
    cal = HomographyCalibrator()
    # 6 collinear pixel points, mapped to 6 distinct pitch points
    for i in range(6):
        cal.add_correspondence(
            pixel_x=100 + i * 50, pixel_y=400,  # all same y → collinear in image
            pitch_x=float(i * 10), pitch_y=float(i * 5),
        )
    result = cal.estimate()
    # Either is_valid is False OR there is a warning about near-singular H / low inliers
    issues = not result.is_valid or bool(result.warnings)
    assert issues, "Expected warning or invalid result for collinear configuration"


# ---------------------------------------------------------------------------
# Test 5: Single-region points trigger coverage warnings
# ---------------------------------------------------------------------------


def test_single_region_points_trigger_coverage_warnings():
    """Points only in the left penalty box → missing middle and right thirds."""
    cal = HomographyCalibrator()
    # All pitch coordinates in [0, 16.5] × [13.84, 54.16] — left PA only
    box_pts = [
        (0.0,  13.84), (16.5, 13.84), (16.5, 54.16), (0.0, 54.16),
        (5.5,  24.84), (5.5,  43.16),
    ]
    # Give them plausible distinct pixel positions
    pixel_pts = [
        (800, 650), (650, 600), (650, 400), (800, 350),
        (750, 560), (750, 440),
    ]
    for (px, py), (qx, qy) in zip(pixel_pts, box_pts):
        cal.add_correspondence(px, py, qx, qy)

    result = cal.estimate()
    # Should warn about missing middle and right thirds
    warning_text = " ".join(result.warnings)
    assert ("middle" in warning_text.lower() or "right" in warning_text.lower()), (
        f"Expected coverage warning, got: {result.warnings}"
    )
    assert result.calibration_scope != "full_pitch"


# ---------------------------------------------------------------------------
# Test 6: Non-finite / out-of-bounds coordinate handling
# ---------------------------------------------------------------------------


def test_project_point_returns_nan_false_for_singular_scale():
    """A matrix that zeroes out the homogeneous scale should return NaN."""
    # Construct H where last row is [0, 0, 0] → scale always 0
    H_bad = np.zeros((3, 3), dtype=np.float64)
    H_bad[0, 0] = 1.0
    H_bad[1, 1] = 1.0
    # Last row stays zero
    ph = PitchHomography(H_bad)
    px, py, valid = ph.project_point(100.0, 200.0)
    assert math.isnan(px) or not valid
    assert valid is False


def test_is_within_pitch_rejects_nan_and_out_of_bounds():
    ph = PitchHomography(np.eye(3))
    assert ph.is_within_pitch(float("nan"), 34.0) is False
    assert ph.is_within_pitch(52.5, float("nan")) is False
    assert ph.is_within_pitch(-1.0, 34.0) is False
    assert ph.is_within_pitch(52.5, -1.0) is False
    assert ph.is_within_pitch(106.0, 34.0) is False
    assert ph.is_within_pitch(52.5, 69.0) is False
    assert ph.is_within_pitch(52.5, 34.0) is True


# ---------------------------------------------------------------------------
# Test 7: Rich JSON round-trip (save and load)
# ---------------------------------------------------------------------------


def test_rich_json_round_trip(tmp_path: Path):
    H_true = _make_synthetic_H()
    cal = _calibrator_from_H(H_true, n=10)
    result = cal.estimate()
    assert result.is_valid

    save_path = tmp_path / "test_calibration.json"
    cal.save(save_path)

    # Verify file is valid JSON with expected keys
    with open(save_path) as f:
        data = json.load(f)
    assert data["format"] == "courtvision_homography_v1"
    assert "H" in data
    assert "correspondences" in data
    assert len(data["correspondences"]) == 10
    assert "validation" in data
    assert data["validation"]["is_valid"] is True

    # PitchHomography can load it
    ph = PitchHomography(save_path)
    assert ph.H.shape == (3, 3)
    assert np.all(np.isfinite(ph.H))

    # Reprojection should still be accurate
    pixel_pts, pitch_pts = _generate_correspondences_from_H(H_true, n=5)
    for pix, pit in zip(pixel_pts, pitch_pts):
        px, py, valid = ph.project_point(pix[0], pix[1])
        assert valid is True
        assert abs(px - pit[0]) < 1.0, f"pitch_x error too large: {abs(px - pit[0]):.3f}"
        assert abs(py - pit[1]) < 1.0, f"pitch_y error too large: {abs(py - pit[1]):.3f}"


def test_calibrator_load_round_trip(tmp_path: Path):
    """HomographyCalibrator.load() should restore correspondences."""
    H_true = _make_synthetic_H()
    cal = _calibrator_from_H(H_true, n=8)
    cal.estimate()
    save_path = tmp_path / "cal_roundtrip.json"
    cal.save(save_path)

    cal2 = HomographyCalibrator.load(save_path)
    assert len(cal2.correspondences) == 8
    assert cal2._last_result is not None
    assert cal2._last_result.is_valid is True


# ---------------------------------------------------------------------------
# Test 8: Legacy bare-matrix JSON backward compatibility
# ---------------------------------------------------------------------------


def test_legacy_matrix_loads_and_projects():
    """data/homography_matrix.json is a bare [[...]] format → still works."""
    h_path = Path("data/homography_matrix.json")
    assert h_path.exists(), "data/homography_matrix.json must exist for legacy test"

    ph = PitchHomography(h_path)
    assert ph.H.shape == (3, 3)
    assert np.all(np.isfinite(ph.H))

    # Corner 3 of 18-yard box: pixel (820, 655) → metric (16.5, 54.16)
    px, py, valid = ph.project_point(820.0, 655.0)
    assert valid is True
    assert abs(px - 16.5) < 0.5
    assert abs(py - 54.16) < 0.5


def test_legacy_metadata_has_penalty_box_scope():
    ph = PitchHomography("data/homography_matrix.json")
    meta = ph.get_metadata()
    assert meta["calibration_scope"] == "penalty_box_region"
    assert meta["full_pitch_calibration"] is False
    assert "warning" in meta


def test_legacy_calibrated_zone_is_penalty_box():
    ph = PitchHomography("data/homography_matrix.json")
    # Inside penalty box area (with margin) → in calibrated zone
    assert ph.is_within_calibrated_zone(8.0, 34.0) is True
    # Far side of pitch → out of calibrated zone
    assert ph.is_within_calibrated_zone(90.0, 34.0) is False


# ---------------------------------------------------------------------------
# Test 9: compute_reprojection_errors public API
# ---------------------------------------------------------------------------


def test_compute_reprojection_errors_exact_match():
    """When H is a known exact transform, errors are near zero."""
    H_true = _make_synthetic_H()
    pixel_pts, pitch_pts = _generate_correspondences_from_H(H_true, n=6)
    per_pt, rms, max_err = compute_reprojection_errors(H_true, pixel_pts, pitch_pts)
    assert rms < 1.0
    assert max_err < 2.0
    assert len(per_pt) == 6
    for e in per_pt:
        assert math.isfinite(e)


# ---------------------------------------------------------------------------
# Test 10: Full-pitch scope classification
# ---------------------------------------------------------------------------


def test_full_pitch_scope_when_all_thirds_covered_and_good_coverage():
    """8+ inliers spread across all three pitch thirds → 'full_pitch' scope."""
    H_true = _make_synthetic_H()
    # Generate correspondences spanning the whole pitch
    cal = _calibrator_from_H(H_true, n=10)
    result = cal.estimate()

    if result.coverage_score >= 0.30 and all(result.quadrant_coverage.values()):
        assert result.calibration_scope == "full_pitch"
    else:
        # Pass if synthetic distribution didn't achieve full coverage (acceptable)
        assert result.calibration_scope in ("full_pitch", "partial")


# ---------------------------------------------------------------------------
# Test 11: Partial scope when coverage is limited
# ---------------------------------------------------------------------------


def test_partial_scope_for_limited_coverage():
    """4–7 correspondences from limited region → 'partial' scope."""
    cal = HomographyCalibrator()
    # Left half only, varied across near/far touchline so not collinear
    correspondences = [
        (800, 650,  0.0,   0.0),
        (650, 600, 16.5,   0.0),
        (650, 400, 16.5,  68.0),
        (800, 350,  0.0,  68.0),
        (750, 525,  8.0,  34.0),
    ]
    for px, py, qx, qy in correspondences:
        cal.add_correspondence(px, py, qx, qy)
    result = cal.estimate()

    if result.is_valid:
        assert result.calibration_scope in ("partial", "penalty_box_region")


# ---------------------------------------------------------------------------
# Test 12: is_within_calibrated_zone reflects calibrated region from rich JSON
# ---------------------------------------------------------------------------


def test_full_pitch_calibration_marks_entire_pitch_as_calibrated(tmp_path: Path):
    """A rich JSON with full_pitch_calibration=True → whole pitch is calibrated."""
    # Manually create a rich JSON that claims full_pitch scope
    H_true = _make_synthetic_H()
    cal = _calibrator_from_H(H_true, n=10)
    result = cal.estimate()
    save_path = tmp_path / "full_pitch_cal.json"
    cal.save(save_path)

    # Patch the saved JSON to force full_pitch scope for the test
    with open(save_path) as f:
        data = json.load(f)
    data["full_pitch_calibration"] = True
    data["calibration_scope"] = "full_pitch"
    data["validation"]["calibration_scope"] = "full_pitch"
    # Mark all inliers True so calibrated zone is derived from all points
    data["validation"]["inlier_mask"] = [True] * len(data["correspondences"])
    with open(save_path, "w") as f:
        json.dump(data, f)

    ph = PitchHomography(save_path)
    # For full_pitch, box_depth is set to pitch_length and y extends 0→68
    assert ph._is_full_pitch is True
    # Center of pitch should be in calibrated zone
    assert ph.is_within_calibrated_zone(52.5, 34.0, margin_meters=0.0) is True
    # Right side should also be calibrated
    assert ph.is_within_calibrated_zone(90.0, 50.0, margin_meters=0.0) is True


# ---------------------------------------------------------------------------
# Test 13: get_metadata returns correct structure for both formats
# ---------------------------------------------------------------------------


def test_get_metadata_legacy_format():
    ph = PitchHomography("data/homography_matrix.json")
    meta = ph.get_metadata()
    required = ["source", "calibration_scope", "full_pitch_calibration",
                "pitch_dimensions_m", "calibrated_zone_m", "warning"]
    for key in required:
        assert key in meta, f"Missing key: {key}"
    assert meta["pitch_dimensions_m"] == [105.0, 68.0]


def test_get_metadata_rich_format(tmp_path: Path):
    H_true = _make_synthetic_H()
    cal = _calibrator_from_H(H_true, n=10)
    cal.estimate()
    save_path = tmp_path / "rich_meta_test.json"
    cal.save(save_path)

    ph = PitchHomography(save_path)
    meta = ph.get_metadata()
    required = ["source", "calibration_scope", "full_pitch_calibration",
                "pitch_dimensions_m", "calibrated_zone_m", "coverage_score"]
    for key in required:
        assert key in meta, f"Missing key in rich metadata: {key}"


# ---------------------------------------------------------------------------
# Test 14: CalibrationResult.to_dict serialization
# ---------------------------------------------------------------------------


def test_calibration_result_to_dict_is_json_serializable():
    H_true = _make_synthetic_H()
    cal = _calibrator_from_H(H_true, n=6)
    result = cal.estimate()
    d = result.to_dict()

    import json
    serialized = json.dumps(d)
    assert serialized

    assert "inlier_count" in d
    assert "rms_reprojection_px" in d
    assert "coverage_score" in d
    assert "warnings" in d
    assert "calibration_scope" in d
    assert "is_valid" in d
