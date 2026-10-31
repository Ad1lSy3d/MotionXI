"""
tests/test_calibration.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for full-pitch and partial homography calibration, error metrics,
calibrated-zone boundary tracking, and UI landmark mappings.

All tests use synthetic data so results are deterministic and independent of
real match footage or external video files.
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
from scripts.calibrate_pitch_full import MENU_ITEMS, HOTKEY_TO_LABEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def _make_synthetic_H() -> np.ndarray:
    """Return an invertible projective transform for synthetic testing."""
    src = np.array(
        [[100, 700], [1180, 700], [900, 250], [380, 250]],
        dtype=np.float32,
    )
    dst = np.array(
        [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(src, dst, 0)
    return H.astype(np.float64)


def _generate_correspondences_from_H(
    H: np.ndarray,
    n: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate n synthetic (pixel, pitch) correspondences consistent with H."""
    np.random.seed(42)
    pitch_pts = np.column_stack([
        np.random.uniform(5.0, PITCH_LENGTH - 5.0, n),
        np.random.uniform(5.0, PITCH_WIDTH - 5.0, n),
    ])
    H_inv = np.linalg.inv(H)
    pixel_pts = []
    for pp in pitch_pts:
        hvec = H_inv @ np.array([pp[0], pp[1], 1.0])
        scale = hvec[2] if abs(hvec[2]) > 1e-9 else 1e-9
        pixel_pts.append([hvec[0] / scale, hvec[1] / scale])
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
# Core Estimation & Accuracy Tests
# ---------------------------------------------------------------------------


def test_synthetic_homography_recovery_accuracy():
    """A known H should be recovered within tight inlier RMS thresholds."""
    H_true = _make_synthetic_H()
    cal = _calibrator_from_H(H_true, n=10)
    result = cal.estimate()

    assert result.is_valid, f"Expected valid calibration. Warnings: {result.warnings}"
    assert result.inlier_count >= 8
    assert result.rms_reprojection_px < 1.0
    assert result.rms_reprojection_m < 0.5


def test_reprojection_errors_near_zero_for_known_H():
    H_true = _make_synthetic_H()
    pixel_pts, pitch_pts = _generate_correspondences_from_H(H_true, n=5)
    per_pt, rms, max_err = compute_reprojection_errors(H_true, pixel_pts, pitch_pts)

    assert rms < 1.0
    assert max_err < 2.0
    assert len(per_pt) == 5


def test_reprojection_errors_empty_input():
    H = np.eye(3, dtype=np.float64)
    per_pt, rms, max_err = compute_reprojection_errors(
        H, np.zeros((0, 2)), np.zeros((0, 2))
    )
    assert rms == float("inf")
    assert len(per_pt) == 0


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


def test_collinear_points_produce_degenerate_warning():
    """Collinear points should produce a degenerate or invalid homography."""
    cal = HomographyCalibrator()
    for i in range(6):
        cal.add_correspondence(
            pixel_x=100 + i * 50, pixel_y=400,
            pitch_x=float(i * 10), pitch_y=float(i * 5),
        )
    result = cal.estimate()
    assert (not result.is_valid) or bool(result.warnings)


# ---------------------------------------------------------------------------
# Issue 1 Regression: Partial Calibration & Min/Max Bounds Tracking
# ---------------------------------------------------------------------------


def test_partial_calibration_right_side_of_pitch(tmp_path: Path):
    """
    REGRESSION TEST FOR ISSUE 1:
    A calibration concentrated on the right side of the pitch must track both
    minimum and maximum X bounds. Uncalibrated regions (left side and midfield)
    must NOT be marked as within the calibrated zone.
    """
    H_true = _make_synthetic_H()
    H_inv = np.linalg.inv(H_true)

    # 6 landmarks strictly inside the right penalty area / goal area (X in [88.5, 105.0])
    right_pitch_points = [
        (88.5, 13.84),
        (88.5, 54.16),
        (105.0, 13.84),
        (105.0, 54.16),
        (94.0, 34.0),
        (99.5, 24.84),
    ]

    cal = HomographyCalibrator()
    for idx, (px, py) in enumerate(right_pitch_points):
        hvec = H_inv @ np.array([px, py, 1.0])
        u, v = hvec[0] / hvec[2], hvec[1] / hvec[2]
        cal.add_correspondence(u, v, px, py, label=f"right_lm_{idx}")

    result = cal.estimate()
    assert result.is_valid
    assert result.calibration_scope in ("partial", "penalty_box_region")

    # Check calibrated_zone_m tracks both minimum and maximum X bounds!
    zone = result.calibrated_zone_m
    x_range = zone["x_range"]
    y_range = zone["y_range"]

    # Minimum X must NOT be 0.0! It should be around 88.5 - 5.0 = 83.5 m
    assert x_range[0] >= 80.0, f"Expected x_min >= 80.0, got {x_range[0]}"
    assert x_range[1] <= 105.0, f"Expected x_max <= 105.0, got {x_range[1]}"
    assert y_range[0] <= 15.0
    assert y_range[1] >= 50.0

    # Save and load with PitchHomography
    save_path = tmp_path / "right_box_cal.json"
    cal.save(save_path)
    ph = PitchHomography(save_path)

    assert ph.calibrated_x_min >= 80.0
    assert ph.calibrated_x_max == pytest.approx(105.0, abs=0.5)

    # Inside right penalty box -> within calibrated zone
    assert ph.is_within_calibrated_zone(94.0, 34.0, margin_meters=0.0) is True
    assert ph.is_within_calibrated_zone(90.0, 20.0, margin_meters=0.0) is True

    # Midfield point (X=52.5) -> MUST NOT be in calibrated zone!
    assert ph.is_within_calibrated_zone(52.5, 34.0, margin_meters=0.0) is False

    # Left penalty box point (X=10.0) -> MUST NOT be in calibrated zone!
    assert ph.is_within_calibrated_zone(10.0, 34.0, margin_meters=0.0) is False
    assert ph.is_within_calibrated_zone(0.0, 34.0, margin_meters=0.0) is False

    # Test margin tolerance
    # Point at X=81.0 is outside with margin=0.0, but inside with margin=5.0
    assert ph.is_within_calibrated_zone(81.0, 34.0, margin_meters=0.0) is False
    assert ph.is_within_calibrated_zone(81.0, 34.0, margin_meters=5.0) is True


# ---------------------------------------------------------------------------
# Issue 2 Regression: Conservative Scope & Insufficient Spatial Coverage
# ---------------------------------------------------------------------------


def test_insufficient_spatial_coverage_remains_partial():
    """
    REGRESSION TEST FOR ISSUE 2:
    Landmarks that touch all three thirds along a narrow line/corridor or with
    < 50% coverage must NOT be classified as full_pitch. They must remain partial.
    """
    H_true = _make_synthetic_H()
    H_inv = np.linalg.inv(H_true)

    # 6 points touching left (10m), middle (52.5m), and right (95m),
    # but all placed along a narrow line near center line (Y in [32, 36])
    narrow_pitch_points = [
        (10.0, 33.0),
        (15.0, 35.0),
        (52.5, 33.5),
        (55.0, 34.5),
        (90.0, 33.0),
        (95.0, 35.0),
    ]

    cal = HomographyCalibrator()
    for idx, (px, py) in enumerate(narrow_pitch_points):
        hvec = H_inv @ np.array([px, py, 1.0])
        u, v = hvec[0] / hvec[2], hvec[1] / hvec[2]
        cal.add_correspondence(u, v, px, py, label=f"corridor_{idx}")

    result = cal.estimate()
    # Must remain partial because width span is tiny and coverage score is near 0
    assert result.calibration_scope == "partial", (
        f"Expected 'partial' scope for narrow corridor, got '{result.calibration_scope}'"
    )
    # Must contain explicit warning regarding coverage and extrapolation
    warning_text = " ".join(result.warnings).lower()
    assert "narrow pitch width" in warning_text or "partial" in warning_text


def test_full_pitch_scope_when_conservative_criteria_met():
    """
    When >= 10 well-distributed landmarks span both ends, both touchlines,
    and center with coverage >= 50%, scope is 'full_pitch', and the accuracy
    limitation warning is explicitly attached.
    """
    H_true = _make_synthetic_H()
    H_inv = np.linalg.inv(H_true)

    # 12 well-distributed landmarks covering the four corners, both penalty areas,
    # halfway line, center circle, and touchlines
    full_pitch_points = [
        (0.0, 0.0),
        (0.0, 68.0),
        (105.0, 0.0),
        (105.0, 68.0),
        (16.5, 13.84),
        (16.5, 54.16),
        (88.5, 13.84),
        (88.5, 54.16),
        (52.5, 0.0),
        (52.5, 68.0),
        (52.5, 34.0),
        (11.0, 34.0),
    ]

    cal = HomographyCalibrator()
    for idx, (px, py) in enumerate(full_pitch_points):
        hvec = H_inv @ np.array([px, py, 1.0])
        u, v = hvec[0] / hvec[2], hvec[1] / hvec[2]
        cal.add_correspondence(u, v, px, py, label=f"full_{idx}")

    result = cal.estimate()
    assert result.is_valid
    assert result.inlier_count == 12
    assert result.coverage_score >= 0.80
    assert result.calibration_scope == "full_pitch"

    # Even for full_pitch, warning must state that geometric coverage alone
    # does not guarantee real-world projection accuracy across the entire pitch
    warning_text = " ".join(result.warnings)
    assert "does not guarantee real-world projection accuracy" in warning_text


# ---------------------------------------------------------------------------
# Issue 3 Regression: Error Metrics Distinction (Inlier vs Overall)
# ---------------------------------------------------------------------------


def test_inlier_and_overall_error_metrics_distinction():
    """
    REGRESSION TEST FOR ISSUE 3:
    Reprojection error (px) and forward error (m) must be distinguished.
    Inlier residuals must reflect the fitted model, while overall residuals
    reflect corrupted/outlier points.
    """
    H_true = _make_synthetic_H()
    pixel_pts, pitch_pts = _generate_correspondences_from_H(H_true, n=8)

    cal = HomographyCalibrator()
    for pix, pit in zip(pixel_pts, pitch_pts):
        cal.add_correspondence(pix[0], pix[1], pit[0], pit[1])

    # Inject an intentional gross outlier (pixel off by 200 pixels)
    cal.add_correspondence(pixel_pts[0][0] + 200.0, pixel_pts[0][1] + 150.0,
                           pitch_pts[0][0], pitch_pts[0][1], label="gross_outlier")

    result = cal.estimate()
    assert result.is_valid
    assert result.inlier_count == 8
    assert result.total_correspondences == 9

    # Inlier image-space RMS should remain small
    assert result.rms_reprojection_px < 2.0
    # Overall image-space RMS should be heavily elevated due to outlier
    assert result.overall_rms_reprojection_px > 10.0

    # Inlier forward-projection RMS should remain small (< 1.0 m)
    assert result.rms_reprojection_m < 1.0
    # Overall forward-projection RMS should be elevated
    assert result.overall_rms_reprojection_m > 5.0

    # Verify metric descriptions in to_dict()
    d = result.to_dict()
    assert "metric_definitions" in d
    assert "rms_reprojection_px" in d["metric_definitions"]
    assert "rms_reprojection_m" in d["metric_definitions"]
    assert "limitation_note" in d["metric_definitions"]


# ---------------------------------------------------------------------------
# Issue 4 Regression: Landmark Menu Mapping & Reachability
# ---------------------------------------------------------------------------


def test_landmark_menu_mapping_validity_and_reachability():
    """
    REGRESSION TEST FOR ISSUE 4:
    Every landmark hotkey must:
    1. Be a single character (len == 1).
    2. Be unique across the menu.
    3. Not conflict with reserved controls ('q', 'r', 'e', 's').
    4. Reference a valid landmark in PITCH_LANDMARKS.
    """
    reserved_controls = {"q", "r", "e", "s"}
    seen_hotkeys = set()

    for item in MENU_ITEMS:
        assert len(item) == 3
        hotkey, label, desc = item

        # Single character check
        assert len(hotkey) == 1, f"Hotkey '{hotkey}' must be single character"

        # Unique hotkey check
        assert hotkey not in seen_hotkeys, f"Duplicate hotkey '{hotkey}'"
        seen_hotkeys.add(hotkey)

        # No collision with control keys
        assert hotkey not in reserved_controls, (
            f"Hotkey '{hotkey}' conflicts with reserved control key in {reserved_controls}"
        )

        # Landmark label exists in PITCH_LANDMARKS
        assert label in PITCH_LANDMARKS, (
            f"Menu label '{label}' not found in PITCH_LANDMARKS catalog"
        )

    # HOTKEY_TO_LABEL dictionary must match MENU_ITEMS
    assert len(HOTKEY_TO_LABEL) == len(MENU_ITEMS)


# ---------------------------------------------------------------------------
# Coordinate Boundary & Serialization Tests
# ---------------------------------------------------------------------------


def test_project_point_returns_nan_false_for_singular_scale():
    H_bad = np.zeros((3, 3), dtype=np.float64)
    H_bad[0, 0] = 1.0
    H_bad[1, 1] = 1.0
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


def test_rich_json_round_trip(tmp_path: Path):
    H_true = _make_synthetic_H()
    cal = _calibrator_from_H(H_true, n=10)
    result = cal.estimate()
    assert result.is_valid

    save_path = tmp_path / "test_calibration.json"
    cal.save(save_path)

    with open(save_path) as f:
        data = json.load(f)
    assert data["format"] == "courtvision_homography_v1"
    assert "H" in data
    assert "calibrated_zone_m" in data
    assert "correspondences" in data
    assert "validation" in data

    ph = PitchHomography(save_path)
    assert ph.H.shape == (3, 3)
    assert np.all(np.isfinite(ph.H))

    pixel_pts, pitch_pts = _generate_correspondences_from_H(H_true, n=5)
    for pix, pit in zip(pixel_pts, pitch_pts):
        px, py, valid = ph.project_point(pix[0], pix[1])
        assert valid is True
        assert abs(px - pit[0]) < 1.0
        assert abs(py - pit[1]) < 1.0


def test_calibrator_load_round_trip(tmp_path: Path):
    H_true = _make_synthetic_H()
    cal = _calibrator_from_H(H_true, n=8)
    cal.estimate()
    save_path = tmp_path / "cal_roundtrip.json"
    cal.save(save_path)

    cal2 = HomographyCalibrator.load(save_path)
    assert len(cal2.correspondences) == 8
    assert cal2._last_result is not None
    assert cal2._last_result.is_valid is True


def test_legacy_matrix_loads_and_projects():
    h_path = Path("data/homography_matrix.json")
    assert h_path.exists(), "data/homography_matrix.json must exist for legacy test"

    ph = PitchHomography(h_path)
    assert ph.H.shape == (3, 3)
    assert np.all(np.isfinite(ph.H))

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
    assert ph.is_within_calibrated_zone(8.0, 34.0) is True
    assert ph.is_within_calibrated_zone(90.0, 34.0) is False


def test_full_pitch_calibration_marks_entire_pitch_as_calibrated(tmp_path: Path):
    H_true = _make_synthetic_H()
    cal = _calibrator_from_H(H_true, n=10)
    result = cal.estimate()
    save_path = tmp_path / "full_pitch_cal.json"
    cal.save(save_path)

    with open(save_path) as f:
        data = json.load(f)
    data["full_pitch_calibration"] = True
    data["calibration_scope"] = "full_pitch"
    data["validation"]["calibration_scope"] = "full_pitch"
    data["calibrated_zone_m"] = {"x_range": [0.0, 105.0], "y_range": [0.0, 68.0]}
    with open(save_path, "w") as f:
        json.dump(data, f)

    ph = PitchHomography(save_path)
    assert ph._is_full_pitch is True
    assert ph.is_within_calibrated_zone(52.5, 34.0, margin_meters=0.0) is True
    assert ph.is_within_calibrated_zone(90.0, 50.0, margin_meters=0.0) is True


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


def test_calibration_result_to_dict_is_json_serializable():
    H_true = _make_synthetic_H()
    cal = _calibrator_from_H(H_true, n=6)
    result = cal.estimate()
    d = result.to_dict()

    serialized = json.dumps(d)
    assert serialized
    assert "inlier_count" in d
    assert "rms_reprojection_px" in d
    assert "rms_reprojection_m" in d
    assert "overall_rms_reprojection_px" in d
    assert "calibrated_zone_m" in d
    assert "warnings" in d
    assert "calibration_scope" in d
