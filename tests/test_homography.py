from pathlib import Path
import numpy as np
import pytest
from courtvision.projection.homography import PitchHomography


def test_homography_matrix_loads_and_shape():
    h_path = Path("data/homography_matrix.json")
    assert h_path.exists(), "data/homography_matrix.json must exist"

    homography = PitchHomography(h_path)
    assert homography.H.shape == (3, 3)
    assert homography.pitch_length == 105.0
    assert homography.pitch_width == 68.0


def test_homography_valid_point_projection():
    homography = PitchHomography("data/homography_matrix.json")

    # Corner 3 of the 18-yard box: pixel (820, 655) calibrated to metric (16.5m, 54.16m)
    px, py, is_valid = homography.project_point(820.0, 655.0)
    assert is_valid is True
    assert px == pytest.approx(16.5, abs=0.5)
    assert py == pytest.approx(54.16, abs=0.5)
    assert homography.is_within_calibrated_zone(px, py) is True


def test_homography_bounds_and_validity_logic():
    homography = PitchHomography("data/homography_matrix.json")

    # Inside pitch center
    assert homography.is_within_pitch(52.5, 34.0) is True

    # Negative coordinates (outside pitch)
    assert homography.is_within_pitch(-5.0, 34.0) is False
    assert homography.is_within_pitch(50.0, -2.0) is False

    # Beyond standard pitch length (105m) or width (68m)
    assert homography.is_within_pitch(110.0, 34.0) is False
    assert homography.is_within_pitch(50.0, 75.0) is False

    # NaN coordinates
    assert homography.is_within_pitch(float("nan"), 34.0) is False

    # Extreme pixel outside calibrated field (e.g. top-left sky / stadium pixel 0, 0)
    px_sky, py_sky, is_valid_sky = homography.project_point(0.0, 0.0)
    assert is_valid_sky is False


def test_batch_point_projection():
    homography = PitchHomography("data/homography_matrix.json")
    pts = np.array([[820.0, 655.0], [43.0, 408.0]], dtype=np.float64)
    res = homography.project_points(pts)
    assert res.shape == (2, 2)
    # Check Corner 3 (16.5, 54.16) and Corner 2 (16.5, 13.84)
    assert res[0, 0] == pytest.approx(16.5, abs=0.5)
    assert res[0, 1] == pytest.approx(54.16, abs=0.5)
    assert res[1, 0] == pytest.approx(16.5, abs=0.5)
    assert res[1, 1] == pytest.approx(13.84, abs=0.5)


def test_project_bbox_foot():
    homography = PitchHomography("data/homography_matrix.json")
    # Corner 3: pixel (820, 655). Say bbox is x1=800, y1=500, x2=840, y2=655
    # Then foot_x = 820.0, foot_y = 655.0
    fx, fy, px, py, is_valid, in_box = homography.project_bbox_foot(800.0, 500.0, 840.0, 655.0)
    assert fx == pytest.approx(820.0)
    assert fy == pytest.approx(655.0)
    assert is_valid is True
    assert in_box is True
    assert px == pytest.approx(16.5, abs=0.5)
    assert py == pytest.approx(54.16, abs=0.5)


def test_homography_metadata():
    homography = PitchHomography("data/homography_matrix.json")
    meta = homography.get_metadata()
    assert meta["calibration_scope"] == "penalty_box_region"
    assert meta["full_pitch_calibration"] is False
    assert meta["pitch_dimensions_m"] == [105.0, 68.0]
    assert "warning" in meta
