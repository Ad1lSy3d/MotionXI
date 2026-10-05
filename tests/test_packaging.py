import importlib
from pathlib import Path
import pytest


def test_courtvision_package_imports():
    """Verify courtvision package and all submodules import cleanly."""
    modules = [
        "courtvision",
        "courtvision.detection",
        "courtvision.detection.detector",
        "courtvision.tracking",
        "courtvision.tracking.bytetrack_tracker",
        "courtvision.tracking.track_types",
        "courtvision.tracking.evaluation",
        "courtvision.projection",
        "courtvision.projection.homography",
        "courtvision.utils",
        "courtvision.utils.video",
    ]
    for mod in modules:
        m = importlib.import_module(mod)
        assert m is not None, f"Failed to import {mod}"


def test_third_party_runtime_dependencies():
    """Verify all external dependencies declared in pyproject.toml import cleanly."""
    deps = [
        "torch",
        "ultralytics",
        "supervision",
        "cv2",
        "scipy",
        "numpy",
        "trackeval",
        "SoccerNet",
    ]
    for dep in deps:
        m = importlib.import_module(dep)
        assert m is not None, f"Failed to import {dep}"


def test_detector_missing_weights_informative_error(tmp_path: Path):
    """Verify PlayerDetector fails with a clear, documented error message when weights are absent."""
    from courtvision.detection.detector import PlayerDetector

    fake_weights = tmp_path / "missing_model.pt"
    with pytest.raises(FileNotFoundError) as exc_info:
        PlayerDetector(model_path=str(fake_weights))

    msg = str(exc_info.value)
    assert "Model checkpoint not found" in msg
    assert "weights/" in msg
    assert "README.md" in msg


def test_video_missing_file_informative_error(tmp_path: Path):
    """Verify get_video_metadata fails with a clear error when input video is absent."""
    from courtvision.utils.video import get_video_metadata

    fake_video = tmp_path / "nonexistent.mp4"
    with pytest.raises(FileNotFoundError) as exc_info:
        get_video_metadata(str(fake_video))

    msg = str(exc_info.value)
    assert "Video file not found" in msg
