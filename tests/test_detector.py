from pathlib import Path
import numpy as np
import pytest
import torch
from courtvision.detection.detector import Detection, PlayerDetector


def test_model_file_exists_and_loads():
    weights_path = Path("weights/best.pt")
    assert weights_path.exists(), "weights/best.pt must exist in repository"

    detector = PlayerDetector(model_path=str(weights_path))
    assert detector.model is not None
    assert detector.model.task == "detect"
    assert 0 in detector.model.names
    assert detector.model.names[0] == "player"


def test_foot_contact_and_bbox_calculation():
    # Example bounding box: x1=100, y1=200, x2=160, y2=350
    x1, y1, x2, y2 = 100.0, 200.0, 160.0, 350.0
    det = Detection.create(
        frame=5,
        timestamp=0.1667,
        class_id=0,
        class_name="player",
        confidence=0.92,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )

    # Foot contact MUST be bottom center: x = (x1+x2)/2, y = y2
    assert det.foot_x == pytest.approx(130.0)
    assert det.foot_y == pytest.approx(350.0)
    assert det.bbox_width == pytest.approx(60.0)
    assert det.bbox_height == pytest.approx(150.0)
    assert det.frame == 5
    assert det.class_name == "player"


def test_detection_required_fields_and_dict_serialization():
    det = Detection.create(
        frame=12,
        timestamp=0.4,
        class_id=0,
        class_name="player",
        confidence=0.88,
        x1=50.0,
        y1=80.0,
        x2=110.0,
        y2=220.0,
    )
    d = det.to_dict()

    required_fields = [
        "frame",
        "timestamp",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_width",
        "bbox_height",
        "foot_x",
        "foot_y",
    ]
    for field in required_fields:
        assert field in d, f"Missing required field: {field}"


def test_detector_inference_on_synthetic_frame():
    detector = PlayerDetector(model_path="weights/best.pt")
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    dets, infer_ms = detector.detect(dummy_frame, frame_idx=0, fps=30.0)
    assert isinstance(dets, list)
    assert isinstance(infer_ms, float)
    assert infer_ms >= 0.0


def test_frame_439_nested_duplicate_suppressed():
    """Verify that the Frame 439 P61/P78 duplicate detection is eliminated."""
    from courtvision.detection.detector import suppress_contained_duplicates

    # Ground truth detection coordinates from Frame 439
    p61_box = Detection.create(439, 14.63, 0, "player", 0.77, 347.9, 434.2, 455.3, 660.6)
    p78_duplicate = Detection.create(439, 14.63, 0, "player", 0.61, 372.4, 526.1, 448.7, 674.5)

    raw_dets = [p61_box, p78_duplicate]
    cleaned = suppress_contained_duplicates(raw_dets)

    assert len(cleaned) == 1, "The nested duplicate P78 detection must be suppressed"
    assert cleaned[0].confidence == pytest.approx(0.77), "The full-body detection (P61) must be preserved"
    assert cleaned[0].y1 == pytest.approx(434.2)


def test_legitimate_overlapping_players_preserved():
    """Verify that separate players with distinct centers and comparable areas are NOT suppressed."""
    from courtvision.detection.detector import suppress_contained_duplicates

    # Two players crossing paths: comparable sizes, different horizontal centers
    p1 = Detection.create(100, 3.33, 0, "player", 0.88, 500.0, 400.0, 560.0, 600.0)
    p2 = Detection.create(100, 3.33, 0, "player", 0.84, 535.0, 410.0, 595.0, 605.0)

    cleaned = suppress_contained_duplicates([p1, p2])
    assert len(cleaned) == 2, "Both legitimate overlapping players must survive"
