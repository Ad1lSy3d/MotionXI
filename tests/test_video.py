from pathlib import Path
import numpy as np
import pytest
from courtvision.utils.video import (
    VideoMetadata,
    calculate_pairwise_ious,
    get_video_metadata,
)


def test_video_metadata_reading():
    video_path = Path("data/test_match.mp4")
    assert video_path.exists(), "data/test_match.mp4 must exist"

    meta: VideoMetadata = get_video_metadata(video_path)
    assert meta.width == 1920
    assert meta.height == 1080
    assert meta.fps == pytest.approx(30.0, abs=0.1)
    assert meta.container_frame_count == 618
    assert meta.frame_count_source == "container_header"


def test_decoded_frame_count_and_warning():
    video_path = Path("data/test_match.mp4")
    meta = get_video_metadata(video_path)

    # Simulate decoding completion at 600 frames
    decoded_meta = meta.with_decoded_count(600)
    assert decoded_meta.decoded_frame_count == 600
    assert decoded_meta.frame_count == 600
    assert decoded_meta.frame_count_source == "decoded_stream"
    assert decoded_meta.frame_count_warning is not None
    assert "reports 618 frames but only 600 frames decoded" in decoded_meta.frame_count_warning
    assert decoded_meta.duration_sec == pytest.approx(20.0, abs=0.1)


def test_pairwise_iou_calculation():
    # Case 1: Two clearly overlapping boxes (50% intersection)
    # Box A: [0, 0, 10, 10] (area 100)
    # Box B: [5, 0, 15, 10] (area 100, intersection [5,0,10,10] = 50, union = 150 -> IoU = 1/3 ~ 0.3333)
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [5.0, 0.0, 15.0, 10.0],
        [100.0, 100.0, 110.0, 110.0],  # Isolated box
    ], dtype=np.float32)

    max_iou, overlap_count, pairs = calculate_pairwise_ious(boxes, min_iou_threshold=0.1)
    assert max_iou == pytest.approx(1.0 / 3.0, abs=0.01)
    assert overlap_count == 1
    assert len(pairs) == 1
    assert pairs[0][0] == 0
    assert pairs[0][1] == 1


def test_pairwise_iou_zero_boxes():
    empty_boxes = np.empty((0, 4), dtype=np.float32)
    max_iou, count, pairs = calculate_pairwise_ious(empty_boxes)
    assert max_iou == 0.0
    assert count == 0
    assert pairs == []


def test_nonexistent_video_raises_error():
    with pytest.raises(FileNotFoundError):
        get_video_metadata("data/nonexistent_file.mp4")
