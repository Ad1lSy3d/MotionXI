from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from courtvision.detection.detector import Detection


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    container_frame_count: int
    decoded_frame_count: Optional[int] = None
    frame_count_source: str = "container_header"
    frame_count_warning: Optional[str] = None
    duration_sec: float = 0.0

    @property
    def frame_count(self) -> int:
        """Return authoritative frame count: decoded count if available, else container count."""
        if self.decoded_frame_count is not None:
            return self.decoded_frame_count
        return self.container_frame_count

    def with_decoded_count(self, decoded_count: int) -> "VideoMetadata":
        """Produce an updated VideoMetadata reflecting actual decoded frames from stream."""
        warning = None
        if decoded_count != self.container_frame_count:
            warning = (
                f"Container metadata reports {self.container_frame_count} frames but only "
                f"{decoded_count} frames decoded successfully."
            )
        duration = float(decoded_count / self.fps) if self.fps > 0 else 0.0
        return VideoMetadata(
            width=self.width,
            height=self.height,
            fps=self.fps,
            container_frame_count=self.container_frame_count,
            decoded_frame_count=decoded_count,
            frame_count_source="decoded_stream",
            frame_count_warning=warning,
            duration_sec=duration,
        )

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "container_frame_count": self.container_frame_count,
            "decoded_frame_count": self.decoded_frame_count,
            "frame_count_source": self.frame_count_source,
            "frame_count_warning": self.frame_count_warning,
            "duration_sec": round(self.duration_sec, 2),
        }


def get_video_metadata(
    video_path: Union[str, Path],
    count_decoded_frames: bool = False,
) -> VideoMetadata:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found at: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video at: {path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        container_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = float(container_frame_count / fps) if fps > 0 else 0.0
    finally:
        cap.release()

    meta = VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        container_frame_count=container_frame_count,
        decoded_frame_count=None,
        frame_count_source="container_header",
        frame_count_warning=None,
        duration_sec=duration,
    )

    if count_decoded_frames:
        cap = cv2.VideoCapture(str(path))
        decoded_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            decoded_count += 1
        cap.release()
        meta = meta.with_decoded_count(decoded_count)

    return meta


def create_video_writer(
    output_path: Union[str, Path],
    fps: float,
    width: int,
    height: int,
    codec: str = "mp4v",
) -> cv2.VideoWriter:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        # Fallback to avc1 if mp4v fails
        fourcc_alt = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(path), fourcc_alt, fps, (width, height))
    return writer


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    frame_idx: int,
    total_frames: int,
    fps: float,
    inference_ms: float = 0.0,
    device_name: str = "mps",
) -> np.ndarray:
    """
    Render high-contrast, broadcast-grade detection visualizations and HUD on a video frame.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Color palette
    box_color = (0, 220, 115)       # Vibrant pitch green (BGR)
    foot_color = (0, 140, 255)      # Orange accent for foot contact
    text_color = (255, 255, 255)    # White
    hud_bg_color = (18, 18, 22)     # Deep dark HUD banner

    # 1. Draw top HUD Banner
    hud_height = 44
    hud_overlay = canvas[:hud_height, :].copy()
    cv2.rectangle(canvas, (0, 0), (w, hud_height), hud_bg_color, -1)
    cv2.addWeighted(canvas[:hud_height, :], 0.75, hud_overlay, 0.25, 0, canvas[:hud_height, :])

    # HUD Telemetry
    timestamp_sec = frame_idx / fps if fps > 0 else 0.0
    minutes = int(timestamp_sec // 60)
    seconds = timestamp_sec % 60
    time_str = f"{minutes:02d}:{seconds:05.2f}"

    hud_text = (
        f"COURTVISION AI | Frame: {frame_idx:03d}/{total_frames:03d} | "
        f"Time: {time_str} | "
        f"Detected: {len(detections)} players | "
        f"Infer: {inference_ms:.1f}ms | "
        f"Device: {device_name.upper()}"
    )
    cv2.putText(
        canvas,
        hud_text,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        text_color,
        2,
        cv2.LINE_AA,
    )

    # 2. Draw Bounding Boxes and Foot-Contact Points
    for det in detections:
        x1, y1 = int(round(det.x1)), int(round(det.y1))
        x2, y2 = int(round(det.x2)), int(round(det.y2))
        fx, fy = int(round(det.foot_x)), int(round(det.foot_y))

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # Label tag
        label = f"{det.class_name} {det.confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        tag_y1 = max(hud_height, y1 - label_h - 6)
        tag_y2 = tag_y1 + label_h + 6
        tag_x2 = x1 + label_w + 8

        cv2.rectangle(canvas, (x1, tag_y1), (tag_x2, tag_y2), box_color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 4, tag_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Foot-contact point indicator (circle with crosshair)
        cv2.circle(canvas, (fx, fy), 4, foot_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (fx, fy), 7, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a distinct, deterministic BGR color for a given track ID."""
    hue = int((track_id * 137.508) % 180)
    hsv_pixel = np.uint8([[[hue, 220, 245]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


def calculate_pairwise_ious(
    boxes: np.ndarray,
    min_iou_threshold: float = 0.1,
) -> Tuple[float, int, List[Tuple[int, int, float]]]:
    """
    Calculate maximum pairwise IoU and count of overlapping pairs among N bounding boxes.

    Args:
        boxes: np.ndarray of shape (N, 4) in [x1, y1, x2, y2] format.
        min_iou_threshold: threshold to consider two boxes overlapping.

    Returns:
        (max_pairwise_iou, overlapping_pairs_count, list_of_overlapping_pairs)
    """
    n = len(boxes)
    if n < 2:
        return 0.0, 0, []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    max_iou = 0.0
    overlapping_pairs: List[Tuple[int, int, float]] = []

    for i in range(n):
        xx1 = np.maximum(x1[i], x1[i + 1:])
        yy1 = np.maximum(y1[i], y1[i + 1:])
        xx2 = np.minimum(x2[i], x2[i + 1:])
        yy2 = np.minimum(y2[i], y2[i + 1:])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        union = areas[i] + areas[i + 1:] - intersection
        ious = np.where(union > 0, intersection / union, 0.0)

        for j_offset, iou in enumerate(ious):
            j = i + 1 + j_offset
            iou_val = float(iou)
            if iou_val > max_iou:
                max_iou = iou_val
            if iou_val >= min_iou_threshold:
                overlapping_pairs.append((i, j, round(iou_val, 4)))

    return round(float(max_iou), 4), len(overlapping_pairs), overlapping_pairs


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Any],
    trail_history: Dict[int, Any],
    frame_idx: int,
    total_frames: int,
    fps: float,
    infer_ms: float = 0.0,
    track_ms: float = 0.0,
    total_unique_ids: int = 0,
    device_name: str = "mps",
) -> np.ndarray:
    """
    Render high-contrast ByteTrack visualizations including ID tags, foot trails, and HUD telemetry.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    hud_bg_color = (18, 18, 22)
    text_color = (255, 255, 255)

    # 1. Top HUD Banner
    hud_height = 44
    hud_overlay = canvas[:hud_height, :].copy()
    cv2.rectangle(canvas, (0, 0), (w, hud_height), hud_bg_color, -1)
    cv2.addWeighted(canvas[:hud_height, :], 0.75, hud_overlay, 0.25, 0, canvas[:hud_height, :])

    timestamp_sec = frame_idx / fps if fps > 0 else 0.0
    minutes = int(timestamp_sec // 60)
    seconds = timestamp_sec % 60
    time_str = f"{minutes:02d}:{seconds:05.2f}"

    hud_text = (
        f"COURTVISION AI | BYTETRACK | Frame: {frame_idx:03d}/{total_frames:03d} | "
        f"Time: {time_str} | "
        f"Active: {len(tracks)} | Total IDs: {total_unique_ids} | "
        f"Infer: {infer_ms:.1f}ms | Track: {track_ms:.1f}ms | "
        f"{device_name.upper()}"
    )
    cv2.putText(
        canvas,
        hud_text,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        text_color,
        2,
        cv2.LINE_AA,
    )

    # 2. Draw Trails for Active Tracks
    for track in tracks:
        t_id = track.track_id
        color = get_track_color(t_id)
        trail = trail_history.get(t_id, [])
        if len(trail) > 1:
            points = list(trail)
            for k in range(1, len(points)):
                pt1 = (int(round(points[k - 1][0])), int(round(points[k - 1][1])))
                pt2 = (int(round(points[k][0])), int(round(points[k][1])))
                thickness = max(1, int(1 + (k / len(points)) * 2))
                cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    # 3. Draw Bounding Boxes and ID Badges
    for track in tracks:
        t_id = track.track_id
        color = get_track_color(t_id)

        x1, y1 = int(round(track.x1)), int(round(track.y1))
        x2, y2 = int(round(track.x2)), int(round(track.y2))
        fx, fy = int(round(track.foot_x)), int(round(track.foot_y))

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # ID Badge
        label = f"ID:{t_id} ({track.confidence:.2f})"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        tag_y1 = max(hud_height, y1 - label_h - 6)
        tag_y2 = tag_y1 + label_h + 6
        tag_x2 = x1 + label_w + 8

        cv2.rectangle(canvas, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 4, tag_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Foot contact anchor
        cv2.circle(canvas, (fx, fy), 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (fx, fy), 7, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas
