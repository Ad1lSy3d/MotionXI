from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    frame: int
    timestamp: float
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    bbox_width: float
    bbox_height: float
    foot_x: float
    foot_y: float

    @classmethod
    def create(
        cls,
        frame: int,
        timestamp: float,
        class_id: int,
        class_name: str,
        confidence: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> "Detection":
        """Factory method calculating bbox dimensions and foot contact point."""
        bbox_width = float(x2 - x1)
        bbox_height = float(y2 - y1)
        foot_x = float((x1 + x2) / 2.0)
        foot_y = float(y2)  # Bottom center is ground-contact point
        return cls(
            frame=int(frame),
            timestamp=float(timestamp),
            class_id=int(class_id),
            class_name=str(class_name),
            confidence=float(confidence),
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            foot_x=foot_x,
            foot_y=foot_y,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def suppress_contained_duplicates(
    detections: List[Detection],
    containment_threshold: float = 0.75,
    iou_threshold: float = 0.20,
    center_dx_threshold: float = 0.40,
    area_ratio_threshold: float = 0.75,
) -> List[Detection]:
    """
    Suppress nested sub-box duplicate detections that survive standard NMS.

    Multi-criteria containment check evaluates pairs of detections:
    - Relative containment: intersection area / smaller box area >= containment_threshold (0.75)
    - Mutual IoU overlap: intersection / union >= iou_threshold (0.20)
    - Horizontal center alignment: |cx_small - cx_large| / w_small <= center_dx_threshold (0.40)
    - Relative box area: small_area / large_area <= area_ratio_threshold (0.75)

    When all four conditions hold, the smaller nested box (e.g. lower-body/torso duplicate)
    is suppressed in favor of the full-body player detection.
    """
    if len(detections) <= 1:
        return detections

    n = len(detections)
    suppressed_indices = set()

    for i in range(n):
        if i in suppressed_indices:
            continue
        for j in range(i + 1, n):
            if j in suppressed_indices:
                continue

            d1 = detections[i]
            d2 = detections[j]

            x1_inter = max(d1.x1, d2.x1)
            y1_inter = max(d1.y1, d2.y1)
            x2_inter = min(d1.x2, d2.x2)
            y2_inter = min(d1.y2, d2.y2)
            w_inter = max(0.0, x2_inter - x1_inter)
            h_inter = max(0.0, y2_inter - y1_inter)
            inter = w_inter * h_inter
            if inter <= 0.0:
                continue

            a1 = d1.bbox_width * d1.bbox_height
            a2 = d2.bbox_width * d2.bbox_height
            union = a1 + a2 - inter
            iou = inter / union if union > 0.0 else 0.0

            if a1 <= a2:
                small, large = d1, d2
                small_idx = i
                small_area, large_area = a1, a2
            else:
                small, large = d2, d1
                small_idx = j
                small_area, large_area = a2, a1

            if small_area <= 0.0 or large_area <= 0.0:
                continue

            containment = inter / small_area
            center_dx = abs(((small.x1 + small.x2) / 2.0) - ((large.x1 + large.x2) / 2.0)) / max(1.0, small.bbox_width)
            area_ratio = small_area / large_area

            # Multi-criteria duplicate check: never delete on containment alone
            if (
                containment >= containment_threshold
                and iou >= iou_threshold
                and center_dx <= center_dx_threshold
                and area_ratio <= area_ratio_threshold
            ):
                suppressed_indices.add(small_idx)

    return [d for idx, d in enumerate(detections) if idx not in suppressed_indices]


class PlayerDetector:
    """YOLOv11m player detection wrapper with automatic device dispatch."""

    def __init__(
        self,
        model_path: str = "weights/best.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        suppress_duplicates: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.model_path}. "
                "Please acquire the trained YOLOv11m checkpoint ('best.pt') and place it in the weights/ directory. "
                "See README.md ('Model Checkpoint Instructions') for details."
            )

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.suppress_duplicates = suppress_duplicates

        # Load weights
        self.model = YOLO(str(self.model_path))

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
    ) -> Tuple[List[Detection], float]:
        """
        Run inference on a single BGR frame.

        Returns:
            Tuple of (list of Detections, inference_time_ms)
        """
        timestamp = frame_idx / fps if fps > 0 else 0.0

        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )

        result = results[0]
        inference_ms = float(result.speed.get("inference", 0.0))

        detections: List[Detection] = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            xyxy_arr = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy_arr[i]
                conf = float(conf_arr[i])
                c_id = int(cls_arr[i])
                c_name = self.model.names.get(c_id, f"class_{c_id}")

                det = Detection.create(
                    frame=frame_idx,
                    timestamp=timestamp,
                    class_id=c_id,
                    class_name=c_name,
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
                detections.append(det)

        if self.suppress_duplicates and detections:
            detections = suppress_contained_duplicates(detections)

        return detections, inference_ms
