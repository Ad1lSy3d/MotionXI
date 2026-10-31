from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class CameraMotionCompensator:
    """
    Frame-to-frame rigid affine camera motion compensator for sports broadcast video.
    Estimates the background geometric transformation M_{t-1 -> t} to align existing
    tracker state with the current video frame coordinates.
    """

    def __init__(
        self,
        max_corners: int = 250,
        quality_level: float = 0.01,
        min_distance: float = 25.0,
        min_inliers: int = 10,
        max_translation_px: float = 150.0,
        min_scale: float = 0.85,
        max_scale: float = 1.15,
        max_rotation_deg: float = 15.0,
    ) -> None:
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.min_inliers = min_inliers
        self.max_translation_px = max_translation_px
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_rotation_deg = max_rotation_deg

        self.prev_gray: Optional[np.ndarray] = None
        self.total_frames: int = 0
        self.compensated_frames: int = 0
        self.failed_frames: int = 0
        self.last_transform: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset compensator history and diagnostics."""
        self.prev_gray = None
        self.total_frames = 0
        self.compensated_frames = 0
        self.failed_frames = 0
        self.last_transform = None

    def estimate_motion(
        self,
        frame: Union[np.ndarray, None],
    ) -> Optional[np.ndarray]:
        """
        Estimate 2x3 affine matrix M mapping coordinates from frame t-1 to frame t.
        Returns None if estimation fails or is deemed unreliable (outlier fallback).
        """
        if frame is None:
            self.last_transform = None
            return None

        self.total_frames += 1

        if frame.ndim == 3:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = frame

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            self.last_transform = None
            return None

        p0 = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )

        if p0 is None or len(p0) < self.min_inliers:
            self.failed_frames += 1
            self.prev_gray = curr_gray
            self.last_transform = None
            return None

        p1, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, p0, None)
        if p1 is None or status is None:
            self.failed_frames += 1
            self.prev_gray = curr_gray
            self.last_transform = None
            return None

        valid_idx = (status.flatten() == 1)
        good_p0 = p0[valid_idx]
        good_p1 = p1[valid_idx]

        if len(good_p0) < self.min_inliers:
            self.failed_frames += 1
            self.prev_gray = curr_gray
            self.last_transform = None
            return None

        # Robust rigid partial affine estimation (rotation + scale + translation)
        M, inliers = cv2.estimateAffinePartial2D(
            good_p0,
            good_p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1000,
        )

        self.prev_gray = curr_gray

        if M is None or inliers is None:
            self.failed_frames += 1
            self.last_transform = None
            return None

        num_inliers = int(np.sum(inliers))
        if num_inliers < self.min_inliers:
            self.failed_frames += 1
            self.last_transform = None
            return None

        # Validate transformation realism
        tx = float(M[0, 2])
        ty = float(M[1, 2])
        trans_dist = math.hypot(tx, ty)
        if trans_dist > self.max_translation_px:
            self.failed_frames += 1
            self.last_transform = None
            return None

        scale = math.hypot(M[0, 0], M[0, 1])
        if scale < self.min_scale or scale > self.max_scale:
            self.failed_frames += 1
            self.last_transform = None
            return None

        rotation_deg = abs(math.degrees(math.atan2(M[1, 0], M[0, 0])))
        if rotation_deg > self.max_rotation_deg:
            self.failed_frames += 1
            self.last_transform = None
            return None

        self.compensated_frames += 1
        self.last_transform = M.astype(np.float64)
        return self.last_transform

    @staticmethod
    def transform_point(
        x: float,
        y: float,
        M: np.ndarray,
    ) -> Tuple[float, float]:
        """Apply 2x3 affine matrix M to a 2D point (x, y)."""
        new_x = float(M[0, 0] * x + M[0, 1] * y + M[0, 2])
        new_y = float(M[1, 0] * x + M[1, 1] * y + M[1, 2])
        return new_x, new_y

    @staticmethod
    def transform_bbox(
        bbox: Tuple[float, float, float, float],
        M: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Apply 2x3 affine matrix M to bounding box (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        new_center_x, new_center_y = CameraMotionCompensator.transform_point(center_x, center_y, M)
        scale = math.hypot(M[0, 0], M[0, 1])
        new_w = w * scale
        new_h = h * scale

        new_x1 = new_center_x - new_w / 2.0
        new_y1 = new_center_y - new_h / 2.0
        new_x2 = new_center_x + new_w / 2.0
        new_y2 = new_center_y + new_h / 2.0
        return new_x1, new_y1, new_x2, new_y2

    @staticmethod
    def transform_velocity(
        vx: float,
        vy: float,
        M: np.ndarray,
    ) -> Tuple[float, float]:
        """Rotate and scale velocity vector by the linear portion of M."""
        new_vx = float(M[0, 0] * vx + M[0, 1] * vy)
        new_vy = float(M[1, 0] * vx + M[1, 1] * vy)
        return new_vx, new_vy

    def get_statistics(self) -> Dict[str, Any]:
        """Return diagnostic metrics of camera motion estimation."""
        fail_rate = (
            (self.failed_frames / self.total_frames * 100.0)
            if self.total_frames > 0
            else 0.0
        )
        return {
            "total_frames": self.total_frames,
            "compensated_frames": self.compensated_frames,
            "failed_frames": self.failed_frames,
            "failure_rate_percent": round(fail_rate, 2),
        }
