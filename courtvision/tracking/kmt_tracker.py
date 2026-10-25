from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from courtvision.detection.detector import Detection
from courtvision.projection.homography import PitchHomography
from courtvision.tracking.camera_motion import CameraMotionCompensator
from courtvision.tracking.track_types import TrackState, TrackedPlayer


class KinematicTrackState(str, Enum):
    TRACKED = "active"
    IN_CLUSTER = "in_cluster"
    GHOST = "ghost"


@dataclass
class KinematicTrack:
    """Internal image-space state for one persistent player identity."""

    track_id: int
    bbox: Tuple[float, float, float, float]
    foot_x: float
    foot_y: float
    confidence: float
    class_id: int
    class_name: str
    last_seen_frame: int
    motion_anchor_frame: int
    velocity: Tuple[float, float] = (0.0, 0.0)
    acceleration: Tuple[float, float] = (0.0, 0.0)
    state: KinematicTrackState = KinematicTrackState.TRACKED
    momentum_history: Deque[Tuple[Tuple[float, float], Tuple[float, float]]] = field(
        default_factory=lambda: deque(maxlen=10)
    )
    cached_velocity: Tuple[float, float] = (0.0, 0.0)
    cached_acceleration: Tuple[float, float] = (0.0, 0.0)
    in_cluster: bool = False


DetectionLike = Union[Detection, Dict[str, Any]]


class KinematicMemoryTracker:
    """Track detections with cached motion, ghost predictions, and speed gating.

    During clusters the kinematic box anchor is frozen, while current detections
    are still returned for rendering and downstream frame-level outputs.
    """

    def __init__(
        self,
        homography: Optional[PitchHomography] = None,
        trail_length: int = 15,
        momentum_window: int = 10,
        cluster_iou_threshold: float = 0.35,
        min_ghost_frames: int = 6,
        max_ghost_frames: int = 25,
        max_implied_speed_px_per_frame: float = 120.0,
        enable_camera_motion: bool = False,
    ) -> None:
        if trail_length < 1:
            raise ValueError("trail_length must be at least 1")
        if momentum_window < 1:
            raise ValueError("momentum_window must be at least 1")
        if not 0.0 <= cluster_iou_threshold <= 1.0:
            raise ValueError("cluster_iou_threshold must be between 0 and 1")
        if min_ghost_frames < 1 or max_ghost_frames < min_ghost_frames:
            raise ValueError("ghost frame limits must satisfy 1 <= min <= max")
        if max_implied_speed_px_per_frame <= 0:
            raise ValueError("max_implied_speed_px_per_frame must be positive")

        self.homography = homography
        self.trail_length = trail_length
        self.momentum_window = momentum_window
        self.cluster_iou_threshold = cluster_iou_threshold
        self.min_ghost_frames = min_ghost_frames
        self.max_ghost_frames = max_ghost_frames
        self.max_implied_speed_px_per_frame = max_implied_speed_px_per_frame
        self.enable_camera_motion = enable_camera_motion
        self.cmc = CameraMotionCompensator() if enable_camera_motion else None

        self.tracks: Dict[int, KinematicTrack] = {}
        self.next_track_id = 1
        self.trail_history: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.trail_length)
        )
        self.track_stats: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "first_frame": None,
                "last_frame": None,
                "hits": 0,
                "confidences": [],
            }
        )
        self.active_tracks_per_frame: List[int] = []

    @staticmethod
    def _read_detection(detection: DetectionLike) -> Dict[str, Any]:
        if isinstance(detection, dict):
            values = detection
            box = values.get("xyxy", values.get("bbox"))
            if box is not None:
                if len(box) != 4:
                    raise ValueError("Detection bbox/xyxy must contain four coordinates")
                x1, y1, x2, y2 = box
            else:
                try:
                    x1, y1, x2, y2 = (
                        values["x1"],
                        values["y1"],
                        values["x2"],
                        values["y2"],
                    )
                except KeyError as exc:
                    raise ValueError("Detection dictionaries require x1, y1, x2, y2") from exc
            confidence = values.get("confidence", values.get("conf", 1.0))
            class_id = values.get("class_id", 0)
            class_name = values.get("class_name", "player")
        else:
            x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
            confidence = detection.confidence
            class_id = detection.class_id
            class_name = detection.class_name

        box_values = tuple(float(value) for value in (x1, y1, x2, y2))
        if not all(math.isfinite(value) for value in box_values):
            raise ValueError("Detection bbox coordinates must be finite")
        if box_values[2] <= box_values[0] or box_values[3] <= box_values[1]:
            raise ValueError("Detection bbox must have positive width and height")
        return {
            "bbox": box_values,
            "confidence": float(confidence),
            "class_id": int(class_id),
            "class_name": str(class_name),
        }

    @staticmethod
    def _foot(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2.0, bbox[3]

    @staticmethod
    def _iou(
        first: Tuple[float, float, float, float],
        second: Tuple[float, float, float, float],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if intersection == 0.0:
            return 0.0
        area_first = (first[2] - first[0]) * (first[3] - first[1])
        area_second = (second[2] - second[0]) * (second[3] - second[1])
        union = area_first + area_second - intersection
        return intersection / union if union > 0.0 else 0.0

    def _predict_bbox(self, track: KinematicTrack, frame_idx: int) -> Tuple[float, ...]:
        delta = frame_idx - track.motion_anchor_frame
        vx, vy = track.cached_velocity
        ax, ay = track.cached_acceleration
        shift_x = vx * delta + 0.5 * ax * delta * delta
        shift_y = vy * delta + 0.5 * ay * delta * delta
        x1, y1, x2, y2 = track.bbox
        return x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y

    def _predicted_foot(self, track: KinematicTrack, frame_idx: int) -> Tuple[float, float]:
        return self._foot(self._predict_bbox(track, frame_idx))

    def _implied_speed(
        self,
        track: KinematicTrack,
        detection: Dict[str, Any],
        frame_idx: int,
    ) -> float:
        delta = frame_idx - track.last_seen_frame
        if delta <= 0:
            return math.inf
        predicted_x, predicted_y = self._predicted_foot(track, frame_idx)
        detection_x, detection_y = self._foot(detection["bbox"])
        return math.hypot(detection_x - predicted_x, detection_y - predicted_y) / delta

    def _associate(
        self,
        candidates: Sequence[KinematicTrack],
        detections: Sequence[Dict[str, Any]],
        frame_idx: int,
    ) -> Tuple[List[Tuple[int, int]], set[int]]:
        if not candidates or not detections:
            return [], set()

        impossible_cost = 1000.0
        costs = np.empty((len(candidates), len(detections)), dtype=np.float64)
        for track_index, track in enumerate(candidates):
            predicted = self._predict_bbox(track, frame_idx)
            for detection_index, detection in enumerate(detections):
                implied_speed = self._implied_speed(track, detection, frame_idx)
                if implied_speed > self.max_implied_speed_px_per_frame:
                    costs[track_index, detection_index] = impossible_cost
                else:
                    costs[track_index, detection_index] = 1.0 - self._iou(
                        predicted, detection["bbox"]
                    )

        row_indices, column_indices = linear_sum_assignment(costs)
        matches: List[Tuple[int, int]] = []
        matched_detections: set[int] = set()
        for row, column in zip(row_indices, column_indices):
            if costs[row, column] >= impossible_cost:
                continue
            matches.append((int(row), int(column)))
            matched_detections.add(int(column))
        return matches, matched_detections

    def _update_kinematics(
        self,
        track: KinematicTrack,
        foot_x: float,
        foot_y: float,
        frame_idx: int,
    ) -> None:
        delta = frame_idx - track.motion_anchor_frame
        if delta > 0:
            new_velocity = (
                (foot_x - track.foot_x) / delta,
                (foot_y - track.foot_y) / delta,
            )
            if track.momentum_history:
                new_acceleration = (
                    (new_velocity[0] - track.velocity[0]) / delta,
                    (new_velocity[1] - track.velocity[1]) / delta,
                )
            else:
                new_acceleration = (0.0, 0.0)
            track.velocity = new_velocity
            track.acceleration = new_acceleration
            track.momentum_history.append((new_velocity, new_acceleration))

    def _cache_momentum(self, track: KinematicTrack) -> None:
        if track.momentum_history:
            track.cached_velocity = tuple(
                float(np.mean([sample[0][axis] for sample in track.momentum_history]))
                for axis in range(2)
            )
            track.cached_acceleration = tuple(
                float(np.mean([sample[1][axis] for sample in track.momentum_history]))
                for axis in range(2)
            )
        else:
            track.cached_velocity = track.velocity
            track.cached_acceleration = track.acceleration

    def _clustered_detection_indices(
        self, detections: Sequence[Dict[str, Any]]
    ) -> set[int]:
        clustered: set[int] = set()
        for first_index, first in enumerate(detections):
            for second_index in range(first_index + 1, len(detections)):
                if (
                    self._iou(first["bbox"], detections[second_index]["bbox"])
                    >= self.cluster_iou_threshold
                ):
                    clustered.add(first_index)
                    clustered.add(second_index)
        return clustered

    def _create_track(
        self,
        detection: Dict[str, Any],
        frame_idx: int,
    ) -> KinematicTrack:
        foot_x, foot_y = self._foot(detection["bbox"])
        track = KinematicTrack(
            track_id=self.next_track_id,
            bbox=detection["bbox"],
            foot_x=foot_x,
            foot_y=foot_y,
            confidence=detection["confidence"],
            class_id=detection["class_id"],
            class_name=detection["class_name"],
            last_seen_frame=frame_idx,
            motion_anchor_frame=frame_idx,
            momentum_history=deque(maxlen=self.momentum_window),
        )
        self.next_track_id += 1
        self.tracks[track.track_id] = track
        return track

    def _to_tracked_player(
        self,
        track: KinematicTrack,
        bbox: Tuple[float, float, float, float],
        frame_idx: int,
        fps: float,
        state: str,
    ) -> TrackedPlayer:
        x1, y1, x2, y2 = bbox
        foot_x, foot_y = self._foot(bbox)
        pitch_x: Optional[float] = None
        pitch_y: Optional[float] = None
        projection_valid = False
        in_calibrated_zone = False
        if self.homography is not None:
            projected_x, projected_y, projection_valid = self.homography.project_point(
                foot_x, foot_y
            )
            in_calibrated_zone = self.homography.is_within_calibrated_zone(
                projected_x, projected_y
            )
            pitch_x = round(projected_x, 3) if math.isfinite(projected_x) else None
            pitch_y = round(projected_y, 3) if math.isfinite(projected_y) else None

        return TrackedPlayer.create(
            frame=frame_idx,
            timestamp=frame_idx / fps if fps > 0 else 0.0,
            track_id=track.track_id,
            class_id=track.class_id,
            class_name=track.class_name,
            confidence=track.confidence,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            projection_valid=projection_valid,
            in_calibrated_zone=in_calibrated_zone,
            state=state,
        )

    def _record_track(self, track: KinematicTrack, frame_idx: int) -> None:
        stat = self.track_stats[track.track_id]
        if stat["first_frame"] is None:
            stat["first_frame"] = frame_idx
        stat["last_frame"] = frame_idx
        stat["hits"] += 1
        stat["confidences"].append(track.confidence)

    def update(
        self,
        detections: List[DetectionLike],
        frame_idx: int,
        fps: float = 30.0,
        frame: Optional[np.ndarray] = None,
    ) -> Tuple[List[TrackedPlayer], float]:
        """Update tracks and return track records plus association time in milliseconds."""
        if frame_idx < 0:
            raise ValueError("frame_idx must be non-negative")
        normalized = [self._read_detection(detection) for detection in detections]
        start_time = time.perf_counter()

        # Frame-to-frame camera motion compensation
        if self.enable_camera_motion and self.cmc is not None and frame is not None:
            M = self.cmc.estimate_motion(frame)
            if M is not None:
                for track in self.tracks.values():
                    track.bbox = CameraMotionCompensator.transform_bbox(track.bbox, M)
                    track.foot_x, track.foot_y = CameraMotionCompensator.transform_point(
                        track.foot_x, track.foot_y, M
                    )
                    track.velocity = CameraMotionCompensator.transform_velocity(
                        track.velocity[0], track.velocity[1], M
                    )
                    track.cached_velocity = CameraMotionCompensator.transform_velocity(
                        track.cached_velocity[0], track.cached_velocity[1], M
                    )

        expired_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if frame_idx - track.last_seen_frame > self.max_ghost_frames
        ]
        for track_id in expired_ids:
            del self.tracks[track_id]

        candidates = sorted(
            (
                track
                for track in self.tracks.values()
                if 0 < frame_idx - track.last_seen_frame <= self.max_ghost_frames
            ),
            key=lambda track: track.track_id,
        )
        matches, matched_detection_indices = self._associate(candidates, normalized, frame_idx)
        clustered_detection_indices = self._clustered_detection_indices(normalized)

        current_tracks: List[Tuple[KinematicTrack, Tuple[float, ...], str]] = []
        for track_index, detection_index in matches:
            track = candidates[track_index]
            detection = normalized[detection_index]
            foot_x, foot_y = self._foot(detection["bbox"])
            is_clustered = detection_index in clustered_detection_indices
            if is_clustered:
                if not track.in_cluster:
                    self._cache_momentum(track)
            else:
                self._update_kinematics(track, foot_x, foot_y, frame_idx)
                self._cache_momentum(track)
                track.bbox = detection["bbox"]
                track.foot_x = foot_x
                track.foot_y = foot_y
                track.motion_anchor_frame = frame_idx
            track.confidence = detection["confidence"]
            track.class_id = detection["class_id"]
            track.class_name = detection["class_name"]
            track.last_seen_frame = frame_idx
            track.in_cluster = is_clustered
            track.state = (
                KinematicTrackState.IN_CLUSTER
                if is_clustered
                else KinematicTrackState.TRACKED
            )
            current_state = (
                KinematicTrackState.IN_CLUSTER.value
                if is_clustered
                else TrackState.ACTIVE.value
            )
            current_tracks.append((track, detection["bbox"], current_state))
        for detection_index, detection in enumerate(normalized):
            if detection_index not in matched_detection_indices:
                track = self._create_track(detection, frame_idx)
                if detection_index in clustered_detection_indices:
                    track.in_cluster = True
                    track.state = KinematicTrackState.IN_CLUSTER
                    current_state = KinematicTrackState.IN_CLUSTER.value
                else:
                    current_state = TrackState.ACTIVE.value
                current_tracks.append((track, detection["bbox"], current_state))

        output_tracks: List[TrackedPlayer] = []
        for track, bbox, state in current_tracks:
            self._record_track(track, frame_idx)
            self.trail_history[track.track_id].append(self._foot(bbox))
            output_tracks.append(
                self._to_tracked_player(track, bbox, frame_idx, fps, state)
            )

        for track in sorted(self.tracks.values(), key=lambda value: value.track_id):
            if track.last_seen_frame == frame_idx:
                continue
            gap = frame_idx - track.last_seen_frame
            if self.min_ghost_frames <= gap <= self.max_ghost_frames:
                ghost_bbox = self._predict_bbox(track, frame_idx)
                track.state = KinematicTrackState.GHOST
                self.trail_history[track.track_id].append(self._foot(ghost_bbox))
                output_tracks.append(
                    self._to_tracked_player(
                        track,
                        ghost_bbox,
                        frame_idx,
                        fps,
                        KinematicTrackState.GHOST.value,
                    )
                )

        self.active_tracks_per_frame.append(len(output_tracks))
        tracker_ms = (time.perf_counter() - start_time) * 1000.0
        return output_tracks, tracker_ms

    def get_track_summary_records(self) -> List[Dict[str, Any]]:
        records = []
        for track_id, stat in sorted(self.track_stats.items()):
            confidences = stat["confidences"]
            records.append(
                {
                    "track_id": track_id,
                    "first_frame": stat["first_frame"],
                    "last_frame": stat["last_frame"],
                    "track_length": int(stat["hits"]),
                    "mean_confidence": round(
                        float(np.mean(confidences)) if confidences else 0.0, 4
                    ),
                }
            )
        return records

    def get_track_statistics(self) -> Dict[str, Any]:
        summaries = self.get_track_summary_records()
        lengths = [record["track_length"] for record in summaries]
        active_counts = self.active_tracks_per_frame
        if lengths:
            avg_len = float(np.mean(lengths))
            median_len = float(np.median(lengths))
            min_len = int(np.min(lengths))
            max_len = int(np.max(lengths))
        else:
            avg_len = median_len = 0.0
            min_len = max_len = 0
        if active_counts:
            avg_active = float(np.mean(active_counts))
            median_active = float(np.median(active_counts))
            max_active = int(np.max(active_counts))
        else:
            avg_active = median_active = 0.0
            max_active = 0
        stats: Dict[str, Any] = {
            "total_unique_track_ids": len(summaries),
            "average_track_length_frames": round(avg_len, 2),
            "median_track_length_frames": round(median_len, 2),
            "min_track_length_frames": min_len,
            "max_track_length_frames": max_len,
            "average_active_tracks_per_frame": round(avg_active, 2),
            "median_active_tracks_per_frame": round(median_active, 2),
            "maximum_active_tracks_per_frame": max_active,
        }
        if self.cmc is not None:
            stats["camera_motion_compensation"] = self.cmc.get_statistics()
        return stats

    def get_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "tracker_name": "KinematicMemoryTracker",
            "momentum_window": self.momentum_window,
            "cluster_iou_threshold": self.cluster_iou_threshold,
            "min_ghost_frames": self.min_ghost_frames,
            "max_ghost_frames": self.max_ghost_frames,
            "max_implied_speed_px_per_frame": self.max_implied_speed_px_per_frame,
            "trail_length": self.trail_length,
            "enable_camera_motion": self.enable_camera_motion,
        }
        if self.cmc is not None:
            cfg["camera_motion_stats"] = self.cmc.get_statistics()
        return cfg
