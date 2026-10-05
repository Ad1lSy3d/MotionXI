from collections import defaultdict, deque
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import numpy as np
import supervision as sv

from courtvision.detection.detector import Detection
from courtvision.projection.homography import PitchHomography
from courtvision.tracking.track_types import TrackState, TrackedPlayer


class FootballByteTracker:
    """
    ByteTrack baseline tracker wrapper for football player tracking.

    Bipartite association tracks low-confidence detections by dividing detections into
    high-confidence (first association) and low-confidence (second association) tiers,
    significantly reducing track fragmentation during partial player occlusions.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        homography: Optional[PitchHomography] = None,
        trail_length: int = 15,
    ) -> None:
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.frame_rate = frame_rate
        self.homography = homography
        self.trail_length = trail_length

        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            frame_rate=self.frame_rate,
        )

        # Track history and statistics storage
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

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
        fps: float,
    ) -> Tuple[List[TrackedPlayer], float]:
        """
        Process frame detections through ByteTrack.

        Returns:
            Tuple of (list of TrackedPlayer instances, tracker_execution_time_ms)
        """
        timestamp = frame_idx / fps if fps > 0 else 0.0

        if not detections:
            sv_dets = sv.Detections.empty()
        else:
            xyxy = np.array(
                [[d.x1, d.y1, d.x2, d.y2] for d in detections],
                dtype=np.float32,
            )
            conf = np.array([d.confidence for d in detections], dtype=np.float32)
            cls_ids = np.array([d.class_id for d in detections], dtype=int)
            sv_dets = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=cls_ids,
            )

        # Time the ByteTrack association step
        start_time = time.perf_counter()
        tracked_sv = self.tracker.update_with_detections(sv_dets)
        tracker_ms = (time.perf_counter() - start_time) * 1000.0

        tracked_players: List[TrackedPlayer] = []
        if tracked_sv.tracker_id is not None and len(tracked_sv.tracker_id) > 0:
            for i in range(len(tracked_sv)):
                track_id = int(tracked_sv.tracker_id[i])
                box = tracked_sv.xyxy[i]
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                conf_val = float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 0.0
                c_id = int(tracked_sv.class_id[i]) if tracked_sv.class_id is not None else 0
                c_name = "player"

                # Calculate foot ground anchor and project if homography available
                foot_x = (x1 + x2) / 2.0
                foot_y = y2

                pitch_x = None
                pitch_y = None
                proj_valid = False
                in_box = False

                if self.homography is not None:
                    px, py, proj_valid = self.homography.project_point(foot_x, foot_y)
                    in_box = self.homography.is_within_calibrated_zone(px, py)
                    pitch_x = round(px, 3) if not np.isnan(px) else None
                    pitch_y = round(py, 3) if not np.isnan(py) else None

                # Update trajectory trail
                self.trail_history[track_id].append((foot_x, foot_y))

                # Update track summary statistics
                stat = self.track_stats[track_id]
                if stat["first_frame"] is None:
                    stat["first_frame"] = frame_idx
                stat["last_frame"] = frame_idx
                stat["hits"] += 1
                stat["confidences"].append(conf_val)

                tracked_player = TrackedPlayer.create(
                    frame=frame_idx,
                    timestamp=timestamp,
                    track_id=track_id,
                    class_id=c_id,
                    class_name=c_name,
                    confidence=conf_val,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    pitch_x=pitch_x,
                    pitch_y=pitch_y,
                    projection_valid=proj_valid,
                    in_calibrated_zone=in_box,
                    state=TrackState.ACTIVE.value,
                )
                tracked_players.append(tracked_player)

        self.active_tracks_per_frame.append(len(tracked_players))
        return tracked_players, tracker_ms

    def get_track_summary_records(self) -> List[Dict[str, Any]]:
        """Return list of track summary dicts for CSV export."""
        records = []
        for track_id, stat in sorted(self.track_stats.items()):
            confs = stat["confidences"]
            mean_conf = float(np.mean(confs)) if confs else 0.0
            track_length = int(stat["hits"])
            records.append({
                "track_id": track_id,
                "first_frame": stat["first_frame"],
                "last_frame": stat["last_frame"],
                "track_length": track_length,
                "mean_confidence": round(mean_conf, 4),
            })
        return records

    def get_track_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive track population statistics."""
        summary_records = self.get_track_summary_records()
        total_unique = len(summary_records)
        lengths = [r["track_length"] for r in summary_records]
        active_counts = self.active_tracks_per_frame

        if lengths:
            avg_len = float(np.mean(lengths))
            median_len = float(np.median(lengths))
            min_len = int(np.min(lengths))
            max_len = int(np.max(lengths))
        else:
            avg_len = median_len = min_len = max_len = 0.0

        if active_counts:
            avg_active = float(np.mean(active_counts))
            median_active = float(np.median(active_counts))
            max_active = int(np.max(active_counts))
        else:
            avg_active = median_active = max_active = 0.0

        return {
            "total_unique_track_ids": total_unique,
            "average_track_length_frames": round(avg_len, 2),
            "median_track_length_frames": round(median_len, 2),
            "min_track_length_frames": min_len,
            "max_track_length_frames": max_len,
            "average_active_tracks_per_frame": round(avg_active, 2),
            "median_active_tracks_per_frame": round(median_active, 2),
            "maximum_active_tracks_per_frame": max_active,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return tracker configuration dictionary."""
        return {
            "tracker_name": "ByteTrack",
            "track_activation_threshold": self.track_activation_threshold,
            "lost_track_buffer": self.lost_track_buffer,
            "minimum_matching_threshold": self.minimum_matching_threshold,
            "frame_rate": self.frame_rate,
            "trail_length": self.trail_length,
        }
