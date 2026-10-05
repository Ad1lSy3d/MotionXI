from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional


class TrackState(str, Enum):
    """
    Conceptual lifecycle states of a tracked player:
    - ACTIVE: Player currently matched with high confidence to a detection in the current frame.
    - LOST: Player temporarily undetected (e.g. brief occlusion, missed detection), but the tracker
      retains the identity in a buffer awaiting re-identification.
    - REMOVED: Tracker has permanently discarded the identity after exceeding lost buffer timeout.
    """
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


@dataclass(frozen=True)
class TrackedPlayer:
    """Represents a tracked player instance at a specific video frame."""
    frame: int
    timestamp: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    foot_x: float
    foot_y: float
    pitch_x: Optional[float] = None
    pitch_y: Optional[float] = None
    projection_valid: bool = False
    in_calibrated_zone: bool = False
    state: str = TrackState.ACTIVE.value

    @classmethod
    def create(
        cls,
        frame: int,
        timestamp: float,
        track_id: int,
        class_id: int,
        class_name: str,
        confidence: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        pitch_x: Optional[float] = None,
        pitch_y: Optional[float] = None,
        projection_valid: bool = False,
        in_calibrated_zone: bool = False,
        state: str = TrackState.ACTIVE.value,
    ) -> "TrackedPlayer":
        width = float(x2 - x1)
        height = float(y2 - y1)
        foot_x = float((x1 + x2) / 2.0)
        foot_y = float(y2)
        return cls(
            frame=int(frame),
            timestamp=float(timestamp),
            track_id=int(track_id),
            class_id=int(class_id),
            class_name=str(class_name),
            confidence=float(confidence),
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            width=width,
            height=height,
            foot_x=foot_x,
            foot_y=foot_y,
            pitch_x=float(pitch_x) if pitch_x is not None else None,
            pitch_y=float(pitch_y) if pitch_y is not None else None,
            projection_valid=bool(projection_valid),
            in_calibrated_zone=bool(in_calibrated_zone),
            state=str(state),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
