from dataclasses import dataclass, asdict
from enum import Enum
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

# Ensure NumPy 2.x compatibility with TrackEval
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool


class FailureCategory(str, Enum):
    HEAVY_OVERLAP = "Heavy overlap / occlusion"
    PLAYER_CROSSING = "Player crossing"
    DETECTION_DROPOUT = "Detection dropout"
    NEW_IDENTITY_AFTER_DISAPPEARANCE = "New identity after disappearance"
    INCORRECT_REAPPEARANCE_ASSOC = "Incorrect association after reappearance"
    LONG_OCCLUSION = "Long occlusion"
    OTHER = "Other"


@dataclass
class GroundTruthBox:
    frame: int  # 1-indexed
    gt_id: int
    x1: float
    y1: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def x2(self) -> float:
        return self.x1 + self.width

    @property
    def y2(self) -> float:
        return self.y1 + self.height

    @property
    def box_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class TrackerBox:
    frame: int  # 1-indexed
    track_id: int
    x1: float
    y1: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def x2(self) -> float:
        return self.x1 + self.width

    @property
    def y2(self) -> float:
        return self.y1 + self.height

    @property
    def box_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union between two (x1, y1, x2, y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def parse_mot_gt_file(gt_path: Path) -> List[GroundTruthBox]:
    """Parse standard MOT format ground truth file."""
    boxes: List[GroundTruthBox] = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            gt_id = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            boxes.append(GroundTruthBox(frame=frame, gt_id=gt_id, x1=x1, y1=y1, width=w, height=h, confidence=conf))
    return boxes


def parse_tracker_csv(tracks_csv_path: Path) -> List[TrackerBox]:
    """
    Parse tracks from either:
    1) courtvision tracks.csv (0-based frame, header with frame,track_id,x1,y1,x2,y2,width,height)
    2) MOTChallenge format csv (1-based frame, frame,id,left,top,width,height,conf,-1,-1,-1)
    """
    boxes: List[TrackerBox] = []
    with open(tracks_csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return boxes

    first_line = lines[0]
    if "frame" in first_line.lower() and "track_id" in first_line.lower():
        # CourtVision CSV format
        headers = [h.strip() for h in first_line.split(",")]
        f_idx = headers.index("frame")
        tid_idx = headers.index("track_id")
        x1_idx = headers.index("x1")
        y1_idx = headers.index("y1")
        w_idx = headers.index("width")
        h_idx = headers.index("height")
        conf_idx = headers.index("confidence") if "confidence" in headers else -1

        for line in lines[1:]:
            parts = line.split(",")
            # Convert 0-indexed video frame to 1-indexed MOT frame
            frame = int(parts[f_idx]) + 1
            track_id = int(parts[tid_idx])
            x1 = float(parts[x1_idx])
            y1 = float(parts[y1_idx])
            w = float(parts[w_idx])
            h = float(parts[h_idx])
            conf = float(parts[conf_idx]) if conf_idx != -1 else 1.0
            boxes.append(TrackerBox(frame=frame, track_id=track_id, x1=x1, y1=y1, width=w, height=h, confidence=conf))
    else:
        # Standard MOT format (already 1-indexed)
        for line in lines:
            parts = line.split(",")
            frame = int(parts[0])
            track_id = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            boxes.append(TrackerBox(frame=frame, track_id=track_id, x1=x1, y1=y1, width=w, height=h, confidence=conf))

    return boxes


def identify_crowded_frames(
    boxes_by_frame: Dict[int, List[Tuple[float, float, float, float]]],
    min_overlapping_pairs: int = 1,
    min_iou: float = 0.20,
) -> Tuple[Set[int], Dict[int, Dict[str, Any]]]:
    """
    Classify frames into crowded vs normal based on player bbox overlaps and pairwise IoUs.

    Returns:
        crowded_frame_set: Set of 1-indexed frame numbers meeting crowding criteria
        frame_crowding_stats: Dict of frame -> {player_count, overlapping_pairs, max_pairwise_iou}
    """
    crowded_frames: Set[int] = set()
    stats: Dict[int, Dict[str, Any]] = {}

    for frame, bboxes in boxes_by_frame.items():
        n = len(bboxes)
        overlapping_pairs = 0
        max_iou = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                iou = compute_iou(bboxes[i], bboxes[j])
                if iou > max_iou:
                    max_iou = iou
                if iou >= min_iou:
                    overlapping_pairs += 1

        is_crowded = (overlapping_pairs >= min_overlapping_pairs) or (max_iou >= min_iou) or (n >= 18)
        if is_crowded:
            crowded_frames.add(frame)

        stats[frame] = {
            "player_count": n,
            "overlapping_pairs": overlapping_pairs,
            "max_pairwise_iou": round(float(max_iou), 4),
            "is_crowded": is_crowded,
        }

    return crowded_frames, stats


from scipy.optimize import linear_sum_assignment


def match_gt_to_tracker(
    gt_boxes: List[GroundTruthBox],
    tracker_boxes: List[TrackerBox],
    iou_threshold: float = 0.5,
) -> Tuple[Dict[int, int], List[int], List[int]]:
    """
    Perform Hungarian bipartite matching between GT boxes and Tracker boxes in a single frame.

    Returns:
        matches: Dict of gt_id -> tracker_id
        unmatched_gt_ids: List of gt_ids not matched
        unmatched_tracker_ids: List of tracker_ids not matched
    """
    if not gt_boxes or not tracker_boxes:
        return {}, [g.gt_id for g in gt_boxes], [t.track_id for t in tracker_boxes]

    cost_matrix = np.zeros((len(gt_boxes), len(tracker_boxes)), dtype=float)
    for i, g in enumerate(gt_boxes):
        for j, t in enumerate(tracker_boxes):
            iou = compute_iou(g.box_xyxy, t.box_xyxy)
            cost_matrix[i, j] = 1.0 - iou if iou >= iou_threshold else 1.0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: Dict[int, int] = {}
    matched_gt_indices: Set[int] = set()
    matched_tracker_indices: Set[int] = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < (1.0 - iou_threshold + 1e-5):
            matches[gt_boxes[r].gt_id] = tracker_boxes[c].track_id
            matched_gt_indices.add(r)
            matched_tracker_indices.add(c)

    unmatched_gt = [gt_boxes[i].gt_id for i in range(len(gt_boxes)) if i not in matched_gt_indices]
    unmatched_tr = [tracker_boxes[j].track_id for j in range(len(tracker_boxes)) if j not in matched_tracker_indices]

    return matches, unmatched_gt, unmatched_tr


@dataclass
class IDSwitchEvent:
    frame: int
    timestamp: float
    gt_id: int
    previous_tracker_id: int
    new_tracker_id: int
    bbox_overlap: Optional[float]
    player_count: int
    crowded_frame: bool
    failure_category: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_id_switch_events(
    gt_boxes: List[GroundTruthBox],
    tracker_boxes: List[TrackerBox],
    crowded_frames: Set[int],
    frame_stats: Dict[int, Dict[str, Any]],
    fps: float = 25.0,
) -> List[IDSwitchEvent]:
    """
    Identify actual identity switch events where a GT identity is assigned a different tracker ID.
    Categorizes the root cause into standard categories based on player overlap, gap duration, and density.
    """
    # Group boxes by frame
    gt_by_frame: Dict[int, List[GroundTruthBox]] = {}
    for g in gt_boxes:
        gt_by_frame.setdefault(g.frame, []).append(g)

    tr_by_frame: Dict[int, List[TrackerBox]] = {}
    for t in tracker_boxes:
        tr_by_frame.setdefault(t.frame, []).append(t)

    all_frames = sorted(set(gt_by_frame.keys()).union(tr_by_frame.keys()))

    # Track match history per gt_id: list of (frame, matched_tracker_id)
    gt_matches: Dict[int, List[Tuple[int, int]]] = {}

    for f in all_frames:
        frame_gts = gt_by_frame.get(f, [])
        frame_trs = tr_by_frame.get(f, [])
        matches, _, _ = match_gt_to_tracker(frame_gts, frame_trs, iou_threshold=0.5)
        for gid, tid in matches.items():
            gt_matches.setdefault(gid, []).append((f, tid))

    events: List[IDSwitchEvent] = []

    for gid, history in gt_matches.items():
        if len(history) < 2:
            continue

        prev_frame, prev_tid = history[0]
        for curr_frame, curr_tid in history[1:]:
            if curr_tid != prev_tid:
                # An ID switch occurred!
                gap = curr_frame - prev_frame
                f_stat = frame_stats.get(curr_frame, {"player_count": 0, "max_pairwise_iou": 0.0, "is_crowded": False})
                overlap = f_stat.get("max_pairwise_iou", 0.0)
                is_crowded = f_stat.get("is_crowded", False) or (curr_frame in crowded_frames)

                # Determine failure category
                if overlap >= 0.35:
                    cat = FailureCategory.HEAVY_OVERLAP.value
                elif gap > 20:
                    cat = FailureCategory.LONG_OCCLUSION.value
                elif gap > 3:
                    cat = FailureCategory.NEW_IDENTITY_AFTER_DISAPPEARANCE.value
                elif is_crowded:
                    cat = FailureCategory.PLAYER_CROSSING.value
                else:
                    cat = FailureCategory.INCORRECT_REAPPEARANCE_ASSOC.value

                events.append(
                    IDSwitchEvent(
                        frame=curr_frame,
                        timestamp=round((curr_frame - 1) / fps, 3),
                        gt_id=gid,
                        previous_tracker_id=prev_tid,
                        new_tracker_id=curr_tid,
                        bbox_overlap=round(overlap, 4) if overlap is not None else None,
                        player_count=f_stat.get("player_count", 0),
                        crowded_frame=is_crowded,
                        failure_category=cat,
                    )
                )

            prev_frame = curr_frame
            prev_tid = curr_tid

    events.sort(key=lambda e: (e.frame, e.gt_id))
    return events


@dataclass
class IdentityFragmentationRecord:
    gt_id: int
    number_of_tracker_ids: int
    longest_continuous_track: int
    number_of_fragments: int
    total_frames: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_identity_fragmentation(
    gt_boxes: List[GroundTruthBox],
    tracker_boxes: List[TrackerBox],
) -> List[IdentityFragmentationRecord]:
    """Calculate fragmentation statistics for each ground-truth identity."""
    gt_by_frame: Dict[int, List[GroundTruthBox]] = {}
    for g in gt_boxes:
        gt_by_frame.setdefault(g.frame, []).append(g)

    tr_by_frame: Dict[int, List[TrackerBox]] = {}
    for t in tracker_boxes:
        tr_by_frame.setdefault(t.frame, []).append(t)

    all_frames = sorted(set(gt_by_frame.keys()).union(tr_by_frame.keys()))
    gt_frame_counts: Dict[int, int] = {}
    for g in gt_boxes:
        gt_frame_counts[g.gt_id] = gt_frame_counts.get(g.gt_id, 0) + 1

    # Match sequence per gt_id: list of matched tracker_ids (or None if unmatched)
    gt_timeline: Dict[int, List[Tuple[int, Optional[int]]]] = {}
    for f in all_frames:
        frame_gts = gt_by_frame.get(f, [])
        frame_trs = tr_by_frame.get(f, [])
        matches, _, _ = match_gt_to_tracker(frame_gts, frame_trs, iou_threshold=0.5)
        for g in frame_gts:
            matched_tid = matches.get(g.gt_id, None)
            gt_timeline.setdefault(g.gt_id, []).append((f, matched_tid))

    records: List[IdentityFragmentationRecord] = []

    for gid, timeline in sorted(gt_timeline.items()):
        total_frames = gt_frame_counts.get(gid, len(timeline))
        # Unique tracker IDs assigned to this GT
        tracker_ids = set(tid for _, tid in timeline if tid is not None)

        # Count fragments (continuous runs of same tracker ID)
        current_tid = None
        current_len = 0
        max_continuous = 0
        fragments = 0

        for _, tid in timeline:
            if tid is not None:
                if tid == current_tid:
                    current_len += 1
                else:
                    if current_tid is not None:
                        fragments += 1
                    current_tid = tid
                    current_len = 1
                if current_len > max_continuous:
                    max_continuous = current_len
            else:
                if current_tid is not None:
                    fragments += 1
                    current_tid = None
                    current_len = 0

        if current_tid is not None:
            fragments += 1

        records.append(
            IdentityFragmentationRecord(
                gt_id=gid,
                number_of_tracker_ids=len(tracker_ids),
                longest_continuous_track=max_continuous,
                number_of_fragments=fragments,
                total_frames=total_frames,
            )
        )

    return records


@dataclass
class OcclusionRecoveryRecord:
    occlusion_start: int
    occlusion_end: int
    occlusion_duration: int
    identity_before: int
    identity_after: int
    identity_recovered: bool
    recovery_frames: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_occlusion_recovery(
    gt_boxes: List[GroundTruthBox],
    tracker_boxes: List[TrackerBox],
    min_gap_frames: int = 3,
) -> List[OcclusionRecoveryRecord]:
    """
    Measure recovery after tracking loss/occlusion.
    Find intervals where a GT identity had a track, went unmatched for >= min_gap_frames,
    then re-appeared and was matched again.
    """
    gt_by_frame: Dict[int, List[GroundTruthBox]] = {}
    for g in gt_boxes:
        gt_by_frame.setdefault(g.frame, []).append(g)

    tr_by_frame: Dict[int, List[TrackerBox]] = {}
    for t in tracker_boxes:
        tr_by_frame.setdefault(t.frame, []).append(t)

    all_frames = sorted(set(gt_by_frame.keys()).union(tr_by_frame.keys()))
    gt_matches: Dict[int, List[Tuple[int, Optional[int]]]] = {}

    for f in all_frames:
        frame_gts = gt_by_frame.get(f, [])
        frame_trs = tr_by_frame.get(f, [])
        matches, _, _ = match_gt_to_tracker(frame_gts, frame_trs, iou_threshold=0.5)
        for g in frame_gts:
            gt_matches.setdefault(g.gt_id, []).append((f, matches.get(g.gt_id, None)))

    records: List[OcclusionRecoveryRecord] = []

    for gid, timeline in gt_matches.items():
        matched_points = [(f, tid) for f, tid in timeline if tid is not None]
        for i in range(len(matched_points) - 1):
            f_prev, tid_prev = matched_points[i]
            f_next, tid_next = matched_points[i + 1]
            gap = f_next - f_prev - 1
            if gap >= min_gap_frames:
                recovered = (tid_prev == tid_next)
                records.append(
                    OcclusionRecoveryRecord(
                        occlusion_start=f_prev + 1,
                        occlusion_end=f_next - 1,
                        occlusion_duration=gap,
                        identity_before=tid_prev,
                        identity_after=tid_next,
                        identity_recovered=recovered,
                        recovery_frames=1 if recovered else None,
                    )
                )

    return records

