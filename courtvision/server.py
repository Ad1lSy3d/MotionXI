from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


from courtvision.analytics.metrics import compute_player_metrics
from courtvision.tracking.track_types import TrackedPlayer
from courtvision.utils.video import finalize_video_for_web

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
WEIGHTS_FILE = ROOT_DIR / "weights" / "best.pt"
HOMOGRAPHY_FILE = DATA_DIR / "homography_matrix.json"
OUTPUT_DIR = ROOT_DIR / "outputs"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}

app = FastAPI(title="CourtVision Local Dashboard")
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR, check_dir=False), name="outputs")
app.mount("/data", StaticFiles(directory=DATA_DIR, check_dir=False), name="data")

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


class JobRequest(BaseModel):
    video_id: str
    mode: Literal["detection", "tracking"]
    tracker: Literal["bytetrack", "kmt"] = "bytetrack"


def _list_videos() -> list[Dict[str, str]]:
    videos: list[Dict[str, str]] = []
    roots = (DATA_DIR, UPLOAD_DIR)
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            try:
                video_id = path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
            except ValueError:
                continue
            if video_id in seen:
                continue
            seen.add(video_id)
            display_name = path.name.split("-", 1)[1] if root == UPLOAD_DIR else path.name
            videos.append({"id": video_id, "name": display_name})
    return videos


def _get_video_path(video_id: str) -> Path:
    video = next((item for item in _list_videos() if item["id"] == video_id), None)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found. Refresh the video list and try again.")
    path = (DATA_DIR / Path(video_id)).resolve()
    try:
        path.relative_to(DATA_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid video selection.") from exc
    return path


def _resolve_job(job_id: str) -> Optional[Tuple[Dict[str, Any], Path]]:
    with _jobs_lock:
        if job_id in _jobs:
            return _jobs[job_id], _jobs[job_id]["output_dir"]
    if job_id == "demo" and (OUTPUT_DIR / "demo").is_dir():
        mock_job = {
            "process": None,
            "mode": "tracking",
            "tracker": "kmt",
            "video_name": "test_match.mp4",
            "output_dir": OUTPUT_DIR / "demo",
            "log_path": OUTPUT_DIR / "demo" / "server.log",
            "cancelled": False,
            "is_disk_job": True,
        }
        return mock_job, OUTPUT_DIR / "demo"
    if (OUTPUT_DIR / "jobs" / job_id).is_dir():
        mock_job = {
            "process": None,
            "mode": "tracking",
            "tracker": "kmt",
            "video_name": "pipeline_output.mp4",
            "output_dir": OUTPUT_DIR / "jobs" / job_id,
            "log_path": OUTPUT_DIR / "jobs" / job_id / "server.log",
            "cancelled": False,
            "is_disk_job": True,
        }
        return mock_job, OUTPUT_DIR / "jobs" / job_id
    if (OUTPUT_DIR / job_id).is_dir():
        tracker_name = "bytetrack" if (OUTPUT_DIR / job_id / "bytetrack_annotated.mp4").is_file() else "kmt"
        mock_job = {
            "process": None,
            "mode": "tracking",
            "tracker": tracker_name,
            "video_name": "test_match.mp4",
            "output_dir": OUTPUT_DIR / job_id,
            "log_path": OUTPUT_DIR / job_id / "server.log",
            "cancelled": False,
            "is_disk_job": True,
        }
        return mock_job, OUTPUT_DIR / job_id
    return None


def _build_job_analytics(job_id: str, output_dir: Path) -> Dict[str, Any]:
    if not output_dir.is_dir():
        return {"job_id": job_id, "has_analytics": False, "message": "Output directory not found."}

    tracks_json = output_dir / "tracks.json"
    tracks_csv = output_dir / "tracks.csv"
    metrics_json = output_dir / "player_physical_metrics.json"
    metrics_csv = output_dir / "player_physical_metrics.csv"
    summary_csv = output_dir / "track_summary.csv"
    meta_json = output_dir / "tracking_metadata.json"
    crowded_csv = output_dir / "crowded_frames.csv"
    id_sw_csv = output_dir / "id_switch_candidates.csv"

    if not (tracks_json.is_file() or tracks_csv.is_file() or metrics_json.is_file()):
        return {"job_id": job_id, "has_analytics": False, "message": "Tracking analytics not yet generated for this job."}

    meta: Dict[str, Any] = {}
    if meta_json.is_file():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    tracks: List[Dict[str, Any]] = []
    if tracks_json.is_file():
        try:
            tracks = json.loads(tracks_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    elif tracks_csv.is_file():
        try:
            with tracks_csv.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    tracks.append({
                        "frame": int(row["frame"]),
                        "timestamp": float(row.get("timestamp", 0.0)),
                        "track_id": int(row["track_id"]),
                        "confidence": float(row.get("confidence", 0.0)),
                        "x1": float(row["x1"]),
                        "y1": float(row["y1"]),
                        "x2": float(row["x2"]),
                        "y2": float(row["y2"]),
                        "pitch_x": float(row["pitch_x"]) if row.get("pitch_x") not in (None, "", "None") else None,
                        "pitch_y": float(row["pitch_y"]) if row.get("pitch_y") not in (None, "", "None") else None,
                        "in_calibrated_zone": row.get("in_calibrated_zone", "").lower() in ("true", "1"),
                        "state": row.get("state", "active"),
                    })
        except Exception:
            pass

    metrics_list: List[Dict[str, Any]] = []
    if metrics_json.is_file():
        try:
            metrics_list = json.loads(metrics_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    elif metrics_csv.is_file():
        try:
            with metrics_csv.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    metrics_list.append({
                        "track_id": int(row["track_id"]),
                        "distance_km": float(row.get("distance_km", 0.0)),
                        "avg_speed_kmh": float(row["avg_speed_kmh"]) if row.get("avg_speed_kmh") not in (None, "", "None") else None,
                        "top_speed_kmh": float(row["top_speed_kmh"]) if row.get("top_speed_kmh") not in (None, "", "None") else None,
                        "max_raw_speed_kmh": float(row["max_raw_speed_kmh"]) if row.get("max_raw_speed_kmh") not in (None, "", "None") else None,
                        "max_acceleration_ms2": float(row["max_acceleration_ms2"]) if row.get("max_acceleration_ms2") not in (None, "", "None") else None,
                        "outlier_count": int(row.get("outlier_count", 0)),
                        "sprint_count": int(row.get("sprint_count", 0)),
                        "time_span_s": float(row["time_span_s"]) if row.get("time_span_s") not in (None, "", "None") else None,
                        "coverage_ratio": float(row.get("coverage_ratio", 0.0)),
                        "valid_samples": int(row.get("valid_samples", 0)),
                        "total_samples": int(row.get("total_samples", 0)),
                        "occlusion_gaps": int(row.get("occlusion_gaps", 0)),
                    })
        except Exception:
            pass

    video_meta = meta.get("video", {})
    exec_meta = meta.get("execution", {})
    tracker_meta = meta.get("tracker", {})
    calib_meta = meta.get("homography", {})

    fps = float(video_meta.get("fps") or 30.0)

    # Determine calibrated zone bounds strictly from metadata (defaulting to 18-yard box: X [0.0, 16.5], Y [13.84, 54.16])
    cal_x_min, cal_x_max = 0.0, 16.5
    cal_y_min, cal_y_max = 13.84, 54.16
    zone = calib_meta.get("calibrated_zone_m")
    if zone and "x_range" in zone and "y_range" in zone:
        cal_x_min, cal_x_max = float(zone["x_range"][0]), float(zone["x_range"][1])
        cal_y_min, cal_y_max = float(zone["y_range"][0]), float(zone["y_range"][1])
    elif calib_meta.get("full_pitch_calibration", False):
        cal_x_min, cal_x_max = 0.0, float(video_meta.get("pitch_length", 105.0))
        cal_y_min, cal_y_max = 0.0, float(video_meta.get("pitch_width", 68.0))

    def _is_calibrated(px: Optional[float], py: Optional[float]) -> bool:
        if px is None or py is None or not math.isfinite(px) or not math.isfinite(py):
            return False
        if px <= 0.0 and math.copysign(1.0, px) < 0.0:
            return False
        return (cal_x_min <= px <= cal_x_max) and (cal_y_min <= py <= cal_y_max)

    # Convert tracks into TrackedPlayer instances to guarantee consistent, outlier-validated metrics
    fresh_metrics_map = {}
    all_outlier_events: List[Dict[str, Any]] = []
    if tracks:
        tracked_objs = [
            TrackedPlayer.create(
                frame=int(t["frame"]),
                timestamp=float(t.get("timestamp", 0.0)),
                track_id=int(t["track_id"]),
                class_id=0,
                class_name="player",
                confidence=float(t.get("confidence", 0.9)),
                x1=float(t["x1"]),
                y1=float(t["y1"]),
                x2=float(t["x2"]),
                y2=float(t["y2"]),
                pitch_x=t.get("pitch_x"),
                pitch_y=t.get("pitch_y"),
                projection_valid=bool(t.get("projection_valid", True)) and (t.get("pitch_x") is not None and t.get("pitch_y") is not None),
                in_calibrated_zone=_is_calibrated(t.get("pitch_x"), t.get("pitch_y")) and bool(t.get("projection_valid", True)),
            )
            for t in tracks
        ]
        computed = compute_player_metrics(
            tracked_objs,
            fps=fps,
            max_plausible_speed_kmh=38.0,
            smooth_window=5,
        )
        fresh_metrics_map = {tid: m.to_dict() for tid, m in computed.items()}
        for tid, m in computed.items():
            for ev in m.outlier_events:
                all_outlier_events.append({
                    "track_id": tid,
                    "tag": f"P{tid:02d}",
                    "from_frame": ev["from_frame"],
                    "to_frame": ev["to_frame"],
                    "dt_s": ev["dt_s"],
                    "distance_m": ev["distance_m"],
                    "raw_speed_kmh": ev["raw_speed_kmh"],
                    "threshold_kmh": ev["threshold_kmh"],
                    "reason": ev["reason"],
                })
        all_outlier_events.sort(key=lambda x: x["raw_speed_kmh"], reverse=True)

    metrics_map = {int(m["track_id"]): m for m in metrics_list} if metrics_list else fresh_metrics_map

    all_outlier_events: List[Dict[str, Any]] = []
    for m in metrics_map.values():
        for ev in m.get("outlier_events", []):
            tid = int(m.get("track_id", 0))
            all_outlier_events.append({
                "track_id": tid,
                "tag": f"P{tid:02d}",
                "from_frame": ev.get("from_frame", 0),
                "to_frame": ev.get("to_frame", 0),
                "dt_s": ev.get("dt_s", 0.0),
                "distance_m": ev.get("distance_m", 0.0),
                "raw_speed_kmh": ev.get("raw_speed_kmh", 0.0),
                "threshold_kmh": ev.get("threshold_kmh", 38.0),
                "reason": ev.get("reason", ""),
            })
    all_outlier_events.sort(key=lambda x: x["raw_speed_kmh"], reverse=True)

    summary_map: Dict[int, Dict[str, Any]] = {}
    if summary_csv.is_file():
        try:
            with summary_csv.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    summary_map[int(row["track_id"])] = row
        except Exception:
            pass

    all_player_ids = sorted(list(set(t["track_id"] for t in tracks)))
    player_trajectories: Dict[int, List[Dict[str, Any]]] = {pid: [] for pid in all_player_ids}
    player_pitch_points: Dict[int, List[Tuple[float, float]]] = {pid: [] for pid in all_player_ids}
    frames_index: Dict[int, List[Dict[str, Any]]] = {}
    frame_tactical: Dict[int, List[Tuple[float, float]]] = {}
    frame_track_counts: Dict[int, set[int]] = {}
    track_durations: Dict[int, int] = {}

    for t in tracks:
        f_idx = t["frame"]
        pid = t["track_id"]
        px = t.get("pitch_x")
        py = t.get("pitch_y")
        is_valid = _is_calibrated(px, py) and bool(t.get("projection_valid", True))

        if f_idx not in frames_index:
            frames_index[f_idx] = []
        if f_idx not in frame_tactical:
            frame_tactical[f_idx] = []
        frame_track_counts.setdefault(f_idx, set()).add(pid)
        track_durations[pid] = track_durations.get(pid, 0) + 1

        hue = (pid * 137.5) % 360
        color = f"hsl({hue:.0f}, 85%, 55%)"

        frames_index[f_idx].append({
            "frame_index": f_idx,
            "timestamp": round(float(t.get("timestamp", f_idx / fps)), 3),
            "tracker_id": pid,
            "id": pid,
            "player_tag": f"P{pid:02d}",
            "tag": f"P{pid:02d}",
            "color": color,
            "bbox": [round(t["x1"], 1), round(t["y1"], 1), round(t["x2"], 1), round(t["y2"], 1)],
            "pitch_x": round(px, 2) if px is not None else None,
            "pitch_y": round(py, 2) if py is not None else None,
            "pitch": [round(px, 2), round(py, 2)] if px is not None and py is not None else None,
            "calibrated": is_valid,
            "valid": is_valid,
            "in_pitch_bounds": bool(px is not None and py is not None and math.isfinite(px) and math.isfinite(py) and 0.0 <= px <= 105.0 and 0.0 <= py <= 68.0),
            "state": t.get("state", "active"),
            "conf": round(float(t.get("confidence", 0.0)), 2),
        })

        if is_valid:
            frame_tactical[f_idx].append((px, py))
            player_pitch_points[pid].append((px, py))

        if px is not None and py is not None and math.isfinite(px) and math.isfinite(py):
            player_trajectories[pid].append({
                "f": f_idx,
                "x": round(px, 2),
                "y": round(py, 2),
                "v": is_valid,
            })

    # Track count analysis (separating unique tracker IDs from simultaneous active players)
    simul_counts = [len(s) for s in frame_track_counts.values()]
    peak_simultaneous = max(simul_counts) if simul_counts else 0
    avg_simultaneous = float(sum(simul_counts) / len(simul_counts)) if simul_counts else 0.0
    sustained_tracks_count = sum(1 for d in track_durations.values() if d >= 15)
    transient_tracks_count = sum(1 for d in track_durations.values() if d < 15)

    active_frame_stats = []
    all_widths = []
    all_lengths = []
    all_centroids_x = []
    all_centroids_y = []

    for f_idx in sorted(frame_tactical.keys()):
        pts = frame_tactical[f_idx]
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            w = max(ys) - min(ys)
            l = max(xs) - min(xs)
            cx = float(sum(xs) / len(xs))
            cy = float(sum(ys) / len(ys))
            all_widths.append(w)
            all_lengths.append(l)
            all_centroids_x.append(cx)
            all_centroids_y.append(cy)
            active_frame_stats.append({
                "frame": f_idx,
                "player_count": len(pts),
                "width_m": round(w, 1),
                "length_m": round(l, 1),
                "centroid": [round(cx, 1), round(cy, 1)],
            })

    team_hm_path = output_dir / "heatmaps" / "team_heatmap.png"
    team_hm_url = f"/outputs/{team_hm_path.relative_to(OUTPUT_DIR).as_posix()}" if team_hm_path.is_file() else None

    tactical = {
        "avg_team_width_m": round(float(sum(all_widths) / len(all_widths)), 1) if all_widths else 0.0,
        "avg_team_length_m": round(float(sum(all_lengths) / len(all_lengths)), 1) if all_lengths else 0.0,
        "team_centroid": [
            round(float(sum(all_centroids_x) / len(all_centroids_x)), 1) if all_centroids_x else 52.5,
            round(float(sum(all_centroids_y) / len(all_centroids_y)), 1) if all_centroids_y else 34.0,
        ],
        "team_heatmap_url": team_hm_url,
        "timeline": active_frame_stats[::2],
    }

    players = []
    total_team_dist_m = 0.0
    max_speed_kmh = 0.0
    total_sprints = 0

    for pid in all_player_ids:
        m = metrics_map.get(pid, {})
        s = summary_map.get(pid, {})

        dist_m = round(float(m.get("distance_km", 0.0)) * 1000.0, 1)
        total_team_dist_m += dist_m

        spd = m.get("avg_speed_kmh")
        top_spd = m.get("top_speed_kmh")
        if top_spd and top_spd > max_speed_kmh:
            max_speed_kmh = top_spd

        sprints = int(m.get("sprint_count", 0))
        total_sprints += sprints

        pts = player_pitch_points[pid]
        avg_pos = [round(float(sum(p[0] for p in pts) / len(pts)), 1), round(float(sum(p[1] for p in pts) / len(pts)), 1)] if pts else None

        hue = (pid * 137.5) % 360
        color = f"hsl({hue:.0f}, 85%, 55%)"

        hm_file = output_dir / "heatmaps" / f"player_{pid}_heatmap.png"
        hm_url = f"/outputs/{hm_file.relative_to(OUTPUT_DIR).as_posix()}" if hm_file.is_file() else None

        players.append({
            "track_id": pid,
            "tag": f"P{pid:02d}",
            "color": color,
            "distance_m": dist_m,
            "distance_km": round(float(m.get("distance_km", 0.0)), 3),
            "avg_speed_kmh": round(float(spd), 1) if spd is not None else None,
            "top_speed_kmh": round(float(top_spd), 1) if top_spd is not None else None,
            "max_raw_speed_kmh": round(float(m["max_raw_speed_kmh"]), 1) if m.get("max_raw_speed_kmh") is not None else None,
            "max_acceleration_ms2": m.get("max_acceleration_ms2"),
            "outlier_count": int(m.get("outlier_count", 0)),
            "sprint_count": sprints,
            "time_span_s": round(float(m["time_span_s"]), 1) if m.get("time_span_s") is not None else None,
            "coverage_ratio": round(float(m.get("coverage_ratio", 0.0)) * 100.0, 1),
            "valid_samples": int(m.get("valid_samples", 0)),
            "total_samples": int(m.get("total_samples", len(player_trajectories[pid]))),
            "occlusion_gaps": int(m.get("occlusion_gaps", 0)),
            "avg_confidence": round(float(s.get("avg_confidence", 0.0)), 2) if "avg_confidence" in s else None,
            "first_frame": int(s.get("first_frame", 0)) if "first_frame" in s else (player_trajectories[pid][0]["f"] if player_trajectories[pid] else 0),
            "last_frame": int(s.get("last_frame", 0)) if "last_frame" in s else (player_trajectories[pid][-1]["f"] if player_trajectories[pid] else 0),
            "avg_pitch_pos": avg_pos,
            "heatmap_url": hm_url,
            "trajectory": player_trajectories[pid],
            "has_calibrated_data": int(m.get("valid_samples", 0)) >= 2,
            "is_sustained": track_durations.get(pid, 0) >= 15,
        })

    players.sort(key=lambda p: (p["distance_m"], p["total_samples"]), reverse=True)

    crowded_list = []
    if crowded_csv.is_file():
        try:
            with crowded_csv.open("r", encoding="utf-8") as f:
                for row in list(csv.DictReader(f))[:10]:
                    crowded_list.append({
                        "frame": int(row["frame"]),
                        "timestamp": float(row["timestamp"]),
                        "player_count": int(row["player_count"]),
                        "max_pairwise_iou": float(row["max_pairwise_iou"]),
                        "overlapping_pairs": int(row["overlapping_pairs"]),
                    })
        except Exception:
            pass

    id_sw_list = []
    if id_sw_csv.is_file():
        try:
            with id_sw_csv.open("r", encoding="utf-8") as f:
                for row in list(csv.DictReader(f))[:15]:
                    id_sw_list.append({
                        "frame": int(row.get("frame", 0)),
                        "timestamp": float(row.get("timestamp", 0.0)),
                        "candidate_type": row.get("candidate_type", ""),
                        "track_id_1": row.get("track_id_1", ""),
                        "track_id_2": row.get("track_id_2", ""),
                        "description": row.get("description", ""),
                    })
        except Exception:
            pass

    dur = video_meta.get("duration_sec")
    if dur is None:
        dur = round(len(frames_index) / fps, 1) if frames_index else 0.0

    # Locate web-playable annotated video and original clean video
    annotated_video_url = None
    clean_video_url = None
    for cand_name in ["kmt_annotated.mp4", "bytetrack_annotated.mp4", "pipeline_output.mp4"]:
        v_cand = output_dir / cand_name
        if v_cand.is_file():
            finalize_video_for_web(v_cand)
            annotated_video_url = f"/outputs/{v_cand.relative_to(OUTPUT_DIR).as_posix()}"
            break

    input_vid_name = video_meta.get("input_video")
    if input_vid_name and (DATA_DIR / Path(input_vid_name).name).is_file():
        clean_video_url = f"/data/{Path(input_vid_name).name}"
    elif (DATA_DIR / "test_match.mp4").is_file():
        clean_video_url = "/data/test_match.mp4"

    summary = {
        "video_name": video_meta.get("input_video", "Match Video"),
        "duration_sec": dur,
        "total_frames": video_meta.get("decoded_frame_count", len(frames_index)),
        "fps": fps,
        "resolution": f"{video_meta.get('width', 1920)}x{video_meta.get('height', 1080)}",
        "total_unique_tracks": len(all_player_ids),
        "peak_simultaneous_tracks": peak_simultaneous,
        "avg_simultaneous_tracks": round(avg_simultaneous, 1),
        "sustained_tracks_count": sustained_tracks_count,
        "transient_tracks_count": transient_tracks_count,
        "calibrated_tracks": sum(1 for p in players if p["has_calibrated_data"]),
        "total_team_distance_m": round(total_team_dist_m, 1),
        "match_top_speed_kmh": round(max_speed_kmh, 1) if max_speed_kmh > 0 else (players[0]["top_speed_kmh"] if players and players[0]["top_speed_kmh"] else None),
        "total_sprints": total_sprints,
        "outliers_rejected_count": len(all_outlier_events),
        "max_raw_spike_kmh": round(max((ev["raw_speed_kmh"] for ev in all_outlier_events), default=0.0), 1),
        "annotated_video_url": annotated_video_url,
        "clean_video_url": clean_video_url,
        "tracker_name": tracker_meta.get("tracker_name", "KinematicMemoryTracker"),
        "detector_model": meta.get("detector", {}).get("model", "YOLOv11m"),
        "avg_detector_infer_time_ms": exec_meta.get("avg_detector_infer_time_ms"),
        "avg_tracker_assoc_time_ms": exec_meta.get("avg_tracker_assoc_time_ms"),
        "effective_processing_fps": exec_meta.get("effective_processing_fps"),
        "crowded_frames_count": len(crowded_list),
        "id_switch_candidates_count": meta.get("id_switch_diagnostics", {}).get("heuristic_candidate_events_count", len(id_sw_list)),
        "camera_motion_enabled": tracker_meta.get("enable_camera_motion", False),
        "camera_motion_stats": tracker_meta.get("camera_motion_stats"),
        "calibration_scope": calib_meta.get("calibration_scope", "penalty_box_region"),
        "calibration_warning": (
            "Physical distance and speed metrics are calculated exclusively from player coordinates "
            "within the calibrated 18-yard penalty box zone. Uncalibrated field areas suffer from perspective "
            "divergence and are excluded from metric accumulation."
        ),
    }

    return {
        "job_id": job_id,
        "has_analytics": True,
        "summary": summary,
        "players": players,
        "tactical": tactical,
        "diagnostics": {
            "speed_outliers": all_outlier_events[:25],
            "crowded_frames": crowded_list,
            "id_switch_candidates": id_sw_list,
            "tracking_fragmentation": {
                "total_tracker_ids": len(all_player_ids),
                "peak_simultaneous": peak_simultaneous,
                "avg_simultaneous": round(avg_simultaneous, 1),
                "sustained_tracks": sustained_tracks_count,
                "transient_tracks": transient_tracks_count,
            },
        },
        "frames": frames_index,
    }


def _job_payload(job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
    if job.get("is_disk_job"):
        status = "completed"
        return_code = 0
    else:
        process: subprocess.Popen[bytes] = job["process"]
        return_code = process.poll()
        if job["cancelled"]:
            status = "cancelled" if return_code is not None else "running"
        elif return_code is None:
            status = "running"
        else:
            status = "completed" if return_code == 0 else "failed"

    output_dir: Path = job["output_dir"]
    files = []
    if output_dir.is_dir():
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                relative = path.relative_to(OUTPUT_DIR).as_posix()
                display_name = path.relative_to(output_dir).as_posix()
                files.append({"name": display_name, "url": f"/outputs/{relative}"})

    try:
        log = job["log_path"].read_text(encoding="utf-8", errors="replace")[-12000:]
    except (FileNotFoundError, OSError):
        log = "Job completed." if job.get("is_disk_job") else ""

    return {
        "id": job_id,
        "mode": job["mode"],
        "tracker": job["tracker"],
        "video_name": job["video_name"],
        "status": status,
        "return_code": return_code,
        "files": files,
        "log": log,
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def status() -> Dict[str, Any]:
    with _jobs_lock:
        active = None
        for job_id, job in reversed(list(_jobs.items())):
            payload = _job_payload(job_id, job)
            if payload["status"] == "running":
                active = payload
                break
    demo_ready = (OUTPUT_DIR / "demo" / "kmt_annotated.mp4").is_file()
    return {
        "model_available": WEIGHTS_FILE.is_file(),
        "homography_available": HOMOGRAPHY_FILE.is_file(),
        "video_count": len(_list_videos()),
        "active_job": active,
        "demo_available": demo_ready,
    }


@app.get("/api/videos")
def videos() -> Dict[str, Any]:
    return {"videos": _list_videos(), "upload_limit_bytes": MAX_UPLOAD_BYTES}


@app.post("/api/videos", status_code=201)
async def upload_video(
    request: Request,
    filename: str = Query(..., min_length=5, max_length=128),
) -> Dict[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,119}\.(?:mp4|mov|avi|mkv)", filename, re.IGNORECASE):
        raise HTTPException(
            status_code=400,
            detail="Use a plain video filename ending in .mp4, .mov, .avi, or .mkv.",
        )

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Video exceeds the 512 MB upload limit.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from exc

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}-{filename}"
    total_bytes = 0
    try:
        with saved_path.open("wb") as uploaded:
            async for chunk in request.stream():
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds the 512 MB upload limit.")
                uploaded.write(chunk)
    except Exception:
        saved_path.unlink(missing_ok=True)
        raise

    if total_bytes == 0:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded video is empty.")

    return {
        "id": saved_path.resolve().relative_to(DATA_DIR.resolve()).as_posix(),
        "name": filename,
    }


@app.post("/api/jobs", status_code=202)
def start_job(request: JobRequest) -> Dict[str, Any]:
    video_path = _get_video_path(request.video_id)
    if not WEIGHTS_FILE.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model weights are missing. Place the team-provided checkpoint at weights/best.pt first.",
        )

    job_id = uuid.uuid4().hex
    output_dir = OUTPUT_DIR / "jobs" / job_id
    log_path = output_dir / "server.log"
    script_name = "run_detection.py" if request.mode == "detection" else "run_tracking.py"
    script_path = ROOT_DIR / "scripts" / script_name
    command = [
        sys.executable,
        str(script_path),
        "--video",
        str(video_path),
        "--weights",
        str(WEIGHTS_FILE),
        "--output-dir",
        str(output_dir),
    ]
    if HOMOGRAPHY_FILE.is_file():
        command.extend(["--homography", str(HOMOGRAPHY_FILE)])
    if request.mode == "tracking":
        command.extend(["--tracker", request.tracker])

    with _jobs_lock:
        for existing_id, existing_job in _jobs.items():
            if _job_payload(existing_id, existing_job)["status"] == "running":
                raise HTTPException(status_code=409, detail="A pipeline job is already running.")
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            with log_path.open("wb") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT_DIR,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not start pipeline: {exc}") from exc

        _jobs[job_id] = {
            "process": process,
            "mode": request.mode,
            "tracker": request.tracker if request.mode == "tracking" else None,
            "video_name": video_path.name,
            "output_dir": output_dir,
            "log_path": log_path,
            "cancelled": False,
        }
        while len(_jobs) > 20:
            oldest_id = next(iter(_jobs))
            oldest_job = _jobs[oldest_id]
            if _job_payload(oldest_id, oldest_job)["status"] == "running":
                break
            del _jobs[oldest_id]

    return _job_payload(job_id, _jobs[job_id])


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    resolved = _resolve_job(job_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Job not found. Jobs are cleared when the server restarts.")
    job, _ = resolved
    return _job_payload(job_id, job)


@app.get("/api/jobs/{job_id}/analytics")
def get_job_analytics(job_id: str) -> Dict[str, Any]:
    resolved = _resolve_job(job_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    _, output_dir = resolved
    return _build_job_analytics(job_id, output_dir)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        process: Optional[subprocess.Popen[bytes]] = job.get("process")
        if process is not None and process.poll() is None:
            job["cancelled"] = True
            process.terminate()
        return _job_payload(job_id, job)
