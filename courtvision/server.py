"""Local web dashboard for the existing CourtVision video pipelines."""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


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

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


class JobRequest(BaseModel):
    video_id: str
    mode: Literal["detection", "tracking"]


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


def _job_payload(job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
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
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                relative = path.relative_to(OUTPUT_DIR).as_posix()
                files.append({"name": path.name, "url": f"/outputs/{relative}"})

    try:
        log = job["log_path"].read_text(encoding="utf-8", errors="replace")[-12000:]
    except FileNotFoundError:
        log = ""

    return {
        "id": job_id,
        "mode": job["mode"],
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
    return {
        "model_available": WEIGHTS_FILE.is_file(),
        "homography_available": HOMOGRAPHY_FILE.is_file(),
        "video_count": len(_list_videos()),
        "active_job": active,
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
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found. Jobs are cleared when the server restarts.")
        return _job_payload(job_id, job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        process: subprocess.Popen[bytes] = job["process"]
        if process.poll() is None:
            job["cancelled"] = True
            process.terminate()
        return _job_payload(job_id, job)
