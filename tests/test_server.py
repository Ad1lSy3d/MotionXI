from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from courtvision import server


@pytest.fixture
def client(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    upload_dir = data_dir / "uploads"
    output_dir = tmp_path / "outputs"
    upload_dir.mkdir(parents=True)
    output_dir.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data_dir)
    monkeypatch.setattr(server, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(server, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(server, "WEIGHTS_FILE", tmp_path / "weights" / "best.pt")
    monkeypatch.setattr(server, "HOMOGRAPHY_FILE", data_dir / "homography_matrix.json")
    with server._jobs_lock:
        server._jobs.clear()
    with TestClient(server.app) as test_client:
        yield test_client
    with server._jobs_lock:
        server._jobs.clear()


def test_dashboard_and_status_are_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Choose a video" in response.text
    assert client.get("/assets/styles.css").status_code == 200
    assert client.get("/assets/app.js").status_code == 200

    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["model_available"] is False


def test_upload_video_and_list_it(client):
    response = client.post(
        "/api/videos?filename=match.mp4",
        content=b"sample video bytes",
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 201
    uploaded = response.json()
    assert uploaded["name"] == "match.mp4"
    assert (server.DATA_DIR / uploaded["id"]).read_bytes() == b"sample video bytes"
    assert uploaded in client.get("/api/videos").json()["videos"]


@pytest.mark.parametrize("filename", ["..%5Csecret.mp4", "notes.txt", "match.mp4%00"])
def test_upload_rejects_unsafe_or_unsupported_filename(client, filename):
    response = client.post(
        f"/api/videos?filename={filename}",
        content=b"sample",
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 400


def test_start_job_requires_model_checkpoint(client):
    upload = client.post(
        "/api/videos?filename=match.mp4",
        content=b"sample video bytes",
    ).json()

    response = client.post(
        "/api/jobs",
        json={"video_id": upload["id"], "mode": "tracking", "tracker": "kmt"},
    )

    assert response.status_code == 503
    assert "weights/best.pt" in response.json()["detail"]


def test_start_job_runs_existing_script(client, monkeypatch, tmp_path):
    upload = client.post(
        "/api/videos?filename=match.mp4",
        content=b"sample video bytes",
    ).json()
    server.WEIGHTS_FILE.parent.mkdir()
    server.WEIGHTS_FILE.write_bytes(b"model")
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(server.subprocess, "Popen", popen)

    response = client.post(
        "/api/jobs",
        json={"video_id": upload["id"], "mode": "tracking", "tracker": "kmt"},
    )

    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "running"
    assert job["tracker"] == "kmt"
    command = popen.call_args.args[0]
    assert command[1].endswith("run_tracking.py")
    assert "--video" in command
    assert "--weights" in command
    assert command[-2:] == ["--tracker", "kmt"]


def test_upload_size_limit_cleans_up_partial_file(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 4)

    response = client.post(
        "/api/videos?filename=match.mp4",
        content=b"this is larger than four bytes",
    )

    assert response.status_code == 413
    assert list(server.UPLOAD_DIR.iterdir()) == []


def test_start_job_defaults_to_bytetrack(client, monkeypatch):
    upload = client.post(
        "/api/videos?filename=match.mp4",
        content=b"sample video bytes",
    ).json()
    server.WEIGHTS_FILE.parent.mkdir()
    server.WEIGHTS_FILE.write_bytes(b"model")
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(server.subprocess, "Popen", popen)

    response = client.post(
        "/api/jobs",
        json={"video_id": upload["id"], "mode": "tracking"},
    )

    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "running"
    assert job["tracker"] == "bytetrack"
    command = popen.call_args.args[0]
    assert command[-2:] == ["--tracker", "bytetrack"]


def test_job_discovers_nested_heatmap_files(client):
    job_id = "test_discovery_job"
    output_dir = server.OUTPUT_DIR / "jobs" / job_id
    output_dir.mkdir(parents=True)
    heatmaps_dir = output_dir / "heatmaps"
    heatmaps_dir.mkdir()
    (heatmaps_dir / "team_heatmap.png").write_bytes(b"png_bytes")
    (output_dir / "tracks.csv").write_bytes(b"csv_bytes")
    (output_dir / "server.log").write_text("done", encoding="utf-8")

    process = Mock()
    process.poll.return_value = 0
    job_record = {
        "process": process,
        "mode": "tracking",
        "tracker": "kmt",
        "video_name": "match.mp4",
        "output_dir": output_dir,
        "log_path": output_dir / "server.log",
        "cancelled": False,
    }
    with server._jobs_lock:
        server._jobs[job_id] = job_record

    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    file_names = [f["name"] for f in data["files"]]
    assert "tracks.csv" in file_names
    assert "heatmaps/team_heatmap.png" in file_names
    heatmap_entry = next(f for f in data["files"] if f["name"] == "heatmaps/team_heatmap.png")
    assert heatmap_entry["url"] == f"/outputs/jobs/{job_id}/heatmaps/team_heatmap.png"


def test_job_analytics_endpoint_returns_structured_data(client):
    import json
    job_id = "test_analytics_job"
    output_dir = server.OUTPUT_DIR / "jobs" / job_id
    output_dir.mkdir(parents=True)
    heatmaps_dir = output_dir / "heatmaps"
    heatmaps_dir.mkdir()
    (heatmaps_dir / "team_heatmap.png").write_bytes(b"png_bytes")
    (heatmaps_dir / "player_1_heatmap.png").write_bytes(b"png_bytes")

    sample_tracks = [
        {
            "frame": 0,
            "timestamp": 0.0,
            "track_id": 1,
            "confidence": 0.9,
            "x1": 100.0,
            "y1": 150.0,
            "x2": 140.0,
            "y2": 250.0,
            "pitch_x": 10.0,
            "pitch_y": 20.0,
            "in_calibrated_zone": True,
            "state": "active",
        },
        {
            "frame": 1,
            "timestamp": 0.033,
            "track_id": 1,
            "confidence": 0.92,
            "x1": 102.0,
            "y1": 150.0,
            "x2": 142.0,
            "y2": 250.0,
            "pitch_x": 10.5,
            "pitch_y": 20.2,
            "in_calibrated_zone": True,
            "state": "active",
        },
    ]
    (output_dir / "tracks.json").write_text(json.dumps(sample_tracks), encoding="utf-8")

    sample_metrics = [
        {
            "track_id": 1,
            "distance_km": 0.005,
            "avg_speed_kmh": 15.2,
            "top_speed_kmh": 22.1,
            "sprint_count": 1,
            "time_span_s": 1.0,
            "coverage_ratio": 1.0,
            "valid_samples": 2,
            "total_samples": 2,
            "occlusion_gaps": 0,
        }
    ]
    (output_dir / "player_physical_metrics.json").write_text(json.dumps(sample_metrics), encoding="utf-8")

    meta = {
        "video": {"input_video": "match.mp4", "duration_sec": 3.0, "decoded_frame_count": 2, "fps": 30.0, "width": 1920, "height": 1080},
        "tracker": {"tracker_name": "KinematicMemoryTracker", "enable_camera_motion": True},
        "homography": {"calibration_scope": "penalty_box_region"},
    }
    (output_dir / "tracking_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    process = Mock()
    process.poll.return_value = 0
    with server._jobs_lock:
        server._jobs[job_id] = {
            "process": process,
            "mode": "tracking",
            "tracker": "kmt",
            "video_name": "match.mp4",
            "output_dir": output_dir,
            "log_path": output_dir / "server.log",
            "cancelled": False,
        }

    response = client.get(f"/api/jobs/{job_id}/analytics")
    assert response.status_code == 200
    data = response.json()
    assert data["has_analytics"] is True
    assert data["summary"]["total_unique_tracks"] == 1
    assert data["summary"]["total_team_distance_m"] == 5.0
    assert len(data["players"]) == 1
    p1 = data["players"][0]
    assert p1["tag"] == "P01"
    assert p1["distance_m"] == 5.0
    assert p1["heatmap_url"] == f"/outputs/jobs/{job_id}/heatmaps/player_1_heatmap.png"
    assert "frames" in data
    assert 0 in data["frames"] or "0" in data["frames"]


def test_job_analytics_endpoint_handles_missing_job(client):
    response = client.get("/api/jobs/nonexistent_job_123/analytics")
    assert response.status_code == 404


def test_job_analytics_endpoint_handles_unprocessed_job(client, tmp_path):
    job_id = "unprocessed_job"
    output_dir = tmp_path / "empty_dir"
    output_dir.mkdir()
    process = Mock()
    process.poll.return_value = None
    with server._jobs_lock:
        server._jobs[job_id] = {
            "process": process,
            "mode": "tracking",
            "tracker": "kmt",
            "video_name": "match.mp4",
            "output_dir": output_dir,
            "log_path": output_dir / "server.log",
            "cancelled": False,
        }
    response = client.get(f"/api/jobs/{job_id}/analytics")
    assert response.status_code == 200
    assert response.json()["has_analytics"] is False


