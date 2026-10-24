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

    response = client.post("/api/jobs", json={"video_id": upload["id"], "mode": "tracking"})

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
