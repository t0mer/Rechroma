import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.jobs.models import Job
from app.main import create_app


def _png() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((16, 16, 3), 90, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _settings(tmp_path, **over) -> Settings:
    return Settings(data_dir=tmp_path / "data", device="cpu", models_dir=tmp_path / "m", **over)


def _echo_processor(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    def proc(job: Job) -> str:
        out = out_dir / f"{job.id}_result.png"
        Image.open(job.input_path).save(out)
        return str(out)

    return proc


def _client(tmp_path, **over) -> TestClient:
    settings = _settings(tmp_path, **over)
    app = create_app(settings, processor=_echo_processor(tmp_path / "results"))
    return TestClient(app)


def _wait_done(client, job_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/api/v1/jobs/{job_id}")
        if r.json()["status"] in ("done", "failed"):
            return r.json()
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_healthz(tmp_path):
    with _client(tmp_path) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["device"] == "cpu"


def test_metrics_prometheus(tmp_path):
    with _client(tmp_path) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "rechroma_up 1" in r.text
        assert "rechroma_queue_depth" in r.text


def test_create_and_fetch_job(tmp_path):
    with _client(tmp_path) as client:
        r = client.post(
            "/api/v1/jobs",
            files={"file": ("in.png", _png(), "image/png")},
            data={"preset": "colorize"},
        )
        assert r.status_code == 201, r.text
        job_id = r.json()["id"]
        assert r.json()["preset"] == "colorize"
        done = _wait_done(client, job_id)
        assert done["status"] == "done"
        assert done["has_result"] is True
        result = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        assert result.headers["content-type"] == "image/png"


def test_rejects_non_image_upload(tmp_path):
    with _client(tmp_path) as client:
        r = client.post(
            "/api/v1/jobs",
            files={"file": ("x.png", b"not an image", "image/png")},
        )
        assert r.status_code == 400


def test_invalid_preset_rejected(tmp_path):
    with _client(tmp_path) as client:
        r = client.post(
            "/api/v1/jobs",
            files={"file": ("in.png", _png(), "image/png")},
            data={"preset": "nonsense"},
        )
        assert r.status_code == 422


def test_missing_job_404(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/api/v1/jobs/deadbeef").status_code == 404


def test_auth_required_when_token_set(tmp_path):
    with _client(tmp_path, web_auth_token="secret") as client:
        assert client.get("/api/v1/jobs").status_code == 401
        ok = client.get("/api/v1/jobs", headers={"X-API-Token": "secret"})
        assert ok.status_code == 200
        bearer = client.get("/api/v1/jobs", headers={"Authorization": "Bearer secret"})
        assert bearer.status_code == 200


@pytest.mark.parametrize("tok", ["wrong", ""])
def test_auth_rejects_bad_token(tmp_path, tok):
    with _client(tmp_path, web_auth_token="secret") as client:
        assert client.get("/api/v1/jobs", headers={"X-API-Token": tok}).status_code == 401
