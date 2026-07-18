import io
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.jobs.models import Job
from app.jobs.processor import make_dispatch_processor
from app.main import create_app


def _png() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((16, 16, 3), 90, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _stub_animate(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def proc(job: Job) -> str:
        out = out_dir / f"{job.id}_result.mp4"
        out.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        return str(out)

    return proc


def _client(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", device="cpu", models_dir=tmp_path / "m")
    aproc = _stub_animate(tmp_path / "data" / "results")
    app = create_app(settings, processor=make_dispatch_processor(lambda j: "", lambda j: "", aproc))
    return TestClient(app)


def _wait_done(client, job_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if client.get(f"/api/v1/jobs/{job_id}").json()["status"] in ("done", "failed"):
            return client.get(f"/api/v1/jobs/{job_id}").json()
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_animate_creates_animate_job(tmp_path):
    with _client(tmp_path) as client:
        r = client.post(
            "/api/v1/jobs",
            files={"file": ("p.png", _png(), "image/png")},
            data={"preset": "animate"},
        )
        assert r.status_code == 201, r.text
        assert r.json()["kind"] == "animate"
        jid = r.json()["id"]
        done = _wait_done(client, jid)
        assert done["status"] == "done"
        res = client.get(f"/api/v1/jobs/{jid}/result")
        assert res.status_code == 200
        assert res.headers["content-type"] == "video/mp4"
