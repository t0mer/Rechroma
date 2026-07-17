import io
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.uploads import UploadError, save_validated_video, sniff_media_type
from app.config import Settings
from app.core.video import VideoCaps
from app.jobs.models import Job
from app.jobs.processor import make_dispatch_processor
from app.main import create_app

ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _clip_bytes(tmp_path, seconds=1, fps=8):
    p = tmp_path / "c.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=64x48:rate={fps}:duration={seconds}",
            "-pix_fmt",
            "yuv420p",
            str(p),
        ],
        check=True,
        capture_output=True,
    )
    return p.read_bytes()


def _stub_video_processor(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def proc(job: Job) -> str:
        out = out_dir / f"{job.id}_result.mp4"
        out.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        return str(out)

    return proc


def _client(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", device="cpu", models_dir=tmp_path / "m")
    vproc = _stub_video_processor(tmp_path / "data" / "results")
    app = create_app(settings, processor=make_dispatch_processor(lambda job: "", vproc))
    return TestClient(app)


@ffmpeg
def test_submit_video_creates_video_job(tmp_path):
    with _client(tmp_path) as client:
        r = client.post(
            "/api/v1/jobs",
            files={"file": ("c.mp4", _clip_bytes(tmp_path), "video/mp4")},
            data={"preset": "colorize"},
        )
        assert r.status_code == 201, r.text
        assert r.json()["kind"] == "video"
        jid = r.json()["id"]
        for _ in range(200):
            j = client.get(f"/api/v1/jobs/{jid}").json()
            if j["status"] in ("done", "failed"):
                break
            time.sleep(0.02)
        assert j["status"] == "done"
        res = client.get(f"/api/v1/jobs/{jid}/result")
        assert res.status_code == 200
        assert res.headers["content-type"] == "video/mp4"


def test_image_still_works(tmp_path):
    settings = Settings(data_dir=tmp_path / "d", device="cpu", models_dir=tmp_path / "m")

    def iproc(job):
        out = Path(settings.data_dir) / "results" / f"{job.id}_result.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.open(job.input_path).save(out)
        return str(out)

    app = create_app(settings, processor=make_dispatch_processor(iproc, lambda j: ""))
    with TestClient(app) as client:
        buf = io.BytesIO()
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buf, "PNG")
        r = client.post("/api/v1/jobs", files={"file": ("i.png", buf.getvalue(), "image/png")})
        assert r.status_code == 201
        assert r.json()["kind"] == "image"


@ffmpeg
def test_sniff_and_cap_reject(tmp_path):
    data = _clip_bytes(tmp_path, seconds=2, fps=8)
    assert sniff_media_type(data) == "video"
    tight = VideoCaps(
        max_seconds=1,
        max_resolution=1080,
        max_fps=24,
        smoothing_window=1,
        render_factor=21,
        crf=18,
    )
    with pytest.raises(UploadError, match="too long"):
        save_validated_video(data, tmp_path / "in", "j", max_bytes=10_000_000, caps=tight)


def test_sniff_non_media_is_none():
    assert sniff_media_type(b"not media at all") is None
