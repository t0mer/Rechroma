import httpx
import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.core.engines.base import AnimateCancelled
from app.core.engines.cloud import CloudEngine


def _settings(tmp_path, **kw):
    return Settings(data_dir=tmp_path / "d", device="cpu", models_dir=tmp_path / "m", **kw)


def _image():
    return Image.fromarray(np.full((8, 8, 3), 120, np.uint8), "RGB")


def test_check_gating(tmp_path):
    off = CloudEngine(_settings(tmp_path))
    assert off.check(_settings(tmp_path))[0] is False  # disabled by default

    no_key = _settings(tmp_path, animate_cloud_enabled=True)
    assert CloudEngine(no_key).check(no_key) == (
        False,
        "Set REPLICATE_API_TOKEN to enable the cloud engine",
    )

    ok = _settings(
        tmp_path, animate_cloud_enabled=True, replicate_api_token="tok", animate_cloud_model="o/n"
    )
    assert CloudEngine(ok).check(ok) == (True, "")


def _client(handler):
    return httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.replicate.com/v1"
    )


def test_run_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.engines.cloud.time.sleep", lambda *_: None)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/predictions"):
            return httpx.Response(
                201,
                json={
                    "id": "abc",
                    "status": "processing",
                    "urls": {"get": "https://api.replicate.com/v1/predictions/abc"},
                },
            )
        if request.method == "GET" and path.endswith("/predictions/abc"):
            return httpx.Response(
                200, json={"id": "abc", "status": "succeeded", "output": "https://cdn/out.mp4"}
            )
        if request.url.host == "cdn":
            return httpx.Response(200, content=b"MP4BYTES")
        return httpx.Response(404)

    s = _settings(
        tmp_path, animate_cloud_enabled=True, replicate_api_token="tok", animate_cloud_model="o/n"
    )
    engine = CloudEngine(s)
    out = tmp_path / "out.mp4"
    seen: list[float] = []
    engine._run(_client(handler), _image(), out, seen.append, None)
    assert out.read_bytes() == b"MP4BYTES"
    assert seen and seen[-1] == 1.0


def test_run_cancel_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.engines.cloud.time.sleep", lambda *_: None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "id": "abc",
                    "status": "processing",
                    "urls": {"get": "https://api.replicate.com/v1/predictions/abc"},
                },
            )
        return httpx.Response(200, json={"id": "abc", "status": "processing"})

    s = _settings(
        tmp_path, animate_cloud_enabled=True, replicate_api_token="tok", animate_cloud_model="o/n"
    )
    engine = CloudEngine(s)
    with pytest.raises(AnimateCancelled):
        engine._run(_client(handler), _image(), tmp_path / "o.mp4", None, lambda: True)


def test_run_provider_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "bad"})

    s = _settings(
        tmp_path, animate_cloud_enabled=True, replicate_api_token="tok", animate_cloud_model="o/n"
    )
    engine = CloudEngine(s)
    from app.core.engines.base import EngineError

    with pytest.raises(EngineError):
        engine._run(_client(handler), _image(), tmp_path / "o.mp4", None, None)
