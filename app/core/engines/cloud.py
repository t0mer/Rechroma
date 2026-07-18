"""Cloud animate engine: hosted image-to-video via Replicate.

Opt-in and off by default. When enabled and a token is configured, the uploaded
photo is sent to Replicate (a third party) as a data URI, a prediction is polled
to completion, and the resulting mp4 is downloaded. The token is read from the
environment only (``REPLICATE_API_TOKEN``); it is never logged or returned by the
API. This is the only engine that transmits the user's photo off-box, which the
UI surfaces as an explicit privacy note.
"""

import base64
import io
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from PIL import Image

from app.config import Settings

from .base import AnimateCancelled, AnimateEngine, EngineError

_API_BASE = "https://api.replicate.com/v1"
_POLL_INTERVAL = 2.0  # seconds between prediction polls
_TIMEOUT = 30.0  # per-request HTTP timeout


class CloudEngine(AnimateEngine):
    name = "cloud"
    label = "Cloud (Replicate)"
    requires_gpu = False
    requires_key = True
    notes = "Best quality. Sends your photo to Replicate (a third party). Pay-per-use."

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def check(self, settings: Settings) -> tuple[bool, str]:
        if not settings.animate_cloud_enabled:
            return False, "Cloud engine is disabled"
        if not settings.replicate_api_token:
            return False, "Set REPLICATE_API_TOKEN to enable the cloud engine"
        if not settings.animate_cloud_model:
            return False, "Set animate_cloud_model (a Replicate model, e.g. owner/name)"
        return True, ""

    def animate(
        self,
        image: Image.Image,
        out_path: Path,
        workspace: Path,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        token = self.settings.replicate_api_token or ""
        client = httpx.Client(
            base_url=_API_BASE,
            headers={"Authorization": f"Bearer {token}"},
            timeout=_TIMEOUT,
        )
        try:
            self._run(client, image, out_path, on_progress, should_cancel)
        finally:
            client.close()

    # --- internals (client injected for testability) ----------------------
    def _run(
        self,
        client: httpx.Client,
        image: Image.Image,
        out_path: Path,
        on_progress: Callable[[float], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> None:
        def report(f: float) -> None:
            if on_progress:
                on_progress(max(0.0, min(1.0, f)))

        def cancelled() -> bool:
            return bool(should_cancel and should_cancel())

        model = self.settings.animate_cloud_model
        payload = {"input": self._build_input(image)}
        report(0.02)
        resp = client.post(f"/models/{model}/predictions", json=payload)
        if resp.status_code >= 400:
            raise EngineError(f"cloud provider rejected the request ({resp.status_code})")
        pred = resp.json()
        poll_url = pred.get("urls", {}).get("get")
        pred_id = pred.get("id")
        report(0.1)

        elapsed = 0.0
        while pred.get("status") not in ("succeeded", "failed", "canceled"):
            if cancelled():
                if pred_id:
                    with httpx_suppressed():
                        client.post(f"/predictions/{pred_id}/cancel")
                raise AnimateCancelled()
            time.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL
            report(min(0.85, 0.1 + elapsed / 120.0))  # coarse ramp; no true % from API
            pred = client.get(poll_url).json() if poll_url else pred

        if pred.get("status") != "succeeded":
            detail = pred.get("error") or pred.get("status")
            raise EngineError(f"cloud animation failed: {detail}")

        url = _first_output_url(pred.get("output"))
        if not url:
            raise EngineError("cloud provider returned no video")
        self._download(client, url, out_path)
        report(1.0)

    def _build_input(self, image: Image.Image) -> dict[str, Any]:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        payload: dict[str, Any] = {"image": data_uri}
        if self.settings.animate_cloud_prompt:
            payload["prompt"] = self.settings.animate_cloud_prompt
        return payload

    def _download(self, client: httpx.Client, url: str, out_path: Path) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with client.stream("GET", url) as r:
            if r.status_code >= 400:
                raise EngineError(f"could not download cloud result ({r.status_code})")
            with open(out_path, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)


def _first_output_url(output: Any) -> str | None:
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output and isinstance(output[0], str):
        return output[0]
    return None


class httpx_suppressed:
    """Best-effort context: swallow HTTP errors from a fire-and-forget cancel."""

    def __enter__(self) -> "httpx_suppressed":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc is not None:
            logger.debug("ignoring cloud cancel error: {}", exc)
        return True
