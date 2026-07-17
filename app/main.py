"""Composition root: build the FastAPI app wiring config -> job service -> API.

The web UI and Telegram bot are thin clients over the same job service; this
module owns the HTTP surface, lifespan (start/stop workers + retention), health,
metrics, and serving the built frontend as static assets (CLAUDE.md §4, §6).
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .api.routes import router
from .api.schemas import HealthOut
from .config import Settings, load_settings
from .core.device import resolve_device
from .core.video import VideoCaps
from .jobs.processor import (
    make_dispatch_processor,
    make_pipeline_processor,
    make_video_processor,
)
from .jobs.service import JobService, JobServiceConfig, Processor, run_retention_loop
from .jobs.store import JobStore
from .version import __version__

_FRONTEND_DIST = Path(__file__).resolve().parent / "webui" / "dist"


def video_caps_from_settings(settings: Settings) -> VideoCaps:
    """Build the video processing caps from settings."""
    return VideoCaps(
        max_seconds=settings.video_max_seconds,
        max_resolution=settings.video_max_resolution,
        max_fps=settings.video_max_fps,
        smoothing_window=settings.video_smoothing_window,
        render_factor=settings.video_render_factor,
        crf=settings.video_crf,
    )


def create_app(settings: Settings | None = None, processor: Processor | None = None) -> FastAPI:
    """Build the application. ``processor`` can be injected for tests."""
    settings = settings or load_settings()
    store = JobStore(settings.data_dir / "jobs.db")
    if processor is not None:
        proc = processor
    else:
        image_proc = make_pipeline_processor(
            settings.data_dir / "results",
            settings.device,
            settings.models_dir,
            settings.model_base_url,
        )
        video_proc = make_video_processor(
            settings.data_dir / "results",
            settings.video_workspace_dir or (settings.data_dir / "video"),
            video_caps_from_settings(settings),
            report=store.set_progress,
            device=settings.device,
            models_dir=settings.models_dir,
            base_url=settings.model_base_url,
            # `service` is bound just below; the lambda is only called at run time.
            is_cancelled=lambda jid: service.is_cancelled(jid),
        )
        proc = make_dispatch_processor(image_proc, video_proc)
    service = JobService(
        store,
        proc,
        JobServiceConfig(
            workers=settings.workers,
            rate_limit_per_hour=settings.rate_limit_per_hour,
            retention_seconds=settings.retention_hours * 3600,
        ),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if not settings.web_auth_token:
            logger.warning(
                "web_auth_token is not set — the API/UI is OPEN. Put it behind a reverse proxy "
                "or set WEB_AUTH_TOKEN."
            )
        await service.start()
        stop = False
        retention = asyncio.create_task(run_retention_loop(service, 3600, lambda: stop))
        bot_task: asyncio.Task[None] | None = None
        if settings.telegram_bot_token:
            from .telegram.bot import run_bot

            bot_task = asyncio.create_task(run_bot(settings, service))
        try:
            yield
        finally:
            stop = True
            retention.cancel()
            if bot_task is not None:
                bot_task.cancel()
            await asyncio.gather(retention, return_exceptions=True)
            if bot_task is not None:
                await asyncio.gather(bot_task, return_exceptions=True)
            await service.stop()

    app = FastAPI(
        title="Rechroma",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.service = service
    app.include_router(router)

    @app.get("/healthz", response_model=HealthOut)
    def healthz() -> HealthOut:
        counts = store.status_counts()
        return HealthOut(
            status="ok",
            version=__version__,
            device=resolve_device(settings.device).type,
            queue_depth=counts.get("queued", 0),
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        counts = store.status_counts()
        lines = [
            "# HELP rechroma_up 1 if the service is running",
            "# TYPE rechroma_up gauge",
            "rechroma_up 1",
            "# HELP rechroma_queue_depth queued jobs",
            "# TYPE rechroma_queue_depth gauge",
            f"rechroma_queue_depth {counts.get('queued', 0)}",
            "# HELP rechroma_jobs_total jobs by status",
            "# TYPE rechroma_jobs_total gauge",
        ]
        for st in ("queued", "running", "done", "failed"):
            lines.append(f'rechroma_jobs_total{{status="{st}"}} {counts.get(st, 0)}')
        return "\n".join(lines) + "\n"

    if _FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
        logger.info("serving frontend from {}", _FRONTEND_DIST)

    return app
