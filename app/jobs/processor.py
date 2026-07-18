"""Default job processor: run the restoration pipeline on a job's input image.

Runs in a worker thread (see ``JobService``). Loads the input, builds the steps
for the job's options, threads the image through, and writes the result.
"""

import shutil
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.core.engines import AnimateCancelled, build_engine, resolve_engine_name
from app.core.pipeline import build_steps, run_pipeline
from app.core.video import VideoCancelled, VideoCaps, VideoColorizer

from .models import Job
from .service import JobCancelled, Processor


def make_pipeline_processor(
    output_dir: Path,
    device: str = "auto",
    models_dir: Path = Path("/data/models"),
    base_url: str | None = None,
) -> Processor:
    """Build a ``Processor`` that runs the pipeline and saves the result as PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def process(job: Job) -> str:
        with Image.open(job.input_path) as im:
            image = im.convert("RGB")
        steps = build_steps(job.options, device, models_dir, base_url)
        result = run_pipeline(steps, image)
        out_path = output_dir / f"{job.id}_result.png"
        result.save(out_path)
        return str(out_path)

    return process


def _never_cancel(_id: str) -> bool:
    return False


def make_video_processor(
    output_dir: Path,
    workspace_dir: Path,
    caps: VideoCaps,
    report: Callable[[str, float], None],
    device: str = "auto",
    models_dir: Path = Path("/data/models"),
    base_url: str | None = None,
    is_cancelled: Callable[[str], bool] = _never_cancel,
) -> Processor:
    """Build a ``Processor`` that colorizes a video and reports progress.

    The per-job frame workspace under ``workspace_dir`` is always removed in a
    ``finally``. ``report(job_id, fraction)`` writes progress to the store, and
    ``is_cancelled(job_id)`` lets the frame loop abort a cancelled job.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def process(job: Job) -> str:
        vc = VideoColorizer(
            model=job.options.colorizer_model,
            device=device,
            models_dir=models_dir,
            base_url=base_url,
            caps=caps,
        )
        ws = Path(workspace_dir) / job.id
        out_path = output_dir / f"{job.id}_result.mp4"
        try:
            vc.colorize_video(
                Path(job.input_path),
                out_path,
                ws,
                on_progress=lambda f: report(job.id, f),
                should_cancel=lambda: is_cancelled(job.id),
            )
        except VideoCancelled as e:
            raise JobCancelled() from e
        finally:
            shutil.rmtree(ws, ignore_errors=True)
        return str(out_path)

    return process


def _load_animate_source(input_path: str) -> Image.Image:
    with Image.open(input_path) as im:
        return im.convert("RGB")


def make_animate_processor(
    output_dir: Path,
    workspace_dir: Path,
    settings: Settings,
    report: Callable[[str, float], None],
    is_cancelled: Callable[[str], bool] = _never_cancel,
) -> Processor:
    """Build a ``Processor`` that animates a portrait into an mp4 (living portrait).

    The engine (tpsmm / diffusion / cloud) is selected per job from the job's
    options, falling back to the configured default. The per-job workspace is
    always removed in a ``finally``; ``report`` writes progress and
    ``is_cancelled`` lets a running job abort.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def process(job: Job) -> str:
        engine_name = resolve_engine_name(job.options.animate_engine, settings)
        engine = build_engine(engine_name, settings)
        ws = Path(workspace_dir) / job.id
        out_path = output_dir / f"{job.id}_result.mp4"
        try:
            engine.animate(
                _load_animate_source(job.input_path),
                out_path,
                ws,
                on_progress=lambda f: report(job.id, f),
                should_cancel=lambda: is_cancelled(job.id),
            )
        except AnimateCancelled as e:
            raise JobCancelled() from e
        finally:
            shutil.rmtree(ws, ignore_errors=True)
        return str(out_path)

    return process


def make_dispatch_processor(
    image_proc: Processor,
    video_proc: Processor,
    animate_proc: Processor | None = None,
) -> Processor:
    """Route a job to the image, video, or animate processor by ``job.kind``."""

    def process(job: Job) -> str:
        if job.kind == "video":
            return video_proc(job)
        if job.kind == "animate":
            if animate_proc is None:
                raise RuntimeError("animate processor not configured")
            return animate_proc(job)
        return image_proc(job)

    return process
