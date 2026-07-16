"""Default job processor: run the restoration pipeline on a job's input image.

Runs in a worker thread (see ``JobService``). Loads the input, builds the steps
for the job's options, threads the image through, and writes the result.
"""

from pathlib import Path

from PIL import Image

from app.core.pipeline import build_steps, run_pipeline

from .models import Job
from .service import Processor


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
