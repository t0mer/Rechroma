import numpy as np
from PIL import Image

from app.core.pipeline import PipelineOptions
from app.jobs import processor as processor_mod
from app.jobs.models import Job, JobStatus


def test_processor_runs_steps_and_saves(tmp_path, monkeypatch):
    src = tmp_path / "in.png"
    Image.fromarray(np.full((10, 10, 3), 50, np.uint8)).save(src)

    class DoubleWidthStep:
        name = "fake"

        def process(self, image):
            return image.resize((image.width * 2, image.height))

    monkeypatch.setattr(processor_mod, "build_steps", lambda *a, **k: [DoubleWidthStep()])

    proc = processor_mod.make_pipeline_processor(tmp_path / "out")
    job = Job("j1", JobStatus.QUEUED, PipelineOptions(), str(src))
    out_path = proc(job)

    assert out_path.endswith("j1_result.png")
    assert Image.open(out_path).size == (20, 10)
