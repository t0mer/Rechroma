import numpy as np
from PIL import Image

from app.core.pipeline import PipelineOptions, run_pipeline, step_names


def _img() -> Image.Image:
    return Image.fromarray(np.full((8, 8, 3), 100, np.uint8), "RGB")


def test_preset_step_order_colorize():
    assert step_names(PipelineOptions(preset="colorize")) == ["colorize"]


def test_preset_step_order_restore():
    names = step_names(PipelineOptions(preset="restore", upscale=2))
    assert names == ["restore_faces", "upscale"]  # no colorize for already-colour photos


def test_preset_step_order_full():
    names = step_names(PipelineOptions(preset="full", upscale=2))
    assert names == ["restore_faces", "colorize", "upscale"]  # restore before colorize


def test_full_without_upscale_has_no_upscale_step():
    assert step_names(PipelineOptions(preset="full", upscale=None)) == ["restore_faces", "colorize"]


def test_run_pipeline_applies_steps_in_order():
    calls = []

    class FakeStep:
        def __init__(self, name):
            self.name = name

        def process(self, image):
            calls.append(self.name)
            return image

    out = run_pipeline([FakeStep("a"), FakeStep("b")], _img())
    assert calls == ["a", "b"]
    assert out.size == (8, 8)
