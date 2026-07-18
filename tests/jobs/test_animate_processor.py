from pathlib import Path

from app.core.pipeline import PipelineOptions
from app.jobs import processor as pm
from app.jobs.models import Job, JobStatus


def test_dispatch_routes_animate():
    calls = []
    disp = pm.make_dispatch_processor(
        lambda job: (calls.append("img"), "i")[1],
        lambda job: (calls.append("vid"), "v")[1],
        lambda job: (calls.append("ani"), "a")[1],
    )
    disp(Job("a", JobStatus.QUEUED, PipelineOptions(), "/i/a", kind="image"))
    disp(Job("v", JobStatus.QUEUED, PipelineOptions(), "/i/v", kind="video"))
    disp(Job("n", JobStatus.QUEUED, PipelineOptions(), "/i/n", kind="animate"))
    assert calls == ["img", "vid", "ani"]


def test_animate_processor_reports_and_cleans(tmp_path, monkeypatch):
    reported = []

    class FakeAnimator:
        def __init__(self, *a, **k):
            pass

        def animate(self, image, out_path, workspace, on_progress=None, should_cancel=None):
            Path(workspace).mkdir(parents=True, exist_ok=True)
            (Path(workspace) / "marker").write_text("x")
            if on_progress:
                on_progress(0.5)
                on_progress(1.0)
            Path(out_path).write_bytes(b"\x00")

    monkeypatch.setattr(pm, "FaceAnimator", FakeAnimator)
    monkeypatch.setattr(pm, "_load_animate_source", lambda p: object())
    proc = pm.make_animate_processor(
        tmp_path / "out",
        tmp_path / "ws",
        driver_path=tmp_path / "drv.mp4",
        report=lambda jid, f: reported.append((jid, f)),
        device="cpu",
    )
    out = proc(
        Job("n", JobStatus.QUEUED, PipelineOptions(), str(tmp_path / "in.png"), kind="animate")
    )
    assert out.endswith("n_result.mp4")
    assert reported[-1] == ("n", 1.0)
    assert not (tmp_path / "ws" / "n").exists()  # workspace cleaned
