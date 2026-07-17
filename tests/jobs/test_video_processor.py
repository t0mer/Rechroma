from pathlib import Path

from app.core.pipeline import PipelineOptions
from app.core.video import VideoCaps
from app.jobs import processor as pm
from app.jobs.models import Job, JobStatus


def test_dispatch_routes_by_kind():
    calls = []

    def img(job):
        calls.append("img")
        return "img.png"

    def vid(job):
        calls.append("vid")
        return "vid.mp4"

    disp = pm.make_dispatch_processor(img, vid)
    disp(Job("a", JobStatus.QUEUED, PipelineOptions(), "/i/a", kind="image"))
    disp(Job("b", JobStatus.QUEUED, PipelineOptions(), "/i/b", kind="video"))
    assert calls == ["img", "vid"]


def test_video_processor_reports_progress_and_cleans_workspace(tmp_path, monkeypatch):
    reported = []
    caps = VideoCaps(30, 1080, 24, 3, 21, 18)

    class FakeVC:
        def __init__(self, *a, **k):
            pass

        def colorize_video(
            self, in_path, out_path, workspace, on_progress=None, should_cancel=None
        ):
            Path(workspace).mkdir(parents=True, exist_ok=True)
            (Path(workspace) / "marker").write_text("x")
            if on_progress:
                on_progress(0.5)
                on_progress(1.0)
            Path(out_path).write_bytes(b"\x00")

    monkeypatch.setattr(pm, "VideoColorizer", FakeVC)
    proc = pm.make_video_processor(
        tmp_path / "out",
        tmp_path / "ws",
        caps,
        report=lambda jid, f: reported.append((jid, f)),
        device="cpu",
    )
    out = proc(
        Job("v", JobStatus.QUEUED, PipelineOptions(), str(tmp_path / "in.mp4"), kind="video")
    )
    assert out.endswith("v_result.mp4")
    assert reported[-1] == ("v", 1.0)
    assert not (tmp_path / "ws" / "v").exists()  # cleaned in finally
