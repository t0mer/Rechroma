import asyncio
import time
from pathlib import Path

from app.core.pipeline import PipelineOptions
from app.jobs.models import Job, JobStatus
from app.jobs.service import JobCancelled, JobService, JobServiceConfig
from app.jobs.store import JobStore


def _svc(tmp_path, processor) -> JobService:
    store = JobStore(tmp_path / "j.db")
    return JobService(store, processor, JobServiceConfig(rate_limit_per_hour=0))


def _mkfile(p) -> str:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return str(p)


async def _wait(cond, timeout=5.0):
    end = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < end:
        if cond():
            return True
        await asyncio.sleep(0.02)
    return False


async def test_cancel_queued_removes(tmp_path):
    # Workers not started -> the job stays queued.
    svc = _svc(tmp_path, lambda job: "out")
    inp = _mkfile(tmp_path / "in.png")
    job = await svc.submit(PipelineOptions(), inp)
    assert svc.store.get(job.id) is not None
    assert svc.request_cancel(job.id) == "removed"
    assert svc.store.get(job.id) is None
    assert not Path(inp).exists()


async def test_cancel_terminal_removes_files(tmp_path):
    svc = _svc(tmp_path, lambda job: "out")
    inp = _mkfile(tmp_path / "in.png")
    out = _mkfile(tmp_path / "out.png")
    svc.store.add(Job("d", JobStatus.DONE, PipelineOptions(), inp, result_path=out, created_at=1.0))
    assert svc.request_cancel("d") == "removed"
    assert svc.store.get("d") is None
    assert not Path(inp).exists()
    assert not Path(out).exists()


async def test_cancel_unknown_returns_none(tmp_path):
    svc = _svc(tmp_path, lambda job: "out")
    assert svc.request_cancel("nope") is None


async def test_cancel_running_cooperative_aborts(tmp_path):
    def proc(job):
        for _ in range(1000):
            if svc.is_cancelled(job.id):
                raise JobCancelled()
            time.sleep(0.01)
        return "out"

    svc = _svc(tmp_path, proc)
    await svc.start()
    inp = _mkfile(tmp_path / "in.mp4")
    job = await svc.submit(PipelineOptions(), inp, kind="video")
    assert await _wait(lambda: (j := svc.store.get(job.id)) and j.status is JobStatus.RUNNING)
    assert svc.request_cancel(job.id) == "cancelling"
    assert await _wait(lambda: svc.store.get(job.id) is None)
    assert not Path(inp).exists()
    await svc.stop()


async def test_cancel_running_image_discards_result(tmp_path):
    def proc(job):
        time.sleep(0.3)  # uninterruptible pass
        out = tmp_path / "res.png"
        out.write_bytes(b"r")
        return str(out)

    svc = _svc(tmp_path, proc)
    await svc.start()
    inp = _mkfile(tmp_path / "in.png")
    job = await svc.submit(PipelineOptions(), inp, kind="image")
    assert await _wait(lambda: (j := svc.store.get(job.id)) and j.status is JobStatus.RUNNING)
    svc.request_cancel(job.id)
    assert await _wait(lambda: svc.store.get(job.id) is None)
    assert not (tmp_path / "res.png").exists()  # result discarded
    assert not Path(inp).exists()
    await svc.stop()
