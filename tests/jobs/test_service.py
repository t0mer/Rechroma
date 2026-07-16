import asyncio

import pytest

from app.core.pipeline import PipelineOptions
from app.jobs.models import JobStatus
from app.jobs.service import JobService, JobServiceConfig, RateLimitError
from app.jobs.store import JobStore


def _service(tmp_path, processor, **cfg) -> JobService:
    store = JobStore(tmp_path / "jobs.db")
    clock = cfg.pop("clock", None)
    return JobService(store, processor, JobServiceConfig(**cfg), clock=clock)


async def _wait_terminal(svc, job_id, timeout=5.0):
    async def _poll():
        while True:
            j = svc.store.get(job_id)
            if j and j.status in (JobStatus.DONE, JobStatus.FAILED):
                return j
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(_poll(), timeout)


async def test_job_runs_to_done(tmp_path):
    def proc(job):
        return job.input_path + ".out"

    svc = _service(tmp_path, proc)
    await svc.start()
    job = await svc.submit(PipelineOptions(), "/tmp/a.png")
    done = await _wait_terminal(svc, job.id)
    assert done.status is JobStatus.DONE
    assert done.result_path == "/tmp/a.png.out"
    await svc.stop()


async def test_processor_error_marks_failed(tmp_path):
    def proc(job):
        raise ValueError("boom")

    svc = _service(tmp_path, proc)
    await svc.start()
    job = await svc.submit(PipelineOptions(), "/tmp/b.png")
    done = await _wait_terminal(svc, job.id)
    assert done.status is JobStatus.FAILED
    assert "boom" in done.error
    await svc.stop()


async def test_rate_limit_enforced(tmp_path):
    def proc(job):
        return "x"

    svc = _service(tmp_path, proc, rate_limit_per_hour=2, clock=lambda: 1000.0)
    await svc.start()
    await svc.submit(PipelineOptions(), "/tmp/1.png", source_ref="chat")
    await svc.submit(PipelineOptions(), "/tmp/2.png", source_ref="chat")
    with pytest.raises(RateLimitError):
        await svc.submit(PipelineOptions(), "/tmp/3.png", source_ref="chat")
    await svc.stop()


async def test_cleanup_removes_expired(tmp_path):
    out = tmp_path / "r.png"
    out.write_bytes(b"x")
    inp = tmp_path / "i.png"
    inp.write_bytes(b"y")

    def proc(job):
        return str(out)

    t = {"now": 1000.0}
    svc = _service(tmp_path, proc, retention_seconds=100.0, clock=lambda: t["now"])
    await svc.start()
    job = await svc.submit(PipelineOptions(), str(inp))
    await _wait_terminal(svc, job.id)
    t["now"] = 2000.0  # advance past retention
    removed = svc.cleanup_expired()
    assert removed == 1
    assert not out.exists()
    assert not inp.exists()
    assert svc.store.get(job.id) is None
    await svc.stop()


async def test_recover_interrupted_on_start(tmp_path):
    from app.jobs.models import Job

    store = JobStore(tmp_path / "jobs.db")
    store.add(Job("stuck", JobStatus.RUNNING, PipelineOptions(), "/i/stuck", created_at=1.0))
    svc = JobService(store, lambda job: "x")
    await svc.start()
    assert store.get("stuck").status is JobStatus.FAILED
    await svc.stop()
