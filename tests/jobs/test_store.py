from app.core.pipeline import PipelineOptions
from app.jobs.models import Job, JobStatus
from app.jobs.store import JobStore


def _store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs.db")


def _job(jid: str, status: JobStatus = JobStatus.QUEUED) -> Job:
    return Job(
        id=jid,
        status=status,
        options=PipelineOptions(preset="full", upscale=2),
        input_path=f"/tmp/{jid}.png",
        created_at=100.0,
    )


def test_add_and_get_roundtrip(tmp_path):
    store = _store(tmp_path)
    store.add(_job("a"))
    got = store.get("a")
    assert got is not None
    assert got.id == "a"
    assert got.status is JobStatus.QUEUED
    assert got.options.preset == "full"
    assert got.options.upscale == 2


def test_get_missing_returns_none(tmp_path):
    assert _store(tmp_path).get("nope") is None


def test_update_status_and_result(tmp_path):
    store = _store(tmp_path)
    store.add(_job("b"))
    store.update("b", status=JobStatus.DONE, result_path="/tmp/b_out.png", finished_at=200.0)
    got = store.get("b")
    assert got.status is JobStatus.DONE
    assert got.result_path == "/tmp/b_out.png"
    assert got.finished_at == 200.0


def test_list_orders_newest_first(tmp_path):
    store = _store(tmp_path)
    store.add(Job("x", JobStatus.QUEUED, PipelineOptions(), "/i/x", created_at=1.0))
    store.add(Job("y", JobStatus.QUEUED, PipelineOptions(), "/i/y", created_at=2.0))
    ids = [j.id for j in store.list_jobs(limit=10)]
    assert ids == ["y", "x"]


def test_queue_position_counts_earlier_queued(tmp_path):
    store = _store(tmp_path)
    store.add(Job("q1", JobStatus.QUEUED, PipelineOptions(), "/i/q1", created_at=1.0))
    store.add(Job("q2", JobStatus.QUEUED, PipelineOptions(), "/i/q2", created_at=2.0))
    assert store.queue_position("q1") == 1
    assert store.queue_position("q2") == 2


def test_persists_across_reopen(tmp_path):
    JobStore(tmp_path / "jobs.db").add(_job("c"))
    reopened = JobStore(tmp_path / "jobs.db")
    assert reopened.get("c") is not None


def test_recover_running_marks_failed(tmp_path):
    store = _store(tmp_path)
    store.add(_job("r", status=JobStatus.RUNNING))
    n = store.recover_interrupted()
    assert n == 1
    assert store.get("r").status is JobStatus.FAILED


def test_count_recent_for_source(tmp_path):
    store = _store(tmp_path)
    store.add(
        Job("s1", JobStatus.DONE, PipelineOptions(), "/i/s1", source_ref="chat9", created_at=100.0)
    )
    store.add(
        Job(
            "s2", JobStatus.QUEUED, PipelineOptions(), "/i/s2", source_ref="chat9", created_at=200.0
        )
    )
    store.add(
        Job(
            "s3", JobStatus.QUEUED, PipelineOptions(), "/i/s3", source_ref="other", created_at=200.0
        )
    )
    assert store.count_recent_for_source("chat9", since=150.0) == 1
    assert store.count_recent_for_source("chat9", since=50.0) == 2
