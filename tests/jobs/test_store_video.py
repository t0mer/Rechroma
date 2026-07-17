import sqlite3

from app.core.pipeline import PipelineOptions
from app.jobs.models import Job, JobStatus
from app.jobs.store import JobStore


def _job(jid, kind="video"):
    return Job(
        jid, JobStatus.QUEUED, PipelineOptions(), f"/i/{jid}",
        kind=kind, name=f"{jid}.mp4", created_at=1.0,
    )


def test_kind_progress_and_name_persist(tmp_path):
    store = JobStore(tmp_path / "j.db")
    store.add(_job("v1"))
    got = store.get("v1")
    assert got.kind == "video"
    assert got.progress == 0.0
    assert got.name == "v1.mp4"
    store.set_progress("v1", 0.5)
    assert store.get("v1").progress == 0.5


def test_migrates_old_schema(tmp_path):
    # Simulate a pre-v2 DB without the kind/progress columns.
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, options TEXT NOT NULL,"
            " input_path TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'web', source_ref TEXT,"
            " result_path TEXT, error TEXT, created_at REAL NOT NULL, started_at REAL,"
            " finished_at REAL)"
        )
        conn.execute(
            "INSERT INTO jobs (id,status,options,input_path,source,created_at) VALUES"
            " ('old','done','{\"preset\":\"full\"}','/i/old','web',1.0)"
        )
    store = JobStore(db)  # init must add the missing columns
    got = store.get("old")
    assert got is not None
    assert got.kind == "image"  # default
    assert got.progress == 0.0
    assert got.name == ""  # default
