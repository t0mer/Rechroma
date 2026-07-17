"""SQLite-backed job store (WAL mode), the single source of truth for job state.

Uses stdlib ``sqlite3`` — no external database (CLAUDE.md §2). A short-lived
connection per operation keeps this safe to call from the async event loop and
from worker threads without sharing a connection.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .models import Job, JobStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    options      TEXT NOT NULL,
    input_path   TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'web',
    source_ref   TEXT,
    kind         TEXT NOT NULL DEFAULT 'image',
    progress     REAL NOT NULL DEFAULT 0,
    name         TEXT NOT NULL DEFAULT '',
    result_path  TEXT,
    error        TEXT,
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs (created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_source_ref ON jobs (source_ref);
"""


class JobStore:
    """Persistent job records with the queue's derived views."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            # Migrate pre-v2 databases created before the video columns existed.
            cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
            if "kind" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'image'")
            if "progress" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN progress REAL NOT NULL DEFAULT 0")
            if "name" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN name TEXT NOT NULL DEFAULT ''")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            status=JobStatus(row["status"]),
            options=Job.options_from_json(row["options"]),
            input_path=row["input_path"],
            source=row["source"],
            source_ref=row["source_ref"],
            kind=row["kind"],
            progress=row["progress"],
            name=row["name"],
            result_path=row["result_path"],
            error=row["error"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def add(self, job: Job) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO jobs
                   (id, status, options, input_path, source, source_ref, kind, progress, name,
                    result_path, error, created_at, started_at, finished_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    job.id,
                    job.status.value,
                    job.options_json(),
                    job.input_path,
                    job.source,
                    job.source_ref,
                    job.kind,
                    job.progress,
                    job.name,
                    job.result_path,
                    job.error,
                    job.created_at,
                    job.started_at,
                    job.finished_at,
                ),
            )

    def get(self, job_id: str) -> Job | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def update(self, job_id: str, **fields: object) -> None:
        if not fields:
            return
        if "status" in fields and isinstance(fields["status"], JobStatus):
            fields["status"] = fields["status"].value
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def set_progress(self, job_id: str, value: float) -> None:
        """Update a job's progress fraction (0.0–1.0). Cheap; called frequently."""
        with self._conn() as conn:
            conn.execute("UPDATE jobs SET progress=? WHERE id=?", (float(value), job_id))

    def list_jobs(self, limit: int = 50, offset: int = 0) -> list[Job]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def queue_position(self, job_id: str) -> int | None:
        """1-based position of a queued job among all queued jobs, oldest first."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT created_at, status FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if row is None or row["status"] != JobStatus.QUEUED.value:
                return None
            (ahead,) = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status=? AND created_at<=?",
                (JobStatus.QUEUED.value, row["created_at"]),
            ).fetchone()
        return int(ahead)

    def recover_interrupted(self) -> int:
        """Mark any ``running`` jobs as ``failed`` (called at startup after a crash)."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status=?, error=? WHERE status=?",
                (JobStatus.FAILED.value, "interrupted by restart", JobStatus.RUNNING.value),
            )
            return cur.rowcount

    def status_counts(self) -> dict[str, int]:
        """Number of jobs per status (for healthz / metrics)."""
        with self._conn() as conn:
            rows = conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
        return {row[0]: int(row[1]) for row in rows}

    def count_recent_for_source(self, source_ref: str, since: float) -> int:
        """Number of jobs for ``source_ref`` created at or after ``since`` (rate limiting)."""
        with self._conn() as conn:
            (n,) = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE source_ref=? AND created_at>=?",
                (source_ref, since),
            ).fetchone()
        return int(n)

    def delete(self, job_id: str) -> None:
        """Delete a single job record."""
        with self._conn() as conn:
            conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def delete_older_than(self, cutoff: float) -> list[Job]:
        """Delete jobs finished before ``cutoff``; return them so callers can unlink files."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE finished_at IS NOT NULL AND finished_at < ?",
                (cutoff,),
            ).fetchall()
            jobs = [self._row_to_job(r) for r in rows]
            conn.execute(
                "DELETE FROM jobs WHERE finished_at IS NOT NULL AND finished_at < ?", (cutoff,)
            )
        return jobs
