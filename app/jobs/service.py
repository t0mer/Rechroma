"""Async in-process job service: queue, workers, quotas, retention.

Workers run the (CPU/GPU heavy, blocking) pipeline in a thread executor so the
event loop is never starved (CLAUDE.md §4). No external queue/broker — an
``asyncio.Queue`` fed from the SQLite store. Processing is injected as a
``processor`` callable so the queue mechanics are testable without models.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .models import Job, JobStatus
from .store import JobStore

Processor = Callable[[Job], str]  # returns the result file path; runs in a worker thread


class RateLimitError(Exception):
    """Raised when a source exceeds its per-window job quota."""


@dataclass
class JobServiceConfig:
    workers: int = 1
    rate_limit_per_hour: int = 10  # per source_ref; 0 disables
    retention_seconds: float = 24 * 3600  # delete finished jobs + files after this; 0 = immediate


class JobService:
    def __init__(
        self,
        store: JobStore,
        processor: Processor,
        config: JobServiceConfig | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.store = store
        self.processor = processor
        self.config = config or JobServiceConfig()
        self._clock = clock or _monotonic_wall
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._running = False

    async def start(self) -> None:
        """Recover interrupted jobs, re-enqueue queued ones, and spin up workers."""
        recovered = self.store.recover_interrupted()
        if recovered:
            logger.warning("marked {} interrupted job(s) as failed on startup", recovered)
        for job in reversed(self.store.list_jobs(limit=1000)):
            if job.status is JobStatus.QUEUED:
                self._queue.put_nowait(job.id)
        self._running = True
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self.config.workers)]
        logger.info("job service started with {} worker(s)", self.config.workers)

    async def stop(self) -> None:
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    async def submit(
        self,
        options: object,
        input_path: str,
        source: str = "web",
        source_ref: str | None = None,
    ) -> Job:
        """Create + enqueue a job, enforcing the per-source rate limit."""
        now = self._clock()
        if self.config.rate_limit_per_hour and source_ref:
            recent = self.store.count_recent_for_source(source_ref, since=now - 3600)
            if recent >= self.config.rate_limit_per_hour:
                raise RateLimitError(f"rate limit reached ({self.config.rate_limit_per_hour}/hour)")
        job = Job(
            id=uuid.uuid4().hex,
            status=JobStatus.QUEUED,
            options=options,  # type: ignore[arg-type]
            input_path=input_path,
            source=source,
            source_ref=source_ref,
            created_at=now,
        )
        self.store.add(job)
        self._queue.put_nowait(job.id)
        return job

    async def _worker(self, index: int) -> None:
        while self._running:
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                break
            try:
                await self._process(job_id)
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001 - worker must never die
                logger.exception("worker {} crashed on job {}", index, job_id)
            finally:
                self._queue.task_done()

    async def _process(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        self.store.update(job_id, status=JobStatus.RUNNING, started_at=self._clock())
        try:
            result_path = await asyncio.to_thread(self.processor, job)
            self.store.update(
                job_id,
                status=JobStatus.DONE,
                result_path=result_path,
                finished_at=self._clock(),
            )
            logger.info("job {} done -> {}", job_id, result_path)
        except Exception as e:  # noqa: BLE001 - surface as failed job, keep serving
            logger.exception("job {} failed", job_id)
            self.store.update(
                job_id, status=JobStatus.FAILED, error=str(e), finished_at=self._clock()
            )

    def cleanup_expired(self) -> int:
        """Delete finished jobs older than the retention window plus their files."""
        cutoff = self._clock() - self.config.retention_seconds
        removed = self.store.delete_older_than(cutoff)
        for job in removed:
            for p in (job.input_path, job.result_path):
                if p:
                    Path(p).unlink(missing_ok=True)
        if removed:
            logger.info("retention: removed {} expired job(s)", len(removed))
        return len(removed)


def _monotonic_wall() -> float:
    import time

    return time.time()


async def run_retention_loop(
    service: JobService, interval_seconds: float, stop: Callable[[], bool]
) -> Awaitable[None] | None:
    """Periodically run retention cleanup until ``stop()`` returns True."""
    while not stop():
        service.cleanup_expired()
        await asyncio.sleep(interval_seconds)
    return None
