"""Submit a job to the shared JobService and await its completion.

Keeps the aiogram handlers thin and testable: this is plain async logic over the
job store, independent of Telegram.
"""

import asyncio

from app.core.pipeline import PipelineOptions
from app.jobs.models import Job, JobStatus
from app.jobs.service import JobService


async def process_and_wait(
    service: JobService,
    options: PipelineOptions,
    input_path: str,
    chat_id: int,
    poll_interval: float = 0.5,
    timeout: float = 900.0,
    on_status: object = None,
    kind: str = "image",
    name: str = "",
) -> Job:
    """Submit a job and poll until it is done/failed. Returns the final Job.

    ``on_status`` (optional async callable) is invoked with the Job whenever its
    status **or rounded progress** changes, so callers can update a progress
    message (useful for long video jobs).
    """
    job = await service.submit(
        options, input_path, source="telegram", source_ref=str(chat_id), kind=kind, name=name
    )
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    last_status: JobStatus | None = None
    last_pct = -1
    while True:
        current = service.store.get(job.id)
        if current is None:
            raise RuntimeError("job vanished from store")
        pct = int(current.progress * 100)
        if current.status != last_status or pct != last_pct:
            last_status, last_pct = current.status, pct
            if on_status is not None:
                await on_status(current)  # type: ignore[operator]
        if current.status in (JobStatus.DONE, JobStatus.FAILED):
            return current
        if loop.time() > deadline:
            raise TimeoutError("job did not finish in time")
        await asyncio.sleep(poll_interval)
