"""Job records and status for the processing queue (CLAUDE.md §4 Job service)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Literal

from app.core.pipeline import PipelineOptions

JobKind = Literal["image", "video"]


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    """A single processing request, persisted in SQLite."""

    id: str
    status: JobStatus
    options: PipelineOptions
    input_path: str
    source: str = "web"  # web | telegram
    source_ref: str | None = None  # e.g. telegram chat id
    kind: JobKind = "image"
    progress: float = 0.0
    name: str = ""  # original upload filename (for display after a page refresh)
    result_path: str | None = None
    error: str | None = None
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    extra: dict = field(default_factory=dict)

    def options_json(self) -> str:
        return json.dumps(asdict(self.options))

    @staticmethod
    def options_from_json(raw: str) -> PipelineOptions:
        data = json.loads(raw)
        return PipelineOptions(**data)
