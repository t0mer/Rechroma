"""Pydantic request/response models for the REST API (CLAUDE.md §6)."""

from typing import Literal

from pydantic import BaseModel, Field

from app.core.pipeline import PipelineOptions
from app.jobs.models import Job


class JobOptionsIn(BaseModel):
    """Job options accepted from the client (mirrors ``PipelineOptions``)."""

    preset: Literal["colorize", "restore", "full"] = "full"
    model: Literal["artistic", "stable"] = "artistic"
    render_factor: int | None = Field(default=None, ge=7, le=45)
    upscale: Literal[2, 4] | None = None
    restore_faces: bool = True

    def to_pipeline_options(self) -> PipelineOptions:
        return PipelineOptions(
            preset=self.preset,
            colorizer_model=self.model,
            render_factor=self.render_factor,
            upscale=self.upscale,
            restore_faces=self.restore_faces,
        )


class JobOut(BaseModel):
    """Job status returned to clients (no filesystem paths leaked)."""

    id: str
    status: str
    kind: str = "image"
    name: str = ""
    preset: str
    progress: float = 0.0
    queue_position: int | None = None
    error: str | None = None
    has_result: bool = False
    created_at: float = 0.0

    @classmethod
    def from_job(cls, job: Job, queue_position: int | None = None) -> "JobOut":
        return cls(
            id=job.id,
            status=str(job.status),
            kind=job.kind,
            name=job.name,
            preset=job.options.preset,
            progress=job.progress,
            queue_position=queue_position,
            error=job.error,
            has_result=job.result_path is not None,
            created_at=job.created_at,
        )


class HealthOut(BaseModel):
    status: str
    version: str
    device: str
    queue_depth: int
