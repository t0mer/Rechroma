"""Pydantic request/response models for the REST API (CLAUDE.md §6)."""

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from app.core.pipeline import PipelineOptions
from app.jobs.models import Job

if TYPE_CHECKING:
    from app.core.engines import EngineInfo


class JobOptionsIn(BaseModel):
    """Job options accepted from the client (mirrors ``PipelineOptions``)."""

    preset: Literal["colorize", "restore", "full", "animate"] = "full"
    model: Literal["artistic", "stable"] = "artistic"
    render_factor: int | None = Field(default=None, ge=7, le=45)
    upscale: Literal[2, 4] | None = None
    restore_faces: bool = True
    engine: Literal["tpsmm", "diffusion", "cloud"] | None = None  # animate backend

    def to_pipeline_options(self) -> PipelineOptions:
        return PipelineOptions(
            preset=self.preset,
            colorizer_model=self.model,
            render_factor=self.render_factor,
            upscale=self.upscale,
            restore_faces=self.restore_faces,
            animate_engine=self.engine,
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


class EngineOut(BaseModel):
    """One animate engine's availability, for the UI engine selector."""

    name: str
    label: str
    requires_gpu: bool
    requires_key: bool
    available: bool
    reason: str
    notes: str

    @classmethod
    def from_info(cls, info: "EngineInfo") -> "EngineOut":
        return cls(
            name=info.name,
            label=info.label,
            requires_gpu=info.requires_gpu,
            requires_key=info.requires_key,
            available=info.available,
            reason=info.reason,
            notes=info.notes,
        )


class HealthOut(BaseModel):
    status: str
    version: str
    device: str
    queue_depth: int
