"""REST API routes under ``/api/v1`` (CLAUDE.md §6).

Every UI action maps to one of these endpoints. Auth (when configured) is applied
to the whole router via :func:`verify_token`.
"""

from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import ValidationError

from app.jobs.models import JobStatus
from app.jobs.service import RateLimitError

from .schemas import JobOptionsIn, JobOut
from .security import verify_token

router = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_token)])


@router.post("/jobs", response_model=JobOut, status_code=status.HTTP_201_CREATED)
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    preset: str = Form("full"),
    model: str = Form("artistic"),
    render_factor: int | None = Form(None),
    upscale: int | None = Form(None),
    restore_faces: bool = Form(True),
) -> JobOut:
    from .uploads import UploadError, save_validated_upload, save_validated_video, sniff_media_type

    settings = request.app.state.settings
    service = request.app.state.service

    try:
        options = JobOptionsIn(
            preset=preset,  # type: ignore[arg-type]
            model=model,  # type: ignore[arg-type]
            render_factor=render_factor,
            upscale=upscale,  # type: ignore[arg-type]
            restore_faces=restore_faces,
        )
    except ValidationError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.errors()) from e

    data = await file.read()
    kind = sniff_media_type(data)
    job_uuid = uuid4().hex
    try:
        if kind == "video":
            if not settings.video_enabled:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="video processing disabled")
            from app.main import video_caps_from_settings

            path = save_validated_video(
                data,
                settings.data_dir / "inputs",
                job_uuid,
                settings.video_max_mb * 1024 * 1024,
                caps=video_caps_from_settings(settings),
            )
        elif kind == "image":
            path = save_validated_upload(
                data,
                settings.data_dir / "inputs",
                job_uuid,
                settings.max_upload_mb * 1024 * 1024,
            )
        else:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="unsupported file type")
    except UploadError as e:
        code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if "too large" in str(e) else 400
        raise HTTPException(code, detail=str(e)) from e

    source_ref = request.client.host if request.client else None
    try:
        job = await service.submit(
            options.to_pipeline_options(),
            str(path),
            source="web",
            source_ref=source_ref,
            kind=kind,
        )
    except RateLimitError as e:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e)) from e

    return JobOut.from_job(job, service.store.queue_position(job.id))


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(request: Request, job_id: str) -> JobOut:
    service = request.app.state.service
    job = service.store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="job not found")
    return JobOut.from_job(job, service.store.queue_position(job.id))


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(request: Request, limit: int = 50, offset: int = 0) -> list[JobOut]:
    service = request.app.state.service
    jobs = service.store.list_jobs(limit=min(limit, 200), offset=offset)
    return [JobOut.from_job(j, service.store.queue_position(j.id)) for j in jobs]


@router.get("/jobs/{job_id}/result")
def get_result(request: Request, job_id: str) -> FileResponse:
    service = request.app.state.service
    job = service.store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="job not found")
    if job.status is not JobStatus.DONE or not job.result_path:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=f"job is {job.status}, no result yet")
    if job.kind == "video":
        return FileResponse(job.result_path, media_type="video/mp4", filename=f"{job_id}.mp4")
    return FileResponse(job.result_path, media_type="image/png", filename=f"{job_id}.png")
