"""FastAPI service for asynchronous TRL assessments."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from trl_scout.config import settings
from trl_scout.pipeline.run import assess
from trl_scout.schemas import Assessment, ProgressEvent

logger = logging.getLogger("trl_scout.api")

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
_CONTENT_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    },
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
}
_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class AssessmentInput(BaseModel):
    """JSON source submitted to the assessment endpoint."""

    model_config = ConfigDict(extra="forbid")

    arxiv_id: str | None = Field(default=None, description="arXiv identifier.")
    url: str | None = Field(default=None, description="HTTP(S) source URL.")
    text: str | None = Field(default=None, description="Raw document text.")


@dataclass
class RunRecord:
    """In-memory status and event state for one run."""

    run_id: str
    status: str = "queued"
    progress: float = 0
    assessment: Assessment | None = None
    error: str | None = None
    events: list[ProgressEvent] = field(default_factory=list)
    subscribers: list[asyncio.Queue[ProgressEvent | None]] = field(default_factory=list)


app = FastAPI(title="TRL-Scout", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.runs = {}
app.state.run_tasks = {}
app.state.run_semaphore = asyncio.Semaphore(2)
app.state.settings = settings
app.state.llm = None


def _record(run_id: str) -> RunRecord:
    """Look up a run or raise a public 404."""
    record = app.state.runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Assessment run was not found.")
    return record


def _safe_filename(filename: str | None) -> str:
    """Return a safe basename for storage below the run directory."""
    name = Path(filename or "upload.bin").name
    name = _FILENAME_RE.sub("_", name).strip("._")
    if not name:
        raise HTTPException(status_code=400, detail="Upload filename is invalid.")
    return name


async def _read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an upload with a hard size limit."""
    size = 0
    chunks: list[bytes] = []
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds the {max_bytes // (1024 * 1024)} MB limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_upload(upload: UploadFile) -> tuple[str, str]:
    """Validate extension and declared content type."""
    filename = _safe_filename(upload.filename)
    extension = Path(filename).suffix.lower()
    if extension not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Use PDF, DOCX, TXT, or Markdown.",
        )
    content_type = (upload.content_type or "application/octet-stream").lower()
    if content_type not in _CONTENT_TYPES[extension]:
        raise HTTPException(
            status_code=415,
            detail=f"Content type '{content_type}' does not match the file extension.",
        )
    return filename, extension


def _source_from_input(payload: AssessmentInput) -> str:
    """Validate and select exactly one JSON source."""
    values = [payload.arxiv_id, payload.url, payload.text]
    if sum(value is not None and bool(value.strip()) for value in values) != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one non-empty arxiv_id, url, or text field.",
        )
    return next(value for value in values if value is not None and value.strip())


def _publish(record: RunRecord, event: ProgressEvent) -> None:
    """Store and broadcast a progress event from any worker thread."""
    record.events.append(event)
    record.progress = event.pct
    for queue in list(record.subscribers):
        queue.put_nowait(event)


async def _run_assessment(record: RunRecord, source: str | Path) -> None:
    """Run assessment off the event loop and finalize status."""
    async with app.state.run_semaphore:
        record.status = "running"
        loop = asyncio.get_running_loop()

        def progress(event: ProgressEvent) -> None:
            loop.call_soon_threadsafe(_publish, record, event)

        try:
            record.assessment = await asyncio.to_thread(
                assess,
                source,
                settings=app.state.settings,
                llm=app.state.llm,
                progress=progress,
                run_id=record.run_id,
            )
            record.status = "done"
            _publish(
                record,
                ProgressEvent(step="run", status="done", detail="Assessment complete.", pct=100),
            )
        except Exception:
            logger.exception("Assessment run %s failed", record.run_id)
            record.status = "failed"
            record.error = "Assessment could not be completed."
            _publish(
                record,
                ProgressEvent(
                    step="run",
                    status="failed",
                    detail=record.error,
                    pct=record.progress,
                ),
            )
        finally:
            for queue in list(record.subscribers):
                queue.put_nowait(None)


def _enqueue(record: RunRecord, source: str | Path) -> None:
    """Schedule a run and retain its task for lifecycle inspection."""
    task = asyncio.create_task(_run_assessment(record, source))
    app.state.run_tasks[record.run_id] = task


@app.post("/v1/assessments", status_code=202)
async def create_assessment(request: Request) -> dict[str, str]:
    """Queue an assessment from an upload or JSON source."""
    content_type = request.headers.get("content-type", "").lower()
    run_id = uuid.uuid4().hex
    record = RunRecord(run_id=run_id)
    app.state.runs[run_id] = record
    run_dir = Path(app.state.settings.data_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if not isinstance(upload, UploadFile) and not (
                hasattr(upload, "filename") and hasattr(upload, "read")
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Multipart request must include a file.",
                )
            filename, _ = _validate_upload(upload)
            content = await _read_upload(upload, app.state.settings.max_upload_mb * 1024 * 1024)
            destination = run_dir / filename
            destination.write_bytes(content)
            _enqueue(record, destination)
        elif content_type.startswith("application/json"):
            try:
                payload = AssessmentInput.model_validate(await request.json())
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON request: {exc}",
                ) from exc
            source = _source_from_input(payload)
            if payload.text is not None and len(source.encode("utf-8")) > (
                app.state.settings.max_upload_mb * 1024 * 1024
            ):
                raise HTTPException(
                    status_code=413,
                    detail=f"Raw text exceeds the {app.state.settings.max_upload_mb} MB limit.",
                )
            _enqueue(record, source)
        else:
            raise HTTPException(
                status_code=415,
                detail=(
                    "Use multipart/form-data with a file or application/json with "
                    "arxiv_id, url, or text."
                ),
            )
    except HTTPException:
        app.state.runs.pop(run_id, None)
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return {"run_id": run_id}


@app.get("/v1/assessments/{run_id}")
async def get_assessment(run_id: str) -> dict[str, Any]:
    """Return status, progress, and the completed assessment if available."""
    record = _record(run_id)
    response: dict[str, Any] = {
        "run_id": run_id,
        "status": record.status,
        "progress": record.progress,
    }
    if record.error:
        response["error"] = record.error
    if record.assessment is not None:
        response["assessment"] = record.assessment.model_dump(mode="json")
    return response


@app.get("/v1/assessments/{run_id}/events")
async def assessment_events(run_id: str) -> StreamingResponse:
    """Stream progress events as Server-Sent Events."""
    record = _record(run_id)
    queue: asyncio.Queue[ProgressEvent | None] = asyncio.Queue()
    for event in record.events:
        queue.put_nowait(event)
    if record.status in {"done", "failed"}:
        queue.put_nowait(None)
    else:
        record.subscribers.append(queue)

    async def stream() -> Any:
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {event.model_dump_json()}\n\n"
        finally:
            if queue in record.subscribers:
                record.subscribers.remove(queue)

    return StreamingResponse(stream(), media_type="text/event-stream")


def _artifact(run_id: str, suffix: str) -> FileResponse:
    """Return a run artifact while preventing path traversal."""
    record = _record(run_id)
    if record.status != "done":
        raise HTTPException(status_code=409, detail="Assessment is not complete.")
    path = Path(app.state.settings.data_dir) / "runs" / run_id / suffix
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{suffix} is not available.")
    return FileResponse(path)


@app.get("/v1/assessments/{run_id}/report.pdf")
async def report_pdf(run_id: str) -> FileResponse:
    """Download the generated PDF report."""
    return _artifact(run_id, "report.pdf")


@app.get("/v1/assessments/{run_id}/report.html")
async def report_html(run_id: str) -> FileResponse:
    """Download the generated HTML report."""
    return _artifact(run_id, "report.html")


@app.get("/v1/assessments/{run_id}/assessment.json")
async def assessment_json(run_id: str) -> FileResponse:
    """Download the completed assessment JSON."""
    return _artifact(run_id, "assessment.json")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok"}
