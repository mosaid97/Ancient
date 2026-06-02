"""Upload router (Track E, Feature 1).

``POST /api/upload`` accepts a multipart file plus the upload options
(document type, OCR toggle, OCR model + optional custom credentials,
language, tier), saves the file under ``staging/<job_id>/``, creates an
``UPLOAD_JOB`` node, and spawns ``scripts/run_upload_pipeline.py`` as a
detached subprocess (the established "long jobs are scripts runners" model).

The subprocess runs ingest -> preprocess -> OCR -> fuse -> layout, updating
the ``UPLOAD_JOB`` node as it goes, then sets ``DOCUMENT.status =
'awaiting_review'``. The UI polls ``GET /api/upload/status/{job_id}``.

Custom OCR credentials are passed to the subprocess via environment
variables (``CUSTOM_OCR_BASE_URL`` / ``CUSTOM_OCR_API_KEY`` /
``CUSTOM_OCR_MODEL``) and are NEVER written to the job node or to disk.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from neo4j import Driver

from apps.backend.api.deps import get_driver
from apps.backend.pipeline import upload_job

log = logging.getLogger(__name__)
router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STAGING_ROOT = _REPO_ROOT / "staging"
_RUNNER = _REPO_ROOT / "scripts" / "run_upload_pipeline.py"

_ALLOWED_OCR_MODELS = {"qwen", "deepseek", "custom"}
_ALLOWED_TIERS = {"primary", "secondary"}
_ALLOWED_LANGUAGES = {"chinese", "japanese", "english", "arabic"}

# Upload guards: cap body size (env-overridable) and bound the number of
# concurrently-running upload jobs so a burst can't spawn unbounded heavy
# OCR subprocesses and OOM the host.
_MAX_UPLOAD_BYTES = int(os.getenv("UPLOAD_MAX_BYTES", str(512 * 1024 * 1024)))  # 512 MiB
_MAX_RUNNING_JOBS = int(os.getenv("UPLOAD_MAX_RUNNING_JOBS", "3"))
_CHUNK_BYTES = 1024 * 1024


@router.post("")
async def create_upload(
    file: UploadFile = File(...),
    doc_type: str = Form("book"),
    ocr_required: bool = Form(True),
    ocr_model: str = Form("qwen"),
    language: str = Form("chinese"),
    tier: str = Form("primary"),
    custom_base_url: str | None = Form(None),
    custom_api_key: str | None = Form(None),
    custom_model: str | None = Form(None),
    driver: Driver = Depends(get_driver),
) -> dict[str, Any]:
    """Accept an upload, persist it to staging, and spawn the pipeline runner."""
    if ocr_model not in _ALLOWED_OCR_MODELS:
        raise HTTPException(status_code=400, detail=f"ocr_model must be one of {_ALLOWED_OCR_MODELS}")
    if tier not in _ALLOWED_TIERS:
        raise HTTPException(status_code=400, detail=f"tier must be one of {_ALLOWED_TIERS}")
    if language not in _ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"language must be one of {_ALLOWED_LANGUAGES}")
    if ocr_model == "custom" and not (custom_base_url and custom_api_key and custom_model):
        raise HTTPException(
            status_code=400,
            detail="custom OCR model requires custom_base_url, custom_api_key and custom_model",
        )

    # Back-pressure: refuse new uploads when too many jobs are still running so
    # we don't spawn unbounded heavy OCR subprocesses.
    running = sum(
        1 for j in upload_job.list_jobs(driver, limit=200) if j.get("status") == "running"
    )
    if running >= _MAX_RUNNING_JOBS:
        raise HTTPException(
            status_code=429,
            detail=f"too many uploads in progress ({running}); retry shortly",
        )

    job_id = uuid.uuid4().hex[:16]
    job_dir = _STAGING_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(file.filename or "upload.bin").name
    dest = job_dir / filename
    # Stream to disk in bounded chunks so a large upload can't buffer the whole
    # body in the uvicorn process memory; enforce a hard size cap.
    written = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds max size of {_MAX_UPLOAD_BYTES} bytes",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"failed to save upload: {exc}")

    upload_job.create_job(
        driver,
        job_id=job_id,
        filename=filename,
        doc_type=doc_type,
        ocr_required=ocr_required,
        ocr_model=ocr_model,
        language=language,
        tier=tier,
    )

    # Build the runner command + environment.
    cmd = [
        sys.executable,
        str(_RUNNER),
        "--job-id", job_id,
        "--path", str(dest),
        "--doc-type", doc_type,
        "--ocr-model", ocr_model,
        "--language", language,
        "--tier", tier,
        "--ocr-required", "1" if ocr_required else "0",
    ]
    env = dict(os.environ)
    if ocr_model == "custom":
        env["CUSTOM_OCR_BASE_URL"] = custom_base_url or ""
        env["CUSTOM_OCR_API_KEY"] = custom_api_key or ""
        env["CUSTOM_OCR_MODEL"] = custom_model or ""

    log_path = job_dir / "pipeline.log"
    try:
        with open(log_path, "ab") as logf:
            subprocess.Popen(  # noqa: S603 — trusted args, detached runner
                cmd,
                cwd=str(_REPO_ROOT),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except Exception as exc:  # noqa: BLE001
        upload_job.update_job(driver, job_id, status="failed", error=f"spawn failed: {exc}")
        raise HTTPException(status_code=500, detail=f"failed to start pipeline: {exc}")

    upload_job.update_job(driver, job_id, status="running", stage="ingest", pct=1.0)
    return {"job_id": job_id, "status": "running", "filename": filename}


@router.get("/status/{job_id}")
async def upload_status(job_id: str, driver: Driver = Depends(get_driver)) -> dict[str, Any]:
    """Return the current state of an upload job (polled by the UI)."""
    job = upload_job.get_job(driver, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    return job


@router.get("/jobs")
async def list_upload_jobs(limit: int = 50, driver: Driver = Depends(get_driver)) -> dict[str, Any]:
    """List recent upload jobs."""
    return {"jobs": upload_job.list_jobs(driver, limit=limit)}


@router.delete("/{job_id}", status_code=204)
async def delete_upload_job(job_id: str, driver: Driver = Depends(get_driver)) -> None:
    """Delete an upload job record and its staging files.

    Refused while the job is still running so we don't orphan an active subprocess.
    """
    job = upload_job.get_job(driver, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    if job.get("status") == "running":
        raise HTTPException(status_code=409, detail="cannot delete a running job; wait for it to finish or fail")

    upload_job.delete_job(driver, job_id)

    job_dir = _STAGING_ROOT / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
