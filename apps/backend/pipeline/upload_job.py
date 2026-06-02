"""UPLOAD_JOB state machine for the website upload workflow (Track E).

An ``UPLOAD_JOB`` node tracks one user upload as it moves through the
ingest -> preprocess -> OCR -> fuse -> layout pipeline run by
``scripts/run_upload_pipeline.py``. The web UI polls
``GET /api/upload/status/{job_id}`` which reads these nodes.

Lifecycle (``status``):
    queued -> running -> awaiting_review   (OCR done, HITL pending)
                       -> failed
    (after HITL approve + index pipeline) -> indexed

``stage`` is a free-form label for the current pipeline step
(``ingest`` / ``preprocess`` / ``ocr`` / ``fuse`` / ``layout`` / ``done``).

Security: user-supplied custom-OCR credentials are NEVER written to the
node. The runner receives them via environment variables at spawn time.
"""
from __future__ import annotations

import logging
from typing import Any

from neo4j import Driver

log = logging.getLogger(__name__)


_CREATE = """
MERGE (j:UPLOAD_JOB {id: $id})
ON CREATE SET
    j.status       = 'queued',
    j.stage        = 'queued',
    j.pct          = 0.0,
    j.filename     = $filename,
    j.docType      = $doc_type,
    j.ocrRequired  = $ocr_required,
    j.ocrModel     = $ocr_model,
    j.language     = $language,
    j.tier         = $tier,
    j.documentId   = $document_id,
    j.error        = null,
    j.createdAt    = timestamp(),
    j.updatedAt    = timestamp()
RETURN j.id AS id
"""

_UPDATE = """
MATCH (j:UPLOAD_JOB {id: $id})
SET j.status     = coalesce($status, j.status),
    j.stage      = coalesce($stage, j.stage),
    j.pct        = coalesce($pct, j.pct),
    j.documentId = coalesce($document_id, j.documentId),
    j.error      = coalesce($error, j.error),
    j.updatedAt  = timestamp()
RETURN j.id AS id
"""

_GET = """
MATCH (j:UPLOAD_JOB {id: $id})
RETURN j.id AS id, j.status AS status, j.stage AS stage, j.pct AS pct,
       j.filename AS filename, j.docType AS doc_type,
       j.ocrRequired AS ocr_required, j.ocrModel AS ocr_model,
       j.language AS language, j.tier AS tier, j.documentId AS document_id,
       j.error AS error, j.createdAt AS created_at, j.updatedAt AS updated_at
"""

_LIST = """
MATCH (j:UPLOAD_JOB)
RETURN j.id AS id, j.status AS status, j.stage AS stage, j.pct AS pct,
       j.filename AS filename, j.docType AS doc_type, j.ocrModel AS ocr_model,
       j.language AS language, j.tier AS tier, j.documentId AS document_id,
       j.error AS error, j.createdAt AS created_at, j.updatedAt AS updated_at
ORDER BY j.createdAt DESC
LIMIT $limit
"""

_DELETE = """
MATCH (j:UPLOAD_JOB {id: $id})
DELETE j
"""


def create_job(
    driver: Driver,
    *,
    job_id: str,
    filename: str,
    doc_type: str,
    ocr_required: bool,
    ocr_model: str,
    language: str,
    tier: str,
    document_id: str | None = None,
) -> str:
    """Create an UPLOAD_JOB node in ``queued`` state."""
    with driver.session() as s:
        s.run(
            _CREATE,
            id=job_id,
            filename=filename,
            doc_type=doc_type,
            ocr_required=ocr_required,
            ocr_model=ocr_model,
            language=language,
            tier=tier,
            document_id=document_id,
        ).consume()
    return job_id


def update_job(
    driver: Driver,
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    pct: float | None = None,
    document_id: str | None = None,
    error: str | None = None,
) -> None:
    """Patch an UPLOAD_JOB node (only non-None fields are written)."""
    with driver.session() as s:
        s.run(
            _UPDATE,
            id=job_id,
            status=status,
            stage=stage,
            pct=pct,
            document_id=document_id,
            error=error,
        ).consume()


def get_job(driver: Driver, job_id: str) -> dict[str, Any] | None:
    """Return one UPLOAD_JOB as a dict, or None if missing."""
    with driver.session() as s:
        row = s.run(_GET, id=job_id).single()
    return dict(row) if row else None


def list_jobs(driver: Driver, *, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent UPLOAD_JOB nodes."""
    with driver.session() as s:
        return [dict(r) for r in s.run(_LIST, limit=limit)]


def delete_job(driver: Driver, job_id: str) -> None:
    """Delete an UPLOAD_JOB node by id."""
    with driver.session() as s:
        s.run(_DELETE, id=job_id).consume()
