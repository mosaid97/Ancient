"""Documents router — lists ingested documents + their pages for the UI.

Powers the interactive hover view's document picker and the HITL queue's
document filter. Page listing returns OCR / layout / fusion status so the
frontend can show which pages are ready for review or interaction.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from neo4j import Driver

from apps.backend.api.deps import get_driver

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
async def list_documents(
    status: str | None = Query(None, description="awaiting_review | approved | indexed"),
    tier: str | None = Query(None, description="primary | secondary"),
    limit: int = Query(200, ge=1, le=1000),
    driver: Driver = Depends(get_driver),
) -> dict[str, Any]:
    """List documents with page counts and lifecycle status."""
    where = []
    if status:
        where.append("d.status = $status")
    if tier:
        where.append("d.tier = $tier")
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""

    cypher = f"""
    MATCH (d:DOCUMENT)
    {where_clause}
    OPTIONAL MATCH (d)-[:CONSIST_OF]->(:CHAPTER)-[:INCLUDE]->(:SECTION)-[:INCLUDE]->(p:PAGE)
    WITH d, count(DISTINCT p) AS page_count
    RETURN d.id AS id, d.title AS title, d.tier AS tier,
           coalesce(d.status, 'indexed') AS status,
           d.backend AS backend, d.edition AS edition,
           d.publicationYear AS publication_year,
           page_count
    ORDER BY d.title
    LIMIT $limit
    """
    try:
        with driver.session() as s:
            rows = s.run(cypher, status=status, tier=tier, limit=limit).data()
    except Exception as exc:  # noqa: BLE001
        log.exception("list_documents failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"documents": rows, "total": len(rows)}


@router.get("/{document_id}/pages")
async def list_pages(
    document_id: str,
    only_with_layout: bool = Query(False, description="only pages with layoutStatus='ok'"),
    driver: Driver = Depends(get_driver),
) -> dict[str, Any]:
    """List pages of a document with per-page OCR / layout / fusion status."""
    layout_clause = "AND p.layoutStatus = 'ok'" if only_with_layout else ""
    cypher = f"""
    MATCH (p:PAGE {{documentId: $doc}})
    WHERE p.role IS NULL OR p.role = 'body'
    {layout_clause}
    RETURN p.id AS page_id,
           p.docPageIndex AS page_index,
           p.mode AS mode,
           p.language AS language,
           p.tier AS tier,
           p.fusionStatus AS fusion_status,
           p.layoutStatus AS layout_status,
           p.layoutRegionCount AS region_count,
           p.hitlStatus AS hitl_status,
           p.fusionCharCount AS char_count
    ORDER BY p.docPageIndex
    """
    try:
        with driver.session() as s:
            rows = s.run(cypher, doc=document_id).data()
    except Exception as exc:  # noqa: BLE001
        log.exception("list_pages failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    if not rows:
        # Distinguish empty doc from missing doc.
        with driver.session() as s:
            exists = s.run("MATCH (d:DOCUMENT {id:$doc}) RETURN d.id AS id", doc=document_id).single()
        if exists is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id!r} not found")
    return {"document_id": document_id, "pages": rows, "total": len(rows)}


# Cascade order: CHUNK → PAGE → SECTION → CHAPTER → DOCUMENT. MinIO objects
# (page images, preprocessed images, OCR JSONs) are deleted before the
# graph nodes so a partial failure leaves the graph intact for retry.
_DELETE_CASCADE = """
MATCH (d:DOCUMENT {id: $doc_id})
OPTIONAL MATCH (d)-[:CONSIST_OF]->(ch:CHAPTER)
OPTIONAL MATCH (ch)-[:INCLUDE]->(sec:SECTION)
OPTIONAL MATCH (sec)-[:INCLUDE]->(p:PAGE)
OPTIONAL MATCH (p)-[:HAS]->(c:CHUNK)
WITH d, ch, sec, p, c, collect(DISTINCT p.imageUri) + collect(DISTINCT p.preprocessedImageUri) AS uris
DETACH DELETE c, p, sec, ch, d
RETURN [u IN uris WHERE u IS NOT NULL] AS uris
"""


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    driver: Driver = Depends(get_driver),
) -> dict[str, Any]:
    """Permanently delete a document and every node it owns.

    Cascade: CHUNK → PAGE → SECTION → CHAPTER → DOCUMENT, plus MinIO
    objects keyed off ``PAGE.imageUri`` / ``preprocessedImageUri``. The
    deletion is a single Cypher transaction so it is all-or-nothing on
    the graph side; MinIO failures are logged but do not abort the call
    (orphan blobs can be GC'd separately, and re-running delete is safe).
    """
    # Confirm the document exists first so callers get a 404 instead of
    # a successful empty cascade for typos.
    with driver.session() as s:
        exists = s.run(
            "MATCH (d:DOCUMENT {id: $doc}) RETURN d.id AS id, d.title AS title",
            doc=document_id,
        ).single()
    if exists is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id!r} not found")

    title = exists["title"]
    uris: list[str] = []
    try:
        with driver.session() as s:
            row = s.run(_DELETE_CASCADE, doc_id=document_id).single()
            uris = (row["uris"] if row else []) or []
    except Exception as exc:
        log.exception("delete_document graph cascade failed for %r: %s", document_id, exc)
        raise HTTPException(status_code=500, detail=f"graph delete failed: {exc}")

    # Best-effort blob cleanup — never fails the request.
    blobs_deleted = 0
    blobs_failed = 0
    if uris:
        try:
            from apps.backend.storage import get_minio_client
            mc = get_minio_client()
            for uri in uris:
                bucket = "ancient-pages"
                key = uri
                if uri.startswith("minio://"):
                    rest = uri.removeprefix("minio://")
                    bucket, _, key = rest.partition("/")
                try:
                    mc.remove_object(bucket, key)
                    blobs_deleted += 1
                except Exception as exc:
                    log.warning("MinIO remove failed for %s/%s: %s", bucket, key, exc)
                    blobs_failed += 1
        except Exception as exc:
            log.warning("delete_document blob cleanup unavailable: %s", exc)

    log.info(
        "delete_document: %r (%s) deleted — %d blobs removed, %d failed",
        document_id, title, blobs_deleted, blobs_failed,
    )
    return {
        "document_id": document_id,
        "title": title,
        "blobs_deleted": blobs_deleted,
        "blobs_failed": blobs_failed,
    }
