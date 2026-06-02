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
