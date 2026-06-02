"""Interactive hover dashboard router (Track E, Feature 5).

Serves the data behind the hover-to-reveal OCR view:

- ``GET  /api/interactive/page/{page_id}`` — preprocessed image URL +
  parsed ``PAGE.layoutJson`` regions (bbox in preprocessed-image pixel
  coordinates + per-region OCR text) + the page's fused text + saved
  comments. The frontend scales each bbox by ``rendered/natural`` image
  size and shows the region text on hover.
- ``POST /api/interactive/comment`` — append a ``COMMENT`` node to a page.
- ``POST /api/interactive/ask`` — document-scoped retrieval (right-panel
  chat/search), reusing the dense search and filtering to one document.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from neo4j import Driver
from pydantic import BaseModel

from apps.backend.api.deps import get_driver

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/page/{page_id}")
async def page_regions(page_id: str, driver: Driver = Depends(get_driver)) -> dict[str, Any]:
    """Return image URL + layout regions + fused text + comments for a page."""
    with driver.session() as s:
        row = s.run(
            """
            MATCH (p:PAGE {id: $pid})
            RETURN p.id AS page_id, p.documentId AS document_id,
                   p.docPageIndex AS page_index,
                   p.layoutJson AS layout_json,
                   p.layoutStatus AS layout_status,
                   p.textFused AS text_fused,
                   p.structuredMarkdown AS structured_markdown,
                   p.preprocessedImageUri AS preprocessed_uri,
                   p.imageUri AS image_uri
            """,
            pid=page_id,
        ).single()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Page {page_id!r} not found")

    regions: list[dict[str, Any]] = []
    raw = row["layout_json"]
    if raw:
        try:
            parsed = json.loads(raw)
            for r in parsed:
                bbox = r.get("bbox")
                if not bbox:
                    continue
                regions.append({
                    "label": r.get("label", "text"),
                    "bbox": bbox,  # [x, y, w, h] in preprocessed pixel space
                    "text": r.get("text", ""),
                    "table_html": r.get("table_html"),
                })
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning("layoutJson parse failed for %s: %s", page_id, exc)

    # Prefer the preprocessed variant because layout bboxes are in its space.
    image_variant = "preprocessed" if row["preprocessed_uri"] else "original"

    with driver.session() as s:
        comments = [
            dict(c)
            for c in s.run(
                """
                MATCH (p:PAGE {id: $pid})-[:HAS_COMMENT]->(c:COMMENT)
                RETURN c.id AS id, c.text AS text, c.author AS author,
                       c.createdAt AS created_at
                ORDER BY c.createdAt
                """,
                pid=page_id,
            )
        ]

    return {
        "page_id": page_id,
        "document_id": row["document_id"],
        "page_index": row["page_index"],
        "image_url": f"/api/image/{page_id}?variant={image_variant}",
        "image_variant": image_variant,
        "layout_status": row["layout_status"],
        "regions": regions,
        "region_count": len(regions),
        "text_fused": row["text_fused"],
        "structured_markdown": row["structured_markdown"],
        "comments": comments,
    }


class CommentIn(BaseModel):
    page_id: str
    text: str
    author: str | None = "anonymous"


@router.post("/comment")
async def add_comment(payload: CommentIn, driver: Driver = Depends(get_driver)) -> dict[str, Any]:
    """Attach a COMMENT node to a page (interactive-view annotations)."""
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="comment text is empty")
    comment_id = uuid.uuid4().hex[:16]
    with driver.session() as s:
        row = s.run(
            """
            MATCH (p:PAGE {id: $pid})
            MERGE (c:COMMENT {id: $cid})
            ON CREATE SET c.text = $text, c.author = $author,
                          c.pageId = $pid, c.createdAt = timestamp()
            MERGE (p)-[:HAS_COMMENT]->(c)
            RETURN c.id AS id
            """,
            pid=payload.page_id, cid=comment_id,
            text=payload.text.strip(), author=payload.author or "anonymous",
        ).single()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Page {payload.page_id!r} not found")
    return {"id": comment_id, "ok": True}


class AskIn(BaseModel):
    document_id: str
    query: str
    top_k: int = 5


@router.post("/ask")
async def ask_document(payload: AskIn, driver: Driver = Depends(get_driver)) -> dict[str, Any]:
    """Document-scoped retrieval for the interactive-view chat panel."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is empty")
    try:
        from apps.backend.pipeline.search import search as dense_search_fn

        raw = dense_search_fn(driver, payload.query, top_k=max(payload.top_k * 4, 20))
    except Exception as exc:  # noqa: BLE001
        log.exception("interactive ask failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    hits: list[dict[str, Any]] = []
    for r in raw:
        spine = getattr(r, "spine", None)
        doc_id = getattr(spine, "document_id", None) if spine else None
        if doc_id is None:
            doc_id = getattr(r.hit, "document_id", None)
        if doc_id != payload.document_id:
            continue
        citation = None
        if spine is not None and hasattr(spine, "citation_label"):
            citation = spine.citation_label()
        hits.append({
            "chunk_id": r.hit.chunk_id,
            "text": r.hit.text,
            "page_id": getattr(r.hit, "page_id", None),
            "score": r.hit.score,
            "citation": citation,
        })
        if len(hits) >= payload.top_k:
            break

    return {"document_id": payload.document_id, "query": payload.query, "hits": hits}
