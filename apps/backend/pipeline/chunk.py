"""Chunking pipeline — splits page text into overlapping CHUNK nodes (plan §6 Stage 5).

Text source priority per page:
  1. structuredMarkdown (layoutStatus='ok')  → markdown_section strategy
  2. textFused (fusionStatus in ['ok','single']) → sliding_window strategy
  3. text (native pages)                      → sliding_window strategy

CHUNK node properties (camelCase per AGENTS.md §4):
  id, pageId, documentId, chunkIndex, text, charCount, tokenEstimate,
  chunkStrategy, chunkSize, overlapSize, language, chunkingAt, embeddingStatus
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver

log = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_OVERLAP = 50
_MIN_CHUNK_CHARS = 10


@dataclass
class ChunkRunReport:
    """Summary of a chunking run."""

    pages_total: int = 0
    pages_chunked: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    chunks_created: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages_total": self.pages_total,
            "pages_chunked": self.pages_chunked,
            "pages_skipped": self.pages_skipped,
            "pages_failed": self.pages_failed,
            "chunks_created": self.chunks_created,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors[:20],
        }


def _sliding_window(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size overlapping windows."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def _markdown_sections(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split on markdown headings; merge small sections; overflow via sliding window."""
    heading_re = re.compile(r"^#{1,6}\s", re.MULTILINE)
    boundaries = [m.start() for m in heading_re.finditer(text)]
    if not boundaries:
        return _sliding_window(text, chunk_size=chunk_size, overlap=overlap)

    sections: list[str] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        sections.append(text[start:end])

    if boundaries[0] > 0:
        sections.insert(0, text[: boundaries[0]])

    result: list[str] = []
    buffer = ""
    for sec in sections:
        if len(buffer) + len(sec) <= chunk_size:
            buffer = (buffer + "\n" + sec).lstrip()
        else:
            if buffer:
                result.append(buffer)
            if len(sec) > chunk_size:
                result.extend(_sliding_window(sec, chunk_size=chunk_size, overlap=overlap))
                buffer = ""
            else:
                buffer = sec
    if buffer:
        result.append(buffer)
    return [c for c in result if len(c) >= _MIN_CHUNK_CHARS]


def resolve_page_text(row: dict[str, Any]) -> tuple[str | None, str]:
    """Return (text, strategy) for a page row from Neo4j.

    Priority: structuredMarkdown (layout ok) → textFused → text (native).
    Strategy: 'markdown_section' for structured markdown, 'sliding_window' otherwise.
    """
    md = row.get("structuredMarkdown")
    if md and row.get("layoutStatus") == "ok":
        return md, "markdown_section"
    fused = row.get("textFused")
    if fused and row.get("fusionStatus") in ("ok", "single"):
        return fused, "sliding_window"
    native = row.get("text")
    if native:
        return native, "sliding_window"
    return None, "sliding_window"


def _make_chunk_id(page_id: str, chunk_index: int) -> str:
    """Human-readable chunk ID; unique by (pageId, chunkIndex)."""
    return f"{page_id}::chunk_{chunk_index:04d}"


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

_PAGE_QUERY = """
MATCH (p:PAGE)
WHERE ($recompute OR p.chunkingAt IS NULL)
  AND (p.textFused IS NOT NULL OR p.text IS NOT NULL OR p.structuredMarkdown IS NOT NULL)
RETURN
  p.id              AS page_id,
  p.documentId      AS document_id,
  p.tier            AS tier,
  p.language        AS language,
  p.textFused       AS textFused,
  p.text            AS text,
  p.structuredMarkdown AS structuredMarkdown,
  p.layoutStatus    AS layoutStatus,
  p.fusionStatus    AS fusionStatus
ORDER BY p.id
SKIP $skip LIMIT $batch
"""

_CHUNK_UPSERT = """
UNWIND $chunks AS c
MERGE (ch:CHUNK {id: c.id})
ON CREATE SET
  ch.pageId          = c.pageId,
  ch.documentId      = c.documentId,
  ch.tier            = c.tier,
  ch.chunkIndex      = c.chunkIndex,
  ch.text            = c.text,
  ch.charCount       = c.charCount,
  ch.tokenEstimate   = c.tokenEstimate,
  ch.chunkStrategy   = c.chunkStrategy,
  ch.chunkSize       = c.chunkSize,
  ch.overlapSize     = c.overlapSize,
  ch.language        = c.language,
  ch.chunkingAt      = c.chunkingAt,
  ch.embeddingStatus = 'pending'
ON MATCH SET
  ch.text            = c.text,
  ch.charCount       = c.charCount,
  ch.tier            = c.tier,
  ch.chunkingAt      = c.chunkingAt
WITH ch, c
MATCH (p:PAGE {id: c.pageId})
MERGE (p)-[:HAS]->(ch)
"""

_STAMP_PAGE = """
MATCH (p:PAGE {id: $page_id})
SET p.chunkingAt = $ts, p.chunkCount = $count
"""


def chunk_pages(
    driver: Driver,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
    batch_size: int = 200,
    max_pages: int | None = None,
    recompute: bool = False,
) -> ChunkRunReport:
    """Chunk all eligible pages and write CHUNK nodes to Neo4j.

    Eligible pages have at least one text source (textFused, text, or
    structuredMarkdown) and have not been chunked yet (or recompute=True).
    """
    report = ChunkRunReport()
    t_start = time.time()
    skip = 0

    while True:
        with driver.session() as s:
            rows = s.run(_PAGE_QUERY, recompute=recompute, skip=skip, batch=batch_size).data()

        if not rows:
            break

        report.pages_total += len(rows)
        batch_chunks: list[dict[str, Any]] = []
        page_counts: dict[str, int] = {}

        for row in rows:
            page_id = row["page_id"]
            try:
                text, strategy = resolve_page_text(row)
                if not text or len(text.strip()) < _MIN_CHUNK_CHARS:
                    report.pages_skipped += 1
                    continue

                split_fn = _markdown_sections if strategy == "markdown_section" else _sliding_window
                raw_chunks = split_fn(text.strip(), chunk_size=chunk_size, overlap=overlap)

                if not raw_chunks:
                    report.pages_skipped += 1
                    continue

                for idx, chunk_text in enumerate(raw_chunks):
                    batch_chunks.append({
                        "id": _make_chunk_id(page_id, idx),
                        "pageId": page_id,
                        "documentId": row.get("document_id") or "",
                        "tier": row.get("tier"),
                        "chunkIndex": idx,
                        "text": chunk_text,
                        "charCount": len(chunk_text),
                        "tokenEstimate": int(len(chunk_text) / 1.5),
                        "chunkStrategy": strategy,
                        "chunkSize": chunk_size,
                        "overlapSize": overlap,
                        "language": row.get("language"),
                        "chunkingAt": datetime.now(timezone.utc).isoformat(),
                    })
                page_counts[page_id] = len(raw_chunks)
                report.pages_chunked += 1
                report.chunks_created += len(raw_chunks)

            except Exception as exc:
                log.error("chunk failed page=%s: %s", page_id, exc)
                report.pages_failed += 1
                report.errors.append(f"{page_id}: {exc}")

        if batch_chunks:
            with driver.session() as s:
                s.run(_CHUNK_UPSERT, chunks=batch_chunks).consume()
            ts_now = datetime.now(timezone.utc).isoformat()
            with driver.session() as s:
                for pid, cnt in page_counts.items():
                    s.run(_STAMP_PAGE, page_id=pid, ts=ts_now, count=cnt).consume()

        log.info(
            "Batch skip=%d rows=%d chunks_this_batch=%d total_chunks=%d",
            skip,
            len(rows),
            len(batch_chunks),
            report.chunks_created,
        )
        skip += len(rows)

        if max_pages is not None and report.pages_total >= max_pages:
            break

    report.duration_seconds = time.time() - t_start
    return report
