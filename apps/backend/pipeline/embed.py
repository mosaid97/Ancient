"""Embedding pipeline — generates 1024-dim text-embedding-v4 vectors for CHUNK nodes.

Reads chunks with embeddingStatus='pending' (or recompute=True), calls
the Silra embeddings API in batches of 32, and writes float[] vectors back
to CHUNK.embedding alongside metadata properties.

CHUNK properties written:
  embedding, embeddingModel, embeddingDims, embeddingStatus, embeddingAt
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver
from openai import OpenAI

log = logging.getLogger(__name__)

_EMBED_DIMS = 1024
_DEFAULT_BATCH = 10   # Silra text-embedding-v4 rejects batches larger than 10
_MAX_RETRIES = 3


def _get_embed_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.silra.cn/v1/"),
    )


def _batch_embed(
    client: OpenAI,
    texts: list[str],
    *,
    model: str,
    batch_size: int = _DEFAULT_BATCH,
) -> list[list[float]]:
    """Embed a list of texts in batches; return list of 1024-dim float vectors."""
    if not texts:
        return []
    result: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.embeddings.create(input=batch, model=model)
                result.extend([item.embedding for item in resp.data])
                break
            except Exception as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = 2**attempt
                log.warning(
                    "embed batch attempt %d failed (%s), retry in %ds", attempt + 1, exc, wait
                )
                time.sleep(wait)
    return result


@dataclass
class EmbedRunReport:
    """Summary of an embedding run."""

    chunks_total: int = 0
    chunks_embedded: int = 0
    chunks_failed: int = 0
    chunks_skipped: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_total": self.chunks_total,
            "chunks_embedded": self.chunks_embedded,
            "chunks_failed": self.chunks_failed,
            "chunks_skipped": self.chunks_skipped,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors[:20],
        }


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

_CHUNK_QUERY = """
MATCH (c:CHUNK)
WHERE (c.embeddingStatus = 'pending' OR $recompute)
  AND c.text IS NOT NULL AND c.charCount > 0
RETURN c.id AS chunk_id, c.text AS text
ORDER BY c.id
SKIP $skip LIMIT $batch
"""

_EMBED_WRITE = """
UNWIND $rows AS r
MATCH (c:CHUNK {id: r.chunk_id})
SET c.embedding       = r.embedding,
    c.embeddingModel  = r.model,
    c.embeddingDims   = r.dims,
    c.embeddingStatus = 'ok',
    c.embeddingAt     = r.ts
"""

_EMBED_FAIL = """
MATCH (c:CHUNK {id: $chunk_id})
SET c.embeddingStatus = 'failed', c.embeddingError = $error
"""


def embed_chunks(
    driver: Driver,
    *,
    model: str | None = None,
    batch_size: int = _DEFAULT_BATCH,
    max_chunks: int | None = None,
    recompute: bool = False,
) -> EmbedRunReport:
    """Embed all pending CHUNK nodes and write vectors back to Neo4j.

    Chunks with embeddingStatus='pending' are processed by default.
    Set recompute=True to re-embed already-embedded chunks.
    """
    embed_model = model or os.getenv("EMBED_LLM_MODEL", "text-embedding-v4")
    client = _get_embed_client()
    report = EmbedRunReport()
    t_start = time.time()
    skip = 0

    while True:
        with driver.session() as s:
            rows = s.run(
                _CHUNK_QUERY, recompute=recompute, skip=skip, batch=batch_size * 4
            ).data()

        if not rows:
            break

        report.chunks_total += len(rows)
        chunk_ids = [r["chunk_id"] for r in rows]
        texts = [r["text"] for r in rows]

        try:
            vectors = _batch_embed(client, texts, model=embed_model, batch_size=batch_size)
        except Exception as exc:
            log.error("Batch embedding failed for %d chunks: %s", len(rows), exc)
            for cid in chunk_ids:
                with driver.session() as s:
                    s.run(_EMBED_FAIL, chunk_id=cid, error=str(exc)).consume()
            report.chunks_failed += len(rows)
            report.errors.append(f"batch skip={skip}: {exc}")
            skip += len(rows)
            continue

        ts_now = datetime.now(timezone.utc).isoformat()
        write_rows = [
            {
                "chunk_id": cid,
                "embedding": vec,
                "model": embed_model,
                "dims": _EMBED_DIMS,
                "ts": ts_now,
            }
            for cid, vec in zip(chunk_ids, vectors)
        ]
        with driver.session() as s:
            s.run(_EMBED_WRITE, rows=write_rows).consume()

        report.chunks_embedded += len(write_rows)
        log.info(
            "Embedded skip=%d n=%d total_embedded=%d",
            skip,
            len(rows),
            report.chunks_embedded,
        )
        skip += len(rows)

        if max_chunks is not None and report.chunks_total >= max_chunks:
            break

    report.duration_seconds = time.time() - t_start
    return report
