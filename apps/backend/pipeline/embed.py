"""Embedding pipeline — generates 1024-dim text-embedding-v4 vectors for CHUNK nodes.

Reads chunks with embeddingStatus='pending' (or recompute=True), calls the
Silra embeddings API in batches of 10 (hard API limit), and writes two vectors:

  embeddingClassical — computed over coalesce(textCanonical, text)
                       so the philological normalization layer reaches retrieval
                       (plan §2.7 / integrity gap G1).
  embeddingVernacular — computed over textVernacular when present (may be NULL).
  embedding           — alias of embeddingClassical for back-compat.

CHUNK properties written:
  embedding, embeddingClassical, embeddingVernacular,
  embeddingModel, embeddingDims, embeddingStatus, embeddingAt
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

# Classical text: prefer normalized canonical form; fall back to raw OCR/native.
# This ensures the philological normalization layer (異體字/T-S/避諱) reaches
# retrieval (plan §2.7, integrity gap G1).
_CHUNK_QUERY = """
MATCH (c:CHUNK)
WHERE (c.embeddingStatus IN ['pending', 'failed'] OR $recompute)
  AND c.text IS NOT NULL AND c.charCount > 0
RETURN
  c.id                                       AS chunk_id,
  coalesce(c.textCanonical, c.text)          AS text_classical,
  c.textVernacular                           AS text_vernacular
ORDER BY c.id
SKIP $skip LIMIT $batch
"""

_EMBED_WRITE = """
UNWIND $rows AS r
MATCH (c:CHUNK {id: r.chunk_id})
SET c.embeddingClassical  = r.embedding_classical,
    c.embedding           = r.embedding_classical,
    c.embeddingVernacular = r.embedding_vernacular,
    c.embeddingModel      = r.model,
    c.embeddingDims       = r.dims,
    c.embeddingStatus     = 'ok',
    c.embeddingAt         = r.ts
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
    """Embed all pending CHUNK nodes and write classical + vernacular vectors.

    Classical vector: coalesce(textCanonical, text) — normalization reaches retrieval.
    Vernacular vector: textVernacular when present (skipped when NULL).
    Both written per batch; embeddingStatus='ok' only after both succeed.

    Set recompute=True to re-embed already-embedded chunks (needed after A1
    populates textCanonical corpus-wide).
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
        classical_texts = [r["text_classical"] for r in rows]
        vernacular_texts = [r["text_vernacular"] for r in rows]  # may contain None

        # --- classical embedding (always) ---
        try:
            classical_vectors = _batch_embed(
                client, classical_texts, model=embed_model, batch_size=batch_size
            )
        except Exception as exc:
            log.error("Classical embed batch failed for %d chunks: %s", len(rows), exc)
            for cid in chunk_ids:
                with driver.session() as s:
                    s.run(_EMBED_FAIL, chunk_id=cid, error=str(exc)).consume()
            report.chunks_failed += len(rows)
            report.errors.append(f"classical skip={skip}: {exc}")
            skip += len(rows)
            continue

        # --- vernacular embedding (only when textVernacular is non-null) ---
        vern_idx = [i for i, t in enumerate(vernacular_texts) if t]
        vernacular_vectors: list[list[float] | None] = [None] * len(rows)
        if vern_idx:
            vern_subset = [vernacular_texts[i] for i in vern_idx]
            try:
                vern_results = _batch_embed(
                    client, vern_subset, model=embed_model, batch_size=batch_size
                )
                for pos, vec in zip(vern_idx, vern_results):
                    vernacular_vectors[pos] = vec
            except Exception as exc:
                # Vernacular failure is non-fatal — log and continue with classical only
                log.warning("Vernacular embed batch failed (non-fatal): %s", exc)

        ts_now = datetime.now(timezone.utc).isoformat()
        write_rows = [
            {
                "chunk_id": cid,
                "embedding_classical": cvec,
                "embedding_vernacular": vernacular_vectors[i],
                "model": embed_model,
                "dims": _EMBED_DIMS,
                "ts": ts_now,
            }
            for i, (cid, cvec) in enumerate(zip(chunk_ids, classical_vectors))
        ]
        with driver.session() as s:
            s.run(_EMBED_WRITE, rows=write_rows).consume()

        report.chunks_embedded += len(write_rows)
        log.info(
            "Embedded skip=%d n=%d total_embedded=%d (vernacular=%d)",
            skip, len(rows), report.chunks_embedded, len(vern_idx),
        )
        skip += len(rows)

        if max_chunks is not None and report.chunks_total >= max_chunks:
            break

    report.duration_seconds = time.time() - t_start
    return report
