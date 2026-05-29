"""B1: Keyword embedding + RELATED edge construction (plan §0.6 Track B, B1).

Two phases:
  1. embed_keywords() — writes KEYWORD.embedding via text-embedding-v4 on KEYWORD.name
  2. link_related_keywords() — for each embedded keyword, queries the ANN vector
     index and MERGEs (:KEYWORD)-[:RELATED {score}]->(:KEYWORD) for cosine ≥ threshold

ADR (AGENTS.md §11 2026-05-29): RELATED edge threshold = 0.85 (text-embedding-v4
cosine space). Lower values produce noisy synonym clusters; higher values miss
cross-script pairs (e.g. 均田 ↔ 均田制).

The RELATED relationship is directed from the lookup keyword to the similar
keyword (arbitrary — both directions are written). Callers that want a symmetric
graph should follow edges in either direction.
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
_DEFAULT_BATCH = 10      # Silra text-embedding-v4 hard limit
_MAX_RETRIES = 5
_DEFAULT_THRESHOLD = 0.85
_ANN_TOP_K = 50          # candidates per query before threshold filter


# ── Cypher ───────────────────────────────────────────────────────────────────

_KEYWORD_FETCH = """
MATCH (k:KEYWORD)
WHERE k.embedding IS NULL
RETURN k.name AS name, k.type AS type
ORDER BY k.name
SKIP $skip LIMIT $batch
"""

_EMBED_WRITE = """
UNWIND $rows AS r
MATCH (k:KEYWORD {name: r.name})
SET k.embedding = r.embedding,
    k.embeddingModel = r.model,
    k.embeddingAt = r.ts
"""

_RELATED_FETCH = """
MATCH (k:KEYWORD)
WHERE k.embedding IS NOT NULL
  AND NOT EXISTS { (k)-[:RELATED]->(:KEYWORD) }
RETURN k.name AS name, k.embedding AS embedding
SKIP $skip LIMIT $batch
"""

_RELATED_MERGE = """
UNWIND $edges AS e
MATCH (a:KEYWORD {name: e.src}), (b:KEYWORD {name: e.dst})
WHERE a <> b
MERGE (a)-[r:RELATED]->(b)
SET r.score = e.score,
    r.model = e.model,
    r.ts    = e.ts
"""

_VECTOR_QUERY = """
CALL db.index.vector.queryNodes($index, $top_k, $embedding)
YIELD node AS k, score
WHERE score >= $threshold AND k.name <> $src_name
RETURN k.name AS name, score
"""


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class EmbedReport:
    keywords_total: int = 0
    keywords_embedded: int = 0
    keywords_failed: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords_total": self.keywords_total,
            "keywords_embedded": self.keywords_embedded,
            "keywords_failed": self.keywords_failed,
            "duration_seconds": round(self.duration_seconds, 2),
            "errors": self.errors[:20],
        }


@dataclass
class RelateReport:
    keywords_processed: int = 0
    edges_created: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords_processed": self.keywords_processed,
            "edges_created": self.edges_created,
            "duration_seconds": round(self.duration_seconds, 2),
            "errors": self.errors[:20],
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

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
                wait = 2 ** attempt
                log.warning("embed attempt %d failed (%s), retry in %ds", attempt + 1, exc, wait)
                time.sleep(wait)
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def embed_keywords(
    driver: Driver,
    *,
    model: str | None = None,
    batch_size: int = _DEFAULT_BATCH,
    max_keywords: int | None = None,
) -> EmbedReport:
    """Embed all KEYWORD nodes that lack an embedding vector.

    Writes KEYWORD.embedding (1024-dim text-embedding-v4) + KEYWORD.embeddingModel
    and KEYWORD.embeddingAt. Uses KEYWORD.name as the embedding input.
    """
    embed_model = model or os.getenv("EMBED_LLM_MODEL", "text-embedding-v4")
    client = _get_embed_client()
    report = EmbedReport()
    t_start = time.time()
    skip = 0
    fetch_batch = batch_size * 20  # fetch more per DB round-trip

    while True:
        with driver.session() as s:
            rows = s.run(_KEYWORD_FETCH, skip=skip, batch=fetch_batch).data()

        if not rows:
            break

        report.keywords_total += len(rows)
        names = [r["name"] for r in rows]
        ts_now = datetime.now(timezone.utc).isoformat()

        # Process in mini-batches of batch_size; handle failures per mini-batch
        # so a transient connection error only skips those 10 items, not all 200.
        write_rows: list[dict] = []
        for i in range(0, len(names), batch_size):
            mini = names[i : i + batch_size]
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = client.embeddings.create(input=mini, model=embed_model)
                    for name, item in zip(mini, resp.data):
                        write_rows.append({
                            "name": name,
                            "embedding": item.embedding,
                            "model": embed_model,
                            "ts": ts_now,
                        })
                    break
                except Exception as exc:
                    if attempt == _MAX_RETRIES - 1:
                        log.warning("Mini-batch embed failed after %d retries (skip=%d+%d): %s", _MAX_RETRIES, skip, i, exc)
                        report.keywords_failed += len(mini)
                        report.errors.append(f"skip={skip}+{i}: {exc}")
                    else:
                        wait = 2 ** attempt
                        log.debug("embed attempt %d/%d failed, retry in %ds: %s", attempt + 1, _MAX_RETRIES, wait, exc)
                        time.sleep(wait)

        if write_rows:
            with driver.session() as s:
                s.run(_EMBED_WRITE, rows=write_rows).consume()
            report.keywords_embedded += len(write_rows)

        log.info(
            "Keyword embed skip=%d n=%d embedded=%d failed=%d total_embedded=%d",
            skip, len(rows), len(write_rows), report.keywords_failed, report.keywords_embedded,
        )
        skip += len(rows)

        if max_keywords is not None and report.keywords_total >= max_keywords:
            break

    report.duration_seconds = time.time() - t_start
    return report


def link_related_keywords(
    driver: Driver,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    top_k: int = _ANN_TOP_K,
    index_name: str = "keyword_embedding_index",
    batch_size: int = 500,
    max_keywords: int | None = None,
    model: str | None = None,
) -> RelateReport:
    """Build (:KEYWORD)-[:RELATED {score}]->(:KEYWORD) edges via ANN vector search.

    For each keyword that has an embedding but no RELATED edges yet, queries the
    ANN index for the top-K nearest neighbors and writes edges for any that score
    above ``threshold``.

    Args:
        driver: Open Neo4j driver.
        threshold: Minimum cosine similarity (default 0.85 per ADR).
        top_k: Number of ANN candidates to retrieve per keyword.
        index_name: Name of the KEYWORD vector index.
        batch_size: Keywords to process per iteration.
        max_keywords: Optional cap (for smoke tests).
        model: Embedding model label written to edge (informational only).
    """
    embed_model = model or os.getenv("EMBED_LLM_MODEL", "text-embedding-v4")
    report = RelateReport()
    t_start = time.time()
    skip = 0

    while True:
        with driver.session() as s:
            rows = s.run(_RELATED_FETCH, skip=skip, batch=batch_size).data()

        if not rows:
            break

        report.keywords_processed += len(rows)
        ts_now = datetime.now(timezone.utc).isoformat()
        edges: list[dict[str, Any]] = []

        for row in rows:
            src_name = row["name"]
            embedding = row["embedding"]
            if not embedding:
                continue
            try:
                with driver.session() as s:
                    similar = s.run(
                        _VECTOR_QUERY,
                        index=index_name,
                        top_k=top_k,
                        embedding=embedding,
                        threshold=threshold,
                        src_name=src_name,
                    ).data()
                for sim_row in similar:
                    edges.append({
                        "src": src_name,
                        "dst": sim_row["name"],
                        "score": float(sim_row["score"]),
                        "model": embed_model,
                        "ts": ts_now,
                    })
            except Exception as exc:
                log.warning("Vector query failed for keyword %r: %s", src_name, exc)
                report.errors.append(f"{src_name}: {exc}")

        if edges:
            with driver.session() as s:
                s.run(_RELATED_MERGE, edges=edges).consume()
            report.edges_created += len(edges)

        log.info(
            "RELATED link skip=%d kw=%d edges_this_batch=%d total_edges=%d",
            skip, len(rows), len(edges), report.edges_created,
        )
        skip += len(rows)

        if max_keywords is not None and report.keywords_processed >= max_keywords:
            break

    report.duration_seconds = time.time() - t_start
    return report
