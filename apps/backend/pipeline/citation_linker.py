"""B2: Entailment-based Secondary→Primary CITES linker (plan §0.6 Track B, B2).

Algorithm (per plan §0.6 B2 + §8):
  1. For each secondary CHUNK, extract candidate quoted/abridged spans using
     CJK quote-mark patterns (「」『』""'' + …省略/…省) and sliding-window
     fallback for unmarked paraphrase.
  2. Embed each span via text-embedding-v4 and retrieve top-K primary CHUNK
     candidates from the chunk_embedding_classical ANN index.
  3. Score (secondary_span, primary_chunk_text) pairs with the
     BGE-reranker-v2-gemma cross-encoder (FlagEmbedding).
  4. MERGE (:CHUNK{tier:'secondary'})-[:CITES {quoteSpan, confidence,
     extractionMethod:'entailment', linkerModelVersion}]->(:CHUNK{tier:'primary'})
     for scores ≥ threshold (default 0.85).

ADR (AGENTS.md §11): CITES threshold = 0.85, BGE-reranker-v2-gemma,
  dense retrieval top-K = 20 before cross-encoder scoring.

Requirements:
    uv add FlagEmbedding   (BGE cross-encoder model)
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver
from openai import OpenAI

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_THRESHOLD = 0.85
_DENSE_TOP_K = 20          # primary candidates to retrieve per span
_MAX_SPANS_PER_CHUNK = 10  # cap on span extraction per secondary chunk
_RERANKER_MODEL = "BAAI/bge-reranker-v2-gemma"
_EMBED_DIMS = 1024
_EMBED_BATCH = 10

# CJK quotation mark pairs
_QUOTE_PAIRS: list[tuple[str, str]] = [
    ("「", "」"),
    ("『", "』"),
    ("“", "”"),   # LEFT/RIGHT DOUBLE QUOTATION MARK
    ("‘", "’"),   # LEFT/RIGHT SINGLE QUOTATION MARK
    ("《", "》"),
    ("〈", "〉"),
]

# 省略 / ellipsis markers that may indicate abridged citation
_ELLIPSIS_RE = re.compile(r"[…⋯]{1,3}(?:省略|省)?")

# Minimum span length for a quote to be worth cross-encoding (chars)
_MIN_SPAN_CHARS = 5
# Maximum span length sent to the reranker (chars) — cap to avoid token overflow
_MAX_SPAN_CHARS = 300


# ── Cypher ────────────────────────────────────────────────────────────────────

_SECONDARY_FETCH = """
MATCH (c:CHUNK {tier: 'secondary'})
WHERE c.embedding IS NOT NULL
  AND c.text IS NOT NULL
  AND NOT EXISTS { (c)-[:CITES]->(:CHUNK) }
RETURN c.id AS chunk_id, c.text AS text, c.language AS language
ORDER BY c.id
SKIP $skip LIMIT $batch
"""

_VECTOR_QUERY_PRIMARY = """
CALL db.index.vector.queryNodes('chunk_embedding_classical', $top_k, $embedding)
YIELD node AS c, score
WHERE c.tier = 'primary' AND c.text IS NOT NULL
RETURN c.id AS chunk_id,
       coalesce(c.textCanonical, c.text) AS text,
       score AS vector_score
"""

_CITES_MERGE = """
UNWIND $edges AS e
MATCH (src:CHUNK {id: e.src_id}), (dst:CHUNK {id: e.dst_id})
MERGE (src)-[r:CITES {quoteSpan: e.quote_span}]->(dst)
ON CREATE SET
  r.confidence          = e.confidence,
  r.extractionMethod    = 'entailment',
  r.linkerModelVersion  = e.model_version,
  r.ts                  = e.ts
ON MATCH SET
  r.confidence          = e.confidence,
  r.linkerModelVersion  = e.model_version,
  r.ts                  = e.ts
"""


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class CitesEdge:
    src_id: str
    dst_id: str
    quote_span: str
    confidence: float
    model_version: str
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "quote_span": self.quote_span,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "ts": self.ts,
        }


@dataclass
class LinkerReport:
    chunks_processed: int = 0
    spans_extracted: int = 0
    edges_created: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_processed": self.chunks_processed,
            "spans_extracted": self.spans_extracted,
            "edges_created": self.edges_created,
            "duration_seconds": round(self.duration_seconds, 2),
            "errors": self.errors[:20],
        }


# ── Span extraction ───────────────────────────────────────────────────────────

def _extract_quoted_spans(text: str) -> list[str]:
    """Extract candidate quoted/abridged spans from a secondary chunk.

    Priority: explicit quote marks → …省略 context window → none.
    Returns up to _MAX_SPANS_PER_CHUNK spans within [_MIN_SPAN_CHARS, _MAX_SPAN_CHARS].
    """
    spans: list[str] = []

    # 1. Explicit quote pairs
    for open_q, close_q in _QUOTE_PAIRS:
        pattern = re.escape(open_q) + r"(.+?)" + re.escape(close_q)
        for m in re.finditer(pattern, text, re.DOTALL):
            span = m.group(1).strip()
            if _MIN_SPAN_CHARS <= len(span) <= _MAX_SPAN_CHARS:
                spans.append(span)

    # 2. Context windows around 省略 markers (extract surrounding sentence)
    for m in _ELLIPSIS_RE.finditer(text):
        # grab 100 chars before and after the marker
        start = max(0, m.start() - 100)
        end = min(len(text), m.end() + 100)
        span = text[start:end].strip()
        if _MIN_SPAN_CHARS <= len(span) <= _MAX_SPAN_CHARS:
            spans.append(span)

    # Deduplicate, preserve order, cap at limit
    seen: set[str] = set()
    result: list[str] = []
    for s in spans:
        if s not in seen:
            seen.add(s)
            result.append(s)
        if len(result) >= _MAX_SPANS_PER_CHUNK:
            break

    return result


# ── Reranker ─────────────────────────────────────────────────────────────────

_reranker_cache: Any = None

def _get_reranker():
    global _reranker_cache
    if _reranker_cache is None:
        from FlagEmbedding import FlagReranker  # type: ignore[import]
        _reranker_cache = FlagReranker(_RERANKER_MODEL, use_fp16=True)
    return _reranker_cache


def _score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    """Run BGE-reranker-v2-gemma cross-encoder on (query, passage) pairs."""
    try:
        reranker = _get_reranker()
        raw_scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(raw_scores, float):
            raw_scores = [raw_scores]
        return [float(s) for s in raw_scores]
    except Exception as exc:
        log.warning("Reranker scoring failed: %s", exc)
        return [0.0] * len(pairs)


# ── Embed helper ─────────────────────────────────────────────────────────────

def _embed_texts(texts: list[str], *, model: str) -> list[list[float]] | None:
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.silra.cn/v1/"),
    )
    results: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i : i + _EMBED_BATCH]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(input=batch, model=model)
                results.extend([item.embedding for item in resp.data])
                break
            except Exception as exc:
                if attempt == 2:
                    log.warning("Span embed failed: %s", exc)
                    return None
                time.sleep(2 ** attempt)
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def run_citation_linker(
    driver: Driver,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    dense_top_k: int = _DENSE_TOP_K,
    batch_size: int = 100,
    max_chunks: int | None = None,
    embed_model: str | None = None,
    reranker_model: str = _RERANKER_MODEL,
) -> LinkerReport:
    """Find and write secondary→primary CITES edges.

    For each secondary CHUNK that has an embedding and no CITES edges yet:
      1. Extract quoted/abridged spans from the chunk text.
      2. Embed each span via text-embedding-v4.
      3. Retrieve top-K primary chunk candidates from chunk_embedding_classical ANN index.
      4. Score (span, primary_text) with BGE-reranker-v2-gemma cross-encoder.
      5. MERGE CITES edges above threshold.

    Args:
        driver: Open Neo4j driver.
        threshold: Cross-encoder score ≥ threshold to write a CITES edge.
        dense_top_k: Number of primary candidates to retrieve per span.
        batch_size: Secondary chunks to process per iteration.
        max_chunks: Optional cap for smoke tests.
        embed_model: Embedding model identifier.
        reranker_model: Reranker model identifier (informational, cached globally).
    """
    model = embed_model or os.getenv("EMBED_LLM_MODEL", "text-embedding-v4")
    report = LinkerReport()
    t_start = time.time()
    skip = 0

    while True:
        with driver.session() as s:
            rows = s.run(_SECONDARY_FETCH, skip=skip, batch=batch_size).data()

        if not rows:
            break

        report.chunks_processed += len(rows)
        all_edges: list[CitesEdge] = []
        ts_now = datetime.now(timezone.utc).isoformat()

        for row in rows:
            chunk_id = row["chunk_id"]
            text = row["text"] or ""

            spans = _extract_quoted_spans(text)
            if not spans:
                skip += 0  # no update — counted in batch skip below
                continue

            report.spans_extracted += len(spans)

            # Embed spans
            span_vectors = _embed_texts(spans, model=model)
            if span_vectors is None:
                report.errors.append(f"{chunk_id}: span embed failed")
                continue

            for span, span_vec in zip(spans, span_vectors):
                # Dense retrieval for primary candidates
                try:
                    with driver.session() as s:
                        candidates = s.run(
                            _VECTOR_QUERY_PRIMARY,
                            top_k=dense_top_k,
                            embedding=span_vec,
                        ).data()
                except Exception as exc:
                    log.warning("Vector query failed for chunk %s span: %s", chunk_id, exc)
                    continue

                if not candidates:
                    continue

                # Cross-encoder scoring
                pairs = [(span, c["text"]) for c in candidates]
                scores = _score_pairs(pairs)

                for candidate, score in zip(candidates, scores):
                    if score >= threshold:
                        all_edges.append(CitesEdge(
                            src_id=chunk_id,
                            dst_id=candidate["chunk_id"],
                            quote_span=span[:200],  # cap stored span length
                            confidence=score,
                            model_version=reranker_model,
                            ts=ts_now,
                        ))

        if all_edges:
            edge_dicts = [e.to_dict() for e in all_edges]
            with driver.session() as s:
                s.run(_CITES_MERGE, edges=edge_dicts).consume()
            report.edges_created += len(all_edges)

        log.info(
            "CITES skip=%d chunks=%d spans=%d edges_this_batch=%d total_edges=%d",
            skip, len(rows), report.spans_extracted,
            len(all_edges), report.edges_created,
        )
        skip += len(rows)

        if max_chunks is not None and report.chunks_processed >= max_chunks:
            break

    report.duration_seconds = time.time() - t_start
    return report
