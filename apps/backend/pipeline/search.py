"""Search pipeline — Phase 7 vector similarity search over CHUNK.embedding.

Entry point: :func:`search` — takes a natural-language query, embeds it,
runs a vector index nearest-neighbour query, optionally expands via the
KEYWORD graph, then enriches every hit with the full v2.1 spine context.

Public API
----------
embed_query(text, *, model) -> list[float]
vector_search(driver, query_vector, *, top_k, tier_filter, language_filter,
              score_threshold) -> list[SearchHit]
enrich_with_spine(driver, hits) -> list[SearchResult]
keyword_expand(driver, query_terms, *, top_k) -> list[str]  # returns chunk ids
search(driver, query, *, top_k, tier_filter, language_filter,
       expand_keywords, score_threshold) -> list[SearchResult]

Spine path walked: CHUNK <-[:HAS]- PAGE <-[:INCLUDE]- SECTION
                   <-[:INCLUDE]- CHAPTER <-[:CONSIST_OF]- DOCUMENT
                   <-[:CONTAIN]- TOPIC
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from neo4j import Driver

from apps.backend.llm.silra import embed as silra_embed

log = logging.getLogger(__name__)

_EMBED_DIMS = 1024
_DEFAULT_TOP_K = 10
_DEFAULT_SCORE_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    """Raw result from the vector index query, before spine enrichment."""

    chunk_id: str
    text: str
    char_count: int
    chunk_index: int
    language: str | None
    page_id: str | None
    document_id: str | None
    score: float  # cosine similarity (0–1)
    embedding_model: str | None = None


@dataclass
class SpineContext:
    """Full v2.1 spine context walking from CHUNK up to TOPIC."""

    # PAGE
    page_id: str | None = None
    page_index: int | None = None
    page_language: str | None = None
    page_mode: str | None = None
    page_tier: str | None = None

    # SECTION
    section_id: str | None = None
    section_title: str | None = None
    section_ordinal: int | None = None

    # CHAPTER
    chapter_id: str | None = None
    chapter_title: str | None = None
    chapter_ordinal: int | None = None

    # DOCUMENT
    document_id: str | None = None
    document_title: str | None = None
    document_author: str | None = None
    document_tier: str | None = None
    document_edition: str | None = None
    document_editorial_layers: str | None = None
    document_publication_period: str | None = None
    document_source_path: str | None = None

    # TOPIC
    topic_name: str | None = None

    # KEYWORD neighbours (names of KEYWORD nodes that MENTION this CHUNK)
    keywords: list[str] = field(default_factory=list)

    def citation_label(self) -> str:
        """Return a human-readable citation string for this chunk."""
        parts: list[str] = []
        if self.document_title:
            parts.append(self.document_title)
        if self.chapter_title:
            parts.append(f"卷{self.chapter_ordinal}" if self.chapter_ordinal else self.chapter_title)
        if self.section_title:
            parts.append(self.section_title)
        if self.page_index is not None:
            parts.append(f"p.{self.page_index + 1}")
        return "《" + "·".join(parts) + "》" if parts else "(unknown)"

    def trust_score(self) -> float:
        """Heuristic trust score based on tier + editorial layers (0–1).

        Primary sources score 1.0; secondary scholarly works score 0.7;
        works with paraphrase layers (箋解, 疏議) score slightly lower
        than pure sources.  The verifier (Phase 7b) adds its own LLM
        evidence score on top of this baseline.
        """
        base = 1.0 if self.document_tier == "primary" else 0.7
        layers_json = self.document_editorial_layers or "[]"
        if "箋解" in layers_json or "疏議" in layers_json:
            base *= 0.9
        return round(base, 3)


@dataclass
class SearchResult:
    """A single ranked search hit enriched with spine context."""

    hit: SearchHit
    spine: SpineContext
    rank: int = 0  # 1-based rank in the result list

    @property
    def score(self) -> float:
        return self.hit.score

    @property
    def text(self) -> str:
        return self.hit.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.hit.score, 6),
            "chunk_id": self.hit.chunk_id,
            "text": self.hit.text[:300],
            "char_count": self.hit.char_count,
            "language": self.hit.language,
            "document_title": self.spine.document_title,
            "document_author": self.spine.document_author,
            "document_tier": self.spine.document_tier,
            "chapter_title": self.spine.chapter_title,
            "chapter_ordinal": self.spine.chapter_ordinal,
            "section_title": self.spine.section_title,
            "page_index": self.spine.page_index,
            "topic": self.spine.topic_name,
            "citation": self.spine.citation_label(),
            "trust_score": self.spine.trust_score(),
            "keywords": self.spine.keywords[:10],
        }


# ---------------------------------------------------------------------------
# Embed query
# ---------------------------------------------------------------------------


def embed_query(text: str, *, model: str | None = None) -> list[float]:
    """Embed a single query string using Silra text-embedding-v4.

    Args:
        text: Natural-language query (Chinese or mixed).
        model: Override the embedding model; defaults to ``EMBED_LLM_MODEL``.

    Returns:
        A 1024-dim float list.

    Raises:
        RuntimeError: If the Silra API is unreachable or returns wrong dims.
    """
    embed_model = model or os.getenv("EMBED_LLM_MODEL", "text-embedding-v4")
    vectors = silra_embed(text, model=embed_model)
    if not vectors:
        raise RuntimeError("embed_query: Silra returned empty embedding list")
    vec = vectors[0]
    if len(vec) != _EMBED_DIMS:
        raise RuntimeError(
            f"embed_query: expected {_EMBED_DIMS} dims, got {len(vec)}"
        )
    return vec


# ---------------------------------------------------------------------------
# Cypher — vector search
# ---------------------------------------------------------------------------

# Tier/language filtering is done post-fetch in Python (Neo4j CE vector index
# does not support metadata predicates in the YIELD clause).
_VECTOR_SEARCH_SIMPLE = """
CALL db.index.vector.queryNodes('chunk_embedding_vector_index', $top_k, $query_vector)
YIELD node AS chunk, score
RETURN
    chunk.id            AS chunk_id,
    chunk.text          AS text,
    chunk.charCount     AS char_count,
    chunk.chunkIndex    AS chunk_index,
    chunk.language      AS language,
    chunk.pageId        AS page_id,
    chunk.documentId    AS document_id,
    chunk.embeddingModel AS embedding_model,
    score
ORDER BY score DESC
"""


def vector_search(
    driver: Driver,
    query_vector: list[float],
    *,
    top_k: int = _DEFAULT_TOP_K,
    tier_filter: str | None = None,
    language_filter: str | None = None,
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
) -> list[SearchHit]:
    """Run a nearest-neighbour query against the chunk_embedding_vector_index.

    Args:
        driver: Open Neo4j driver.
        query_vector: 1024-dim float list.
        top_k: Number of neighbours to retrieve from the index.  When
            tier_filter or language_filter is set we over-fetch by 3× and
            prune in Python.
        tier_filter: ``'primary'`` or ``'secondary'`` to restrict results.
        language_filter: e.g. ``'zh-classical'`` to restrict by page language.
        score_threshold: Minimum cosine similarity to include a result (0–1).

    Returns:
        List of :class:`SearchHit`, sorted by score descending, up to top_k.
    """
    fetch_k = top_k * 3 if (tier_filter or language_filter) else top_k
    t0 = time.time()
    with driver.session() as s:
        rows = s.run(
            _VECTOR_SEARCH_SIMPLE, top_k=fetch_k, query_vector=query_vector
        ).data()
    log.info(
        "vector_search: fetched %d rows in %.2fs (fetch_k=%d)",
        len(rows), time.time() - t0, fetch_k,
    )

    hits: list[SearchHit] = []
    for row in rows:
        if row["score"] < score_threshold:
            continue
        hit = SearchHit(
            chunk_id=row["chunk_id"],
            text=row["text"] or "",
            char_count=row["char_count"] or 0,
            chunk_index=row["chunk_index"] or 0,
            language=row["language"],
            page_id=row["page_id"],
            document_id=row["document_id"],
            score=row["score"],
            embedding_model=row["embedding_model"],
        )
        hits.append(hit)

    # Post-fetch filtering — language filter applied here because the vector
    # index does not support metadata predicates in Neo4j CE < 5.23.
    if language_filter:
        hits = [h for h in hits if h.language == language_filter]

    return hits[:top_k]


# ---------------------------------------------------------------------------
# Cypher — spine enrichment
# ---------------------------------------------------------------------------

_SPINE_QUERY = """
UNWIND $chunk_ids AS cid
MATCH (c:CHUNK {id: cid})<-[:HAS]-(p:PAGE)
OPTIONAL MATCH (p)<-[:INCLUDE]-(sec:SECTION)
OPTIONAL MATCH (sec)<-[:INCLUDE]-(ch:CHAPTER)
OPTIONAL MATCH (ch)<-[:CONSIST_OF]-(d:DOCUMENT)
OPTIONAL MATCH (d)<-[:CONTAIN]-(t:TOPIC)
RETURN
    c.id                        AS chunk_id,
    p.id                        AS page_id,
    p.docPageIndex              AS page_index,
    p.language                  AS page_language,
    p.mode                      AS page_mode,
    p.tier                      AS page_tier,
    sec.id                      AS section_id,
    sec.title                   AS section_title,
    sec.ordinal                 AS section_ordinal,
    ch.id                       AS chapter_id,
    ch.title                    AS chapter_title,
    ch.ordinal                  AS chapter_ordinal,
    d.id                        AS document_id,
    d.title                     AS document_title,
    d.primaryAuthor             AS document_author,
    d.tier                      AS document_tier,
    d.edition                   AS document_edition,
    d.editorialLayers           AS document_editorial_layers,
    d.publicationPeriod         AS document_publication_period,
    d.sourcePath                AS document_source_path,
    t.name                      AS topic_name
"""

_KEYWORD_QUERY = """
UNWIND $chunk_ids AS cid
MATCH (c:CHUNK {id: cid})-[:MENTION]->(k:KEYWORD)
RETURN c.id AS chunk_id, collect(k.name) AS keywords
"""

_TIER_FILTER_QUERY = """
UNWIND $chunk_ids AS cid
MATCH (c:CHUNK {id: cid})<-[:HAS]-(p:PAGE)<-[:INCLUDE]-(sec:SECTION)
      <-[:INCLUDE]-(ch:CHAPTER)<-[:CONSIST_OF]-(d:DOCUMENT)
WHERE d.tier = $tier
RETURN c.id AS chunk_id
"""


def enrich_with_spine(
    driver: Driver,
    hits: list[SearchHit],
    *,
    tier_filter: str | None = None,
) -> list[SearchResult]:
    """Walk each hit up the v2.1 spine and attach KEYWORD neighbours.

    Performs two batched queries:
    1. ``_SPINE_QUERY`` — returns all spine context rows in one round-trip.
    2. ``_KEYWORD_QUERY`` — returns keyword names per chunk in one round-trip.

    Args:
        driver: Open Neo4j driver.
        hits: Raw :class:`SearchHit` list from :func:`vector_search`.
        tier_filter: If set, drop results whose DOCUMENT.tier doesn't match.

    Returns:
        List of :class:`SearchResult` with full spine and keyword context.
    """
    if not hits:
        return []

    chunk_ids = [h.chunk_id for h in hits]
    hit_by_id = {h.chunk_id: h for h in hits}

    # --- spine ---
    with driver.session() as s:
        spine_rows = s.run(_SPINE_QUERY, chunk_ids=chunk_ids).data()

    spine_by_chunk: dict[str, SpineContext] = {}
    for row in spine_rows:
        cid = row["chunk_id"]
        ctx = SpineContext(
            page_id=row.get("page_id"),
            page_index=row.get("page_index"),
            page_language=row.get("page_language"),
            page_mode=row.get("page_mode"),
            page_tier=row.get("page_tier"),
            section_id=row.get("section_id"),
            section_title=row.get("section_title"),
            section_ordinal=row.get("section_ordinal"),
            chapter_id=row.get("chapter_id"),
            chapter_title=row.get("chapter_title"),
            chapter_ordinal=row.get("chapter_ordinal"),
            document_id=row.get("document_id"),
            document_title=row.get("document_title"),
            document_author=row.get("document_author"),
            document_tier=row.get("document_tier"),
            document_edition=row.get("document_edition"),
            document_editorial_layers=row.get("document_editorial_layers"),
            document_publication_period=row.get("document_publication_period"),
            document_source_path=row.get("document_source_path"),
            topic_name=row.get("topic_name"),
        )
        spine_by_chunk[cid] = ctx

    # --- keywords ---
    with driver.session() as s:
        kw_rows = s.run(_KEYWORD_QUERY, chunk_ids=chunk_ids).data()

    for row in kw_rows:
        cid = row["chunk_id"]
        if cid in spine_by_chunk:
            spine_by_chunk[cid].keywords = row.get("keywords", [])

    # --- assemble results ---
    results: list[SearchResult] = []
    for hit in hits:
        spine = spine_by_chunk.get(hit.chunk_id, SpineContext())

        # Tier filtering: if requested, skip mismatched docs.
        if tier_filter and spine.document_tier and spine.document_tier != tier_filter:
            continue

        results.append(SearchResult(hit=hit, spine=spine))

    for i, r in enumerate(results, 1):
        r.rank = i

    return results


# ---------------------------------------------------------------------------
# Keyword-graph expansion
# ---------------------------------------------------------------------------

_KW_EXPAND_QUERY = """
MATCH (k:KEYWORD)
WHERE any(term IN $terms WHERE k.name CONTAINS term)
WITH k ORDER BY k.frequency DESC LIMIT $top_kw
MATCH (c:CHUNK)-[:MENTION]->(k)
WHERE c.embeddingStatus = 'ok'
RETURN DISTINCT c.id AS chunk_id, c.pageId AS page_id,
       c.documentId AS document_id, c.language AS language,
       c.charCount AS char_count
LIMIT $max_chunks
"""


def keyword_expand(
    driver: Driver,
    query_terms: list[str],
    *,
    top_kw: int = 5,
    max_chunks: int = 20,
) -> list[str]:
    """Return chunk ids reachable from KEYWORD nodes matching query terms.

    Useful for hybrid search: adds graph-traversal hits to complement pure
    vector results.

    Args:
        driver: Open Neo4j driver.
        query_terms: List of CJK terms extracted from the user query.
        top_kw: Maximum number of matching KEYWORD nodes to expand from.
        max_chunks: Maximum number of additional chunk ids to return.

    Returns:
        List of chunk ids (may overlap with vector search results; caller
        deduplicates).
    """
    if not query_terms:
        return []
    with driver.session() as s:
        rows = s.run(
            _KW_EXPAND_QUERY, terms=query_terms, top_kw=top_kw, max_chunks=max_chunks
        ).data()
    return [r["chunk_id"] for r in rows]


def _extract_cjk_terms(text: str) -> list[str]:
    """Extract individual CJK characters and 2–4-gram sequences from text."""
    # Single CJK blocks
    cjk_re = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+")
    tokens: list[str] = []
    for m in cjk_re.finditer(text):
        seg = m.group()
        # Add 2–4 grams
        for n in (2, 3, 4):
            for i in range(len(seg) - n + 1):
                tokens.append(seg[i : i + n])
    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result[:30]


# ---------------------------------------------------------------------------
# Main search entry point
# ---------------------------------------------------------------------------


def search(
    driver: Driver,
    query: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
    tier_filter: str | None = None,
    language_filter: str | None = None,
    expand_keywords: bool = True,
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
    embed_model: str | None = None,
) -> list[SearchResult]:
    """Full Phase 7 search: embed → vector index → optional KW expand → spine enrich.

    Args:
        driver: Open Neo4j driver.
        query: Natural-language query in Chinese (or mixed).
        top_k: Number of results to return after dedup + spine enrichment.
        tier_filter: ``'primary'`` or ``'secondary'`` to restrict to one tier.
        language_filter: e.g. ``'zh-classical'`` to restrict by PAGE language.
        expand_keywords: If True, augment vector hits with KEYWORD graph
            expansion hits (up to top_k additional chunk ids).
        score_threshold: Minimum cosine similarity to include a result.
        embed_model: Override embedding model.

    Returns:
        List of :class:`SearchResult` ranked by cosine similarity, enriched
        with full spine context and KEYWORD neighbours.
    """
    t_start = time.time()

    # 1. Embed the query
    try:
        query_vec = embed_query(query, model=embed_model)
    except Exception as exc:
        log.error("search: embed_query failed: %s", exc)
        raise

    # 2. Vector search
    hits = vector_search(
        driver,
        query_vec,
        top_k=top_k,
        tier_filter=None,  # tier filter applied in enrich_with_spine
        language_filter=language_filter,
        score_threshold=score_threshold,
    )
    log.info("search: vector_search returned %d hits", len(hits))

    # 3. Keyword graph expansion
    if expand_keywords and hits:
        terms = _extract_cjk_terms(query)
        extra_ids = keyword_expand(driver, terms, top_kw=5, max_chunks=top_k)
        existing_ids: set[str] = {h.chunk_id for h in hits}
        new_ids = [cid for cid in extra_ids if cid not in existing_ids]
        if new_ids:
            extra_hits = _fetch_chunks_by_ids(driver, new_ids, base_score=0.0)
            hits.extend(extra_hits)
            log.info(
                "search: keyword_expand added %d extra chunk ids (%d new hits)",
                len(extra_ids),
                len(extra_hits),
            )

    # 4. Spine enrichment
    results = enrich_with_spine(driver, hits, tier_filter=tier_filter)

    # Re-sort after merge (keyword-expanded hits have score=0 and will sort last)
    results.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(results[:top_k], 1):
        r.rank = i

    elapsed = time.time() - t_start
    log.info(
        "search: query=%r returned %d results in %.2fs",
        query[:40],
        len(results[:top_k]),
        elapsed,
    )
    return results[:top_k]


# ---------------------------------------------------------------------------
# Helper: fetch chunks by explicit id list (for KW expansion)
# ---------------------------------------------------------------------------

_CHUNK_BY_IDS_QUERY = """
UNWIND $chunk_ids AS cid
MATCH (c:CHUNK {id: cid})
WHERE c.embeddingStatus = 'ok'
RETURN
    c.id            AS chunk_id,
    c.text          AS text,
    c.charCount     AS char_count,
    c.chunkIndex    AS chunk_index,
    c.language      AS language,
    c.pageId        AS page_id,
    c.documentId    AS document_id,
    c.embeddingModel AS embedding_model
"""


def _fetch_chunks_by_ids(
    driver: Driver, chunk_ids: list[str], base_score: float = 0.0
) -> list[SearchHit]:
    if not chunk_ids:
        return []
    with driver.session() as s:
        rows = s.run(_CHUNK_BY_IDS_QUERY, chunk_ids=chunk_ids).data()
    return [
        SearchHit(
            chunk_id=r["chunk_id"],
            text=r["text"] or "",
            char_count=r["char_count"] or 0,
            chunk_index=r["chunk_index"] or 0,
            language=r["language"],
            page_id=r["page_id"],
            document_id=r["document_id"],
            score=base_score,
            embedding_model=r["embedding_model"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# C1–C3: Hybrid search (BM25 + dense + community + RRF + rerank + verifier)
# ---------------------------------------------------------------------------

def hybrid_search(
    driver: Driver,
    query: str,
    bm25_corpus: "BM25Corpus | None" = None,  # type: ignore[name-defined]
    *,
    top_k: int = 10,
    rerank_top_k: int = 20,
    use_community: bool = False,
    verify_span: str | None = None,
) -> "SearchResponse":  # type: ignore[name-defined]
    """C1–C3 hybrid pipeline via LangGraph: intent → HyDE → retrieve → RRF → rerank → verify.

    The pipeline is orchestrated by :data:`apps.backend.agents.search_graph.hybrid_search_graph`.
    HyDE is generated only for ``factual`` / ``interpretive`` queries (conditional edge).
    All other steps (BM25 + dense + community retrieval, RRF fusion, tier boost,
    cross-encoder rerank, verifier + ribbon builder) are sequential graph nodes.

    Args:
        driver: Open Neo4j driver.
        query: Natural-language query.
        bm25_corpus: Pre-built BM25Corpus; if None the BM25 leg is skipped.
        top_k: Results per tier ribbon in the final output.
        rerank_top_k: Candidates to send to the cross-encoder (top-100 → top-k).
        use_community: Enable community-summary routing leg.
        verify_span: Override span for the verifier (defaults to query).

    Returns:
        :class:`~apps.backend.agents.output.SearchResponse` with two tier ribbons.
    """
    import time as _time
    from apps.backend.agents.search_graph import hybrid_search_graph

    t0 = _time.time()

    initial_state = {
        "query": query,
        "driver": driver,
        "bm25_corpus": bm25_corpus,
        "top_k": top_k,
        "rerank_top_k": rerank_top_k,
        "use_community": use_community,
        "verify_span": verify_span,
        # node outputs — populated by the graph
        "intent": "",
        "hyde_passage": None,
        "ranked_lists": [],
        "tier_map": {},
        "fused": [],
        "boosted": [],
        "text_map": {},
        "candidates": [],
        "reranked": [],
        "response": None,
    }

    final_state = hybrid_search_graph.invoke(initial_state)

    response = final_state["response"]
    if response is None:
        from apps.backend.agents.output import SearchResponse
        response = SearchResponse(query=query, intent=final_state.get("intent", ""))

    response.duration_ms = (_time.time() - t0) * 1000
    log.info(
        "hybrid_search (graph): primary=%d secondary=%d in %.1fms",
        len(response.primary_ribbon),
        len(response.secondary_ribbon),
        response.duration_ms,
    )
    return response


# Type alias for forward references
try:
    from apps.backend.retrieval.bm25 import BM25Corpus  # noqa: F401
    from apps.backend.agents.output import SearchResponse  # noqa: F401
except ImportError:
    pass
