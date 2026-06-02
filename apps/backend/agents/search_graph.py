"""LangGraph StateGraph for the hybrid C1-C3 search pipeline.

Graph topology:
    START → intent → (hyde | retrieve) → retrieve → fuse
          → tier_boost → rerank → output → END

Conditional edge after ``intent``:
    factual / interpretive → hyde → retrieve
    everything else        → retrieve

Each node is a pure function; the mutable state flows between them via the
typed ``HybridSearchState`` dict.
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class HybridSearchState(TypedDict):
    # ── required inputs ─────────────────────────────────────────────────────
    query: str
    driver: Any
    bm25_corpus: Any            # BM25Corpus | None
    top_k: int
    rerank_top_k: int
    use_community: bool
    verify_span: str | None
    # ── populated by nodes ──────────────────────────────────────────────────
    intent: str
    hyde_passage: str | None
    ranked_lists: list          # list of [(chunk_id, score)]
    tier_map: dict              # chunk_id → tier
    fused: list                 # [(chunk_id, rrf_score)]
    boosted: list               # [(chunk_id, boosted_score)]
    text_map: dict              # chunk_id → canonical/raw text
    candidates: list            # [(chunk_id, text, score)]
    reranked: list              # list[RerankResult]
    response: Any               # SearchResponse | None


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def node_intent(state: HybridSearchState) -> dict:
    from apps.backend.agents.intent import classify_intent
    intent = classify_intent(state["query"])
    log.info("search_graph intent=%s query=%r", intent, state["query"][:50])
    return {"intent": intent}


def node_hyde(state: HybridSearchState) -> dict:
    from apps.backend.agents.hyde import generate_hyde_passage
    hyde = generate_hyde_passage(state["query"])
    log.info("search_graph hyde generated (%d chars)", len(hyde or ""))
    return {"hyde_passage": hyde}


def node_retrieve(state: HybridSearchState) -> dict:
    from apps.backend.retrieval.dense import dense_search
    from apps.backend.retrieval.community import community_search

    query = state["query"]
    bm25 = state["bm25_corpus"]
    driver = state["driver"]
    intent = state["intent"]
    hyde = state.get("hyde_passage")
    use_community = state["use_community"]

    ranked_lists: list = []

    # BM25 leg
    if bm25 is not None:
        bm25_hits = bm25.query(query, top_k=100)
        if bm25_hits:
            ranked_lists.append(bm25_hits)

    # Dense leg — original query
    dense_hits = dense_search(driver, query, top_k=100)
    if dense_hits:
        ranked_lists.append(dense_hits)

    # Dense leg — HyDE passage
    if hyde:
        hyde_hits = dense_search(driver, hyde, top_k=50)
        if hyde_hits:
            ranked_lists.append(hyde_hits)

    # Community leg
    if use_community and intent == "synthesis":
        comm_hits = community_search(driver, query, top_k=5)
        if comm_hits:
            ranked_lists.append(comm_hits)

    log.info("search_graph retrieve: %d legs", len(ranked_lists))
    return {"ranked_lists": ranked_lists}


def node_fuse(state: HybridSearchState) -> dict:
    from apps.backend.retrieval.fuse import rrf_fuse

    ranked_lists = state["ranked_lists"]
    if not ranked_lists:
        log.warning("search_graph fuse: all retrieval legs empty")
        return {"fused": []}

    fused = rrf_fuse(ranked_lists, top_k=100)
    return {"fused": fused}


def node_tier_boost(state: HybridSearchState) -> dict:
    from apps.backend.agents.intent import tier_boost

    fused = state.get("fused") or []
    if not fused:
        return {"boosted": [], "tier_map": {}}

    driver = state["driver"]
    chunk_ids = [cid for cid, _ in fused]
    tier_map: dict[str, str | None] = {}
    if chunk_ids:
        with driver.session() as s:
            rows = s.run(
                "UNWIND $ids AS cid MATCH (c:CHUNK {id: cid}) RETURN c.id AS id, c.tier AS tier",
                ids=chunk_ids,
            ).data()
        for r in rows:
            tier_map[r["id"]] = r.get("tier")

    boosted = tier_boost(fused, tier_map, state["intent"])
    boosted.sort(key=lambda x: x[1], reverse=True)
    return {"tier_map": tier_map, "boosted": boosted}


def node_rerank(state: HybridSearchState) -> dict:
    from apps.backend.agents.rerank import rerank as rerank_fn, RerankResult

    boosted = state.get("boosted") or []
    query = state["query"]
    rerank_top_k = state["rerank_top_k"]
    driver = state["driver"]

    rerank_candidates_raw = boosted[:100]
    rerank_ids = [cid for cid, _ in rerank_candidates_raw]
    score_map = dict(rerank_candidates_raw)

    text_rows: list[dict] = []
    if rerank_ids:
        with driver.session() as s:
            text_rows = s.run(
                "UNWIND $ids AS cid MATCH (c:CHUNK {id: cid}) "
                "RETURN c.id AS id, coalesce(c.textCanonical, c.text) AS text",
                ids=rerank_ids,
            ).data()
    text_map = {r["id"]: r["text"] or "" for r in text_rows}

    candidates = [
        (cid, text_map.get(cid, ""), score_map[cid])
        for cid in rerank_ids
        if text_map.get(cid)
    ]

    try:
        reranked = rerank_fn(query, candidates, top_k=rerank_top_k)
    except Exception as exc:
        log.warning("search_graph reranker failed, using retrieval order: %s", exc)
        reranked = [
            RerankResult(chunk_id=cid, retrieval_score=s, rerank_score=s, rank=i + 1)
            for i, (cid, _, s) in enumerate(candidates[:rerank_top_k])
        ]

    return {"text_map": text_map, "candidates": candidates, "reranked": reranked}


def node_output(state: HybridSearchState) -> dict:
    from apps.backend.agents.output import build_response

    response = build_response(
        state["driver"],
        state["query"],
        state["intent"],
        state["reranked"],
        verify_span=state.get("verify_span"),
        top_per_ribbon=state["top_k"],
    )
    return {"response": response}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def _route_after_intent(state: HybridSearchState) -> str:
    """Generate a HyDE passage only for factual or interpretive queries."""
    if state.get("intent") in ("factual", "interpretive"):
        return "hyde"
    return "retrieve"


# ---------------------------------------------------------------------------
# Graph construction (compiled once at import time)
# ---------------------------------------------------------------------------


def _build_graph() -> Any:
    builder: StateGraph = StateGraph(HybridSearchState)

    builder.add_node("intent", node_intent)
    builder.add_node("hyde", node_hyde)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("fuse", node_fuse)
    builder.add_node("tier_boost", node_tier_boost)
    builder.add_node("rerank", node_rerank)
    builder.add_node("output", node_output)

    builder.add_edge(START, "intent")
    builder.add_conditional_edges(
        "intent",
        _route_after_intent,
        {"hyde": "hyde", "retrieve": "retrieve"},
    )
    builder.add_edge("hyde", "retrieve")
    builder.add_edge("retrieve", "fuse")
    builder.add_edge("fuse", "tier_boost")
    builder.add_edge("tier_boost", "rerank")
    builder.add_edge("rerank", "output")
    builder.add_edge("output", END)

    return builder.compile()


hybrid_search_graph = _build_graph()
