"""Search API router — exposes both hybrid (C1-C3) and dense-only search."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from neo4j import Driver

from apps.backend.api.deps import get_driver
from apps.backend.pipeline.search import (
    SearchHit,
    SearchResult,
    enrich_with_spine,
    search as dense_search_fn,
)

log = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_translation(chunk_ids: list[str], driver: Driver) -> dict[str, dict]:
    """Batch-fetch textCanonical + textVernacular for a list of chunk IDs."""
    if not chunk_ids:
        return {}
    try:
        with driver.session() as s:
            rows = s.run(
                "UNWIND $ids AS cid "
                "MATCH (c:CHUNK {id: cid}) "
                "RETURN c.id AS id, c.textCanonical AS tc, c.textVernacular AS tv",
                ids=chunk_ids,
            ).data()
        return {r["id"]: {"translation_canonical": r["tc"], "translation_vernacular": r["tv"]}
                for r in rows}
    except Exception:
        return {}


def _spine_result_to_dict(r: SearchResult, tx: dict) -> dict[str, Any]:
    d = r.to_dict()
    d["keywords"] = r.spine.keywords
    d["editorial_layers"] = r.spine.document_editorial_layers
    d["trust_score"] = r.spine.trust_score()
    d["page_mode"] = r.spine.page_mode
    d.update(tx.get(r.hit.chunk_id, {"translation_canonical": None, "translation_vernacular": None}))
    # Normalise field names
    d.setdefault("document_tier", r.spine.document_tier)
    d["verified"] = False
    d["evidence_strength"] = None
    d["verifier_outcome"] = None
    d["intent"] = None
    d["rerank_score"] = None
    return d


def _ribbon_to_dict(
    ribbon_result: Any,
    spine_map: dict[str, SearchResult],
    tx_map: dict[str, dict],
    intent: str,
) -> dict[str, Any]:
    """Convert a RibbonResult + spine enrichment into the flat frontend format."""
    chunk_id = ribbon_result.chunk_id
    spine_r = spine_map.get(chunk_id)
    if spine_r:
        d = _spine_result_to_dict(spine_r, tx_map)
    else:
        # Fallback: minimal dict from ribbon result
        d = {
            "chunk_id": chunk_id,
            "text": ribbon_result.text,
            "char_count": len(ribbon_result.text or ""),
            "document_tier": ribbon_result.tier,
            "score": ribbon_result.retrieval_score,
            "citation": "(unknown)",
            "trust_score": 1.0 if ribbon_result.tier == "primary" else 0.7,
            "keywords": [],
            "editorial_layers": None,
            "translation_canonical": None,
            "translation_vernacular": None,
            "page_mode": None,
        }
    # Augment with hybrid-specific fields
    d["score"] = ribbon_result.retrieval_score
    d["rerank_score"] = ribbon_result.rerank_score
    d["verified"] = ribbon_result.verified
    d["evidence_strength"] = ribbon_result.evidence_strength
    d["verifier_outcome"] = ribbon_result.verifier.outcome
    d["intent"] = intent
    return d


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("")
async def search_endpoint(
    request: Request,
    q: str = Query(..., description="Search query (CJK supported)"),
    mode: str = Query("hybrid", description="hybrid | dense"),
    tier: str = Query("both", description="both | primary | secondary"),
    lang: str = Query("all", description="all | zh-classical | zh-modern | ja-kanbun | ja-modern"),
    top_k: int = Query(10, ge=1, le=50),
    expand_keywords: bool = Query(True),
    driver: Driver = Depends(get_driver),
) -> dict[str, Any]:
    tier_filter = None if tier == "both" else tier
    lang_filter = None if lang == "all" else lang

    # ── Hybrid mode (C1-C3 pipeline) ─────────────────────────────────────────
    if mode == "hybrid":
        try:
            from apps.backend.pipeline.search import hybrid_search
            bm25 = getattr(request.app.state, "bm25_corpus", None)
            if bm25 is None:
                log.info("BM25Corpus not ready yet — running dense-only")

            resp = hybrid_search(driver, q, bm25, top_k=top_k)
        except Exception as exc:
            log.exception("hybrid_search failed, falling back to dense: %s", exc)
            mode = "dense"  # fall through to dense mode below
        else:
            # Gather all chunk_ids for spine enrichment
            all_ribbon = resp.primary_ribbon + resp.secondary_ribbon
            chunk_ids = [r.chunk_id for r in all_ribbon]

            # Enrich with spine (document title, chapter, etc.)
            fake_hits = []
            for rr in all_ribbon:
                fake_hits.append(SearchHit(
                    chunk_id=rr.chunk_id,
                    text=rr.text or "",
                    char_count=len(rr.text or ""),
                    chunk_index=rr.chunk_index or 0,
                    language=lang_filter,
                    page_id=rr.page_id,
                    document_id=rr.document_id,
                    score=rr.retrieval_score,
                ))
            try:
                enriched = enrich_with_spine(driver, fake_hits)
            except Exception:
                enriched = []
            spine_map = {r.hit.chunk_id: r for r in enriched}

            # Batch-fetch translations
            tx_map = _fetch_translation(chunk_ids, driver)

            primary = [_ribbon_to_dict(rr, spine_map, tx_map, resp.intent)
                       for rr in resp.primary_ribbon]
            secondary = [_ribbon_to_dict(rr, spine_map, tx_map, resp.intent)
                         for rr in resp.secondary_ribbon]

            # Apply tier filter if requested
            if tier_filter:
                primary = [d for d in primary if d.get("document_tier") == tier_filter]
                secondary = [d for d in secondary if d.get("document_tier") == tier_filter]

            results = primary + secondary
            for i, r in enumerate(results):
                r.setdefault("rank", i + 1)

            return {
                "query": q,
                "mode": "hybrid",
                "intent": resp.intent,
                "bm25_ready": bm25 is not None,
                "total": len(results),
                "primary_count": len(primary),
                "secondary_count": len(secondary),
                "results": results,
                "primaryRibbon": primary,
                "secondaryRibbon": secondary,
                "duration_ms": resp.duration_ms,
            }

    # ── Dense-only fallback ───────────────────────────────────────────────────
    try:
        results_raw = dense_search_fn(
            driver, q,
            top_k=top_k,
            tier_filter=tier_filter,
            language_filter=lang_filter,
            expand_keywords=expand_keywords,
        )
    except Exception as exc:
        log.exception("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    chunk_ids = [r.hit.chunk_id for r in results_raw]
    tx_map = _fetch_translation(chunk_ids, driver)
    serialized = [_spine_result_to_dict(r, tx_map) for r in results_raw]

    primary = [d for d in serialized if d.get("document_tier") == "primary"]
    secondary = [d for d in serialized if d.get("document_tier") != "primary"]

    return {
        "query": q,
        "mode": "dense",
        "intent": None,
        "bm25_ready": getattr(request.app.state, "bm25_corpus", None) is not None,
        "total": len(serialized),
        "primary_count": len(primary),
        "secondary_count": len(secondary),
        "results": serialized,
        "primaryRibbon": primary,
        "secondaryRibbon": secondary,
        "duration_ms": None,
    }
