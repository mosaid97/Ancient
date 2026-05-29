"""C3: Two-ribbon output formatter + deterministic verifier wiring (plan §0.6 C3).

Assembles the final search response:
  1. Splits reranked results into two ribbons: primary (古籍原典) + secondary (學術研究)
  2. Runs each result through the A3 deterministic verifier (verify_cite)
  3. Chunks that fail verification get outcome='insufficient_evidence' and are
     flagged in the output — they are NOT suppressed (the user sees the evidence
     badge, not a ghost result)
  4. Returns a SearchResponse with both ribbons + per-result VerifierResult

This is the final gate before results reach the user.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from neo4j import Driver

from apps.backend.agents.verifier import VerifierResult, verify_cite

log = logging.getLogger(__name__)

_CHUNK_TEXT_QUERY = """
MATCH (c:CHUNK {id: $chunk_id})
RETURN
  coalesce(c.textCanonical, c.text) AS text,
  c.tier AS tier,
  c.language AS language,
  c.pageId AS page_id,
  c.documentId AS document_id,
  c.chunkIndex AS chunk_index
"""


@dataclass
class RibbonResult:
    """One result in a search ribbon."""

    chunk_id: str
    text: str
    tier: str | None
    page_id: str | None
    document_id: str | None
    chunk_index: int | None
    retrieval_score: float
    rerank_score: float
    verifier: VerifierResult

    @property
    def verified(self) -> bool:
        return self.verifier.ok

    @property
    def evidence_strength(self) -> str | None:
        return self.verifier.evidence_strength

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunkId": self.chunk_id,
            "text": self.text,
            "tier": self.tier,
            "pageId": self.page_id,
            "documentId": self.document_id,
            "chunkIndex": self.chunk_index,
            "retrievalScore": round(self.retrieval_score, 4),
            "rerankScore": round(self.rerank_score, 4),
            "verified": self.verified,
            "evidenceStrength": self.evidence_strength,
            "verifierOutcome": self.verifier.outcome,
        }


@dataclass
class SearchResponse:
    """Two-ribbon search response with verifier outcomes."""

    query: str
    intent: str
    primary_ribbon: list[RibbonResult] = field(default_factory=list)
    secondary_ribbon: list[RibbonResult] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def all_results(self) -> list[RibbonResult]:
        return self.primary_ribbon + self.secondary_ribbon

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "primaryRibbon": [r.to_dict() for r in self.primary_ribbon],
            "secondaryRibbon": [r.to_dict() for r in self.secondary_ribbon],
            "durationMs": round(self.duration_ms, 1),
            "totalResults": len(self.primary_ribbon) + len(self.secondary_ribbon),
        }


def build_response(
    driver: Driver,
    query: str,
    intent: str,
    reranked: list[Any],  # list[RerankResult] from agents/rerank.py
    *,
    verify_span: str | None = None,
    top_per_ribbon: int = 10,
) -> SearchResponse:
    """Build a two-ribbon verifier-gated SearchResponse.

    Args:
        driver: Open Neo4j driver.
        query: Original user query (used as verification span when verify_span is None).
        intent: Intent classification from agents/intent.py.
        reranked: Reranked candidate list from agents/rerank.py.
        verify_span: The specific span to verify against each chunk. Defaults
            to the query itself (exact-substring check after normalization).
        top_per_ribbon: Max results per ribbon.
    """
    import time
    t0 = time.time()
    span = verify_span or query
    response = SearchResponse(query=query, intent=intent)

    for result in reranked:
        if (
            len(response.primary_ribbon) >= top_per_ribbon
            and len(response.secondary_ribbon) >= top_per_ribbon
        ):
            break

        # Fetch chunk text + metadata
        with driver.session() as s:
            rows = s.run(_CHUNK_TEXT_QUERY, chunk_id=result.chunk_id).data()
        if not rows:
            continue
        row = rows[0]
        tier = row.get("tier")

        if tier == "primary" and len(response.primary_ribbon) >= top_per_ribbon:
            continue
        if tier == "secondary" and len(response.secondary_ribbon) >= top_per_ribbon:
            continue

        # Run deterministic verifier
        verifier_result = verify_cite(driver, result.chunk_id, span)

        ribbon_item = RibbonResult(
            chunk_id=result.chunk_id,
            text=row.get("text") or "",
            tier=tier,
            page_id=row.get("page_id"),
            document_id=row.get("document_id"),
            chunk_index=row.get("chunk_index"),
            retrieval_score=result.retrieval_score,
            rerank_score=result.rerank_score,
            verifier=verifier_result,
        )

        if tier == "primary":
            response.primary_ribbon.append(ribbon_item)
        else:
            response.secondary_ribbon.append(ribbon_item)

    response.duration_ms = (time.time() - t0) * 1000
    return response
