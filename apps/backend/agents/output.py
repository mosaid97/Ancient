"""C3: Two-ribbon output formatter + deterministic verifier wiring (plan §0.6 C3).

Assembles the final search response:
  1. Splits reranked results into two ribbons: primary (古籍原典) + secondary (學術研究)
  2. Extracts a verbatim quotable span from the query (or accepts an explicit
     ``verify_span``). If the query is a natural-language question and yields
     no quotable Han run, verification is skipped (outcome='not_applicable');
     the result is shown with ``verified=False``.
  3. Runs each result through the A3 deterministic verifier (verify_cite)
  4. Optionally **gates** results: when ``gate_unverified=True`` (default),
     chunks whose verifier outcome is ``'insufficient_evidence'`` are
     dropped from the ribbon — they do not reach the user. Translation-only
     matches and "not_applicable" results are kept (they aren't false
     citations, just softer signals).
  5. Returns a SearchResponse with both ribbons + per-result VerifierResult.

This is the final gate before results reach the user.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from neo4j import Driver

from apps.backend.agents.verifier import (
    MIN_SPAN_CHARS,
    VerifierResult,
    extract_quotable_span,
    verify_cite,
)
from apps.backend.api import verifier_metrics

log = logging.getLogger(__name__)

_CHUNK_SPINE_QUERY = """
MATCH (c:CHUNK {id: $chunk_id})
OPTIONAL MATCH (c)<-[:HAS]-(p:PAGE)
OPTIONAL MATCH (p)<-[:INCLUDE]-(sec:SECTION)
OPTIONAL MATCH (sec)<-[:INCLUDE]-(ch:CHAPTER)
OPTIONAL MATCH (ch)<-[:CONSIST_OF]-(d:DOCUMENT)
OPTIONAL MATCH (d)<-[:CONTAIN]-(t:TOPIC)
RETURN
  coalesce(c.textCanonical, c.text) AS text,
  c.tier                            AS tier,
  c.language                        AS language,
  c.pageId                          AS page_id,
  c.documentId                      AS document_id,
  c.chunkIndex                      AS chunk_index,
  p.docPageIndex                    AS page_index,
  p.mode                            AS page_mode,
  sec.title                         AS section_title,
  ch.title                          AS chapter_title,
  ch.ordinal                        AS chapter_ordinal,
  d.title                           AS document_title,
  d.primaryAuthor                   AS document_author,
  d.edition                         AS document_edition,
  d.publicationPeriod               AS document_publication_period,
  t.name                            AS topic
"""


def _not_applicable_result(chunk_id: str) -> VerifierResult:
    """Synthesize a 'not_applicable' VerifierResult for queries with no
    extractable verbatim span. The result is still shown, but the badge
    correctly reflects that no source-substring check ran."""
    return VerifierResult(
        chunk_id=chunk_id,
        span="",
        outcome="not_applicable",
        failure_mode=None,
        evidence_strength=None,
    )


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
    # Source provenance — populated from full spine walk
    page_index: int | None = None
    page_mode: str | None = None
    section_title: str | None = None
    chapter_title: str | None = None
    chapter_ordinal: int | None = None
    document_title: str | None = None
    document_author: str | None = None
    document_edition: str | None = None
    document_publication_period: str | None = None
    topic: str | None = None

    @property
    def verified(self) -> bool:
        return self.verifier.ok

    @property
    def evidence_strength(self) -> str | None:
        return self.verifier.evidence_strength

    @property
    def citation(self) -> str:
        parts: list[str] = []
        if self.document_title:
            parts.append(self.document_title)
        if self.chapter_title:
            label = f"卷{self.chapter_ordinal}" if self.chapter_ordinal else self.chapter_title
            parts.append(label)
        if self.section_title:
            parts.append(self.section_title)
        if self.page_index is not None:
            parts.append(f"p.{self.page_index + 1}")
        return "《" + "·".join(parts) + "》" if parts else "(unknown)"

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
            "verifierMatchedIn": self.verifier.matched_in,
            "verifierSpan": self.verifier.span or None,
            # Source provenance
            "citation": self.citation,
            "documentTitle": self.document_title,
            "documentAuthor": self.document_author,
            "documentEdition": self.document_edition,
            "documentPublicationPeriod": self.document_publication_period,
            "chapterTitle": self.chapter_title,
            "chapterOrdinal": self.chapter_ordinal,
            "sectionTitle": self.section_title,
            "pageIndex": self.page_index,
            "pageMode": self.page_mode,
            "topic": self.topic,
        }


@dataclass
class SearchResponse:
    """Two-ribbon search response with verifier outcomes."""

    query: str
    intent: str
    primary_ribbon: list[RibbonResult] = field(default_factory=list)
    secondary_ribbon: list[RibbonResult] = field(default_factory=list)
    duration_ms: float = 0.0
    verify_span: str | None = None
    gated: bool = True
    gated_count: int = 0

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
            "verifySpan": self.verify_span,
            "gated": self.gated,
            "gatedCount": self.gated_count,
        }


def build_response(
    driver: Driver,
    query: str,
    intent: str,
    reranked: list[Any],  # list[RerankResult] from agents/rerank.py
    *,
    verify_span: str | None = None,
    top_per_ribbon: int = 10,
    gate_unverified: bool = True,
    min_span_chars: int = MIN_SPAN_CHARS,
) -> SearchResponse:
    """Build a two-ribbon verifier-gated SearchResponse.

    Args:
        driver: Open Neo4j driver.
        query: Original user query.
        intent: Intent classification from agents/intent.py.
        reranked: Reranked candidate list from agents/rerank.py.
        verify_span: Explicit span to verify against each chunk. When None,
            :func:`extract_quotable_span` pulls the longest contiguous Han
            run from ``query``; if that yields nothing of length
            ``>= min_span_chars`` the verifier is skipped (outcome
            'not_applicable') and results are NOT gated on verification.
        top_per_ribbon: Max results per ribbon.
        gate_unverified: When True (default), drop ribbon items whose
            verifier outcome is ``'insufficient_evidence'`` — they fail
            the source-presence check and would mislead the user. Set
            False for debugging / evaluation harnesses that want to see
            rejected candidates.
        min_span_chars: Minimum length for the auto-extracted quotable
            span. Forwarded to the verifier as well.
    """
    import time
    t0 = time.time()

    explicit_span = verify_span is not None
    span = verify_span if explicit_span else extract_quotable_span(
        query, min_chars=min_span_chars
    )
    response = SearchResponse(
        query=query,
        intent=intent,
        verify_span=span,
        gated=gate_unverified,
    )

    if span is None:
        log.info(
            "build_response: no verbatim span in query=%r — verification skipped",
            query[:60],
        )

    for result in reranked:
        if (
            len(response.primary_ribbon) >= top_per_ribbon
            and len(response.secondary_ribbon) >= top_per_ribbon
        ):
            break

        # Fetch chunk text + full spine provenance in one round-trip
        with driver.session() as s:
            rows = s.run(_CHUNK_SPINE_QUERY, chunk_id=result.chunk_id).data()
        if not rows:
            continue
        row = rows[0]
        tier = row.get("tier")

        if tier == "primary" and len(response.primary_ribbon) >= top_per_ribbon:
            continue
        if tier == "secondary" and len(response.secondary_ribbon) >= top_per_ribbon:
            continue

        # Verify when we have a quotable span; otherwise mark not_applicable.
        if span is None:
            verifier_result = _not_applicable_result(result.chunk_id)
        else:
            verifier_result = verify_cite(
                driver,
                result.chunk_id,
                span,
                min_span_chars=min_span_chars,
            )

        verifier_metrics.record(
            verifier_result.outcome,
            chunk_id=verifier_result.chunk_id,
            failure_mode=verifier_result.failure_mode,
            span=verifier_result.span,
            tier=verifier_result.tier,
        )

        # Gate: drop chunks that explicitly failed source-presence check.
        # 'translation_match' and 'not_applicable' are kept (they are not
        # hallucinations — translation is a softer signal, not_applicable
        # means the query had no verbatim span to check against).
        if gate_unverified and verifier_result.outcome == "insufficient_evidence":
            response.gated_count += 1
            continue

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
            page_index=row.get("page_index"),
            page_mode=row.get("page_mode"),
            section_title=row.get("section_title"),
            chapter_title=row.get("chapter_title"),
            chapter_ordinal=row.get("chapter_ordinal"),
            document_title=row.get("document_title"),
            document_author=row.get("document_author"),
            document_edition=row.get("document_edition"),
            document_publication_period=row.get("document_publication_period"),
            topic=row.get("topic"),
        )

        if tier == "primary":
            response.primary_ribbon.append(ribbon_item)
        else:
            response.secondary_ribbon.append(ribbon_item)

    response.duration_ms = (time.time() - t0) * 1000
    if response.gated_count:
        log.info(
            "build_response: gated %d unverified results (span=%r)",
            response.gated_count,
            span,
        )
    return response
