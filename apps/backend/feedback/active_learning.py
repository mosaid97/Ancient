"""Active-learning priority scorer — Phase 5 (plan §6 Stage 5, Move 2).

Ranks ``NEEDS_REVIEW`` pages by the **canonical priority formula** from
plan §2.5 Move 2:

    score = α·uncertainty + β·downstream_impact_norm + γ·tier_weight

where:
- ``uncertainty  = (1 - fusion_agreement_rate) + λ·llm_coherence_variance``
- ``downstream_impact`` = linked chunk count + linked keyword count
- ``downstream_impact_norm`` = impact / max_impact_in_batch (0–1)
- ``tier_weight(primary) = 1.0``, ``tier_weight(secondary) = 0.3``
- Default weights: α=0.40, β=0.40, γ=0.20, λ=0.30

The formula is the SINGLE canonical definition — referenced by §2.5,
§6 Stage 5, §6 Stage 11, and any future phase that re-ranks the queue.

Public API
----------
PriorityWeights         — weight dataclass
compute_priority_score(page, weights) -> float
get_review_queue(driver, *, max_pages, weights) -> list[ReviewQueueItem]
update_priority_scores(driver, *, batch_size, weights) -> int
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weights dataclass
# ---------------------------------------------------------------------------


@dataclass
class PriorityWeights:
    """Weights for the canonical priority formula (plan §2.5 Move 2).

    Attributes:
        alpha: Weight for uncertainty (inter-engine disagreement).
        beta: Weight for downstream_impact (graph centrality proxy).
        gamma: Weight for tier_weight (primary > secondary).
        lambda_: Coefficient multiplying LLM coherence variance within
            the uncertainty term.
        tier_primary: Tier weight for primary documents.
        tier_secondary: Tier weight for secondary documents.
    """

    alpha: float = 0.40
    beta: float = 0.40
    gamma: float = 0.20
    lambda_: float = 0.30
    tier_primary: float = 1.0
    tier_secondary: float = 0.3


@dataclass
class ReviewQueueItem:
    """One item in the prioritized review queue."""

    page_id: str
    priority_score: float
    evaluation_decision: str
    problem_class: str
    inter_engine_cer: float
    fusion_agreement_rate: float
    downstream_impact: int      # chunk_count + keyword_count
    tier: str
    language: str
    document_title: str | None
    document_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "priority_score": round(self.priority_score, 4),
            "evaluation_decision": self.evaluation_decision,
            "problem_class": self.problem_class,
            "inter_engine_cer": round(self.inter_engine_cer, 4),
            "fusion_agreement_rate": round(self.fusion_agreement_rate, 4),
            "downstream_impact": self.downstream_impact,
            "tier": self.tier,
            "language": self.language,
            "document_title": self.document_title,
            "document_id": self.document_id,
        }


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_priority_score(
    page: dict[str, Any],
    weights: PriorityWeights | None = None,
    *,
    max_downstream_impact: int = 100,
) -> float:
    """Compute the canonical priority score for one page.

    Args:
        page: Dict with keys: ``fusion_agreement_rate``, ``inter_engine_cer``,
            ``llm_coherence_variance`` (optional, defaults to 0),
            ``downstream_impact`` (chunk+keyword count),
            ``tier``.
        weights: :class:`PriorityWeights` instance; uses defaults if None.
        max_downstream_impact: Normalisation denominator for downstream impact.

    Returns:
        Priority score in [0, ~2] (not clamped — higher = review first).
    """
    w = weights or PriorityWeights()

    far = float(page.get("fusion_agreement_rate") or 0.0)
    lcv = float(page.get("llm_coherence_variance") or 0.0)
    uncertainty = (1.0 - far) + w.lambda_ * lcv

    raw_impact = float(page.get("downstream_impact") or 0)
    impact_norm = min(raw_impact / max(max_downstream_impact, 1), 1.0)

    tier = (page.get("tier") or "secondary").lower()
    tier_w = w.tier_primary if tier == "primary" else w.tier_secondary

    return round(
        w.alpha * uncertainty + w.beta * impact_norm + w.gamma * tier_w,
        6,
    )


# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

_QUEUE_QUERY = """
MATCH (p:PAGE)
WHERE p.evaluationDecision IN ['needs_review', 'failed']
  AND p.mode = 'ocr'
OPTIONAL MATCH (p)<-[:HAS]-(c:CHUNK)
OPTIONAL MATCH (c)-[:MENTION]->(k:KEYWORD)
OPTIONAL MATCH (p)<-[:INCLUDE]-(sec:SECTION)<-[:INCLUDE]-(ch:CHAPTER)
             <-[:CONSIST_OF]-(d:DOCUMENT)
WITH p,
     count(DISTINCT c) AS chunk_count,
     count(DISTINCT k) AS keyword_count,
     d.title AS doc_title,
     d.id    AS doc_id
RETURN
    p.id                    AS page_id,
    p.evaluationDecision    AS decision,
    p.problemClass          AS problem_class,
    p.interEngineCer        AS cer,
    p.fusionAgreementRate   AS far,
    p.tier                  AS tier,
    p.language              AS language,
    (chunk_count + keyword_count) AS downstream_impact,
    doc_title,
    doc_id
ORDER BY p.id
LIMIT $limit
"""

_SCORE_WRITE = """
UNWIND $rows AS r
MATCH (p:PAGE {id: r.page_id})
SET p.priorityScore       = r.score,
    p.priorityScoredAt    = r.ts,
    p.downstreamImpact    = r.downstream_impact
"""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_review_queue(
    driver: Driver,
    *,
    max_pages: int = 200,
    weights: PriorityWeights | None = None,
) -> list[ReviewQueueItem]:
    """Return NEEDS_REVIEW pages sorted by priority score (highest first).

    Args:
        driver: Open Neo4j driver.
        max_pages: Maximum queue size.
        weights: Override default priority weights.

    Returns:
        List of :class:`ReviewQueueItem` sorted by priority descending.
    """
    with driver.session() as s:
        rows = s.run(_QUEUE_QUERY, limit=max_pages * 3).data()

    # Compute max_downstream_impact for normalisation
    impacts = [r.get("downstream_impact") or 0 for r in rows]
    max_impact = max(impacts) if impacts else 1

    items: list[ReviewQueueItem] = []
    for row in rows:
        page = {
            "fusion_agreement_rate": row.get("far") or 0.0,
            "inter_engine_cer": row.get("cer") or 0.0,
            "downstream_impact": row.get("downstream_impact") or 0,
            "tier": row.get("tier") or "secondary",
        }
        score = compute_priority_score(page, weights, max_downstream_impact=max_impact)
        items.append(
            ReviewQueueItem(
                page_id=row["page_id"],
                priority_score=score,
                evaluation_decision=row.get("decision") or "",
                problem_class=row.get("problem_class") or "OK",
                inter_engine_cer=float(row.get("cer") or 0),
                fusion_agreement_rate=float(row.get("far") or 0),
                downstream_impact=int(row.get("downstream_impact") or 0),
                tier=row.get("tier") or "secondary",
                language=row.get("language") or "unknown",
                document_title=row.get("doc_title"),
                document_id=row.get("doc_id"),
            )
        )

    items.sort(key=lambda x: x.priority_score, reverse=True)
    log.info("get_review_queue: %d items (max_impact=%d)", len(items[:max_pages]), max_impact)
    return items[:max_pages]


def update_priority_scores(
    driver: Driver,
    *,
    batch_size: int = 500,
    weights: PriorityWeights | None = None,
) -> int:
    """Recompute and persist priority scores for all NEEDS_REVIEW pages.

    Should be called after every CORRECTION write (plan §11 step 5).

    Args:
        driver: Open Neo4j driver.
        batch_size: Pages to fetch per Cypher call.
        weights: Override default priority weights.

    Returns:
        Number of pages updated.
    """
    items = get_review_queue(driver, max_pages=batch_size * 10, weights=weights)
    if not items:
        return 0

    ts = datetime.now(timezone.utc).isoformat()
    write_rows = [
        {
            "page_id": item.page_id,
            "score": item.priority_score,
            "downstream_impact": item.downstream_impact,
            "ts": ts,
        }
        for item in items
    ]
    with driver.session() as s:
        s.run(_SCORE_WRITE, rows=write_rows).consume()

    log.info("update_priority_scores: updated %d pages", len(write_rows))
    return len(write_rows)
