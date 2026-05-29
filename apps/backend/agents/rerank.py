"""C2: BGE-reranker-v2-gemma cross-encoder reranker (plan §0.6 C2).

Reranks top-100 hybrid retrieval candidates down to top-20 using
FlagEmbedding's FlagReranker (BGE-reranker-v2-gemma). The model is
lazy-loaded and cached on first use.

Public API
----------
rerank(query, candidates, *, top_k, threshold) -> list[RerankResult]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

_RERANKER_MODEL = "BAAI/bge-reranker-v2-gemma"
_DEFAULT_TOP_K = 20
_DEFAULT_THRESHOLD = 0.0

_reranker_instance: Any = None


def _get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        try:
            from FlagEmbedding import FlagReranker  # type: ignore[import]
            _reranker_instance = FlagReranker(_RERANKER_MODEL, use_fp16=True)
            log.info("Loaded reranker: %s", _RERANKER_MODEL)
        except Exception as exc:
            log.error("Failed to load reranker %s: %s", _RERANKER_MODEL, exc)
            raise
    return _reranker_instance


@dataclass
class RerankResult:
    """Single reranked candidate."""

    chunk_id: str
    retrieval_score: float   # original RRF / vector score
    rerank_score: float      # cross-encoder score (normalized 0–1)
    rank: int                # 1-based rank after reranking

    @property
    def combined_score(self) -> float:
        """0.7 × rerank + 0.3 × retrieval (retrieval capped at 1.0)."""
        return 0.7 * self.rerank_score + 0.3 * min(self.retrieval_score, 1.0)


def rerank(
    query: str,
    candidates: list[tuple[str, str, float]],
    *,
    top_k: int = _DEFAULT_TOP_K,
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[RerankResult]:
    """Rerank candidate chunks with BGE-reranker-v2-gemma.

    Args:
        query: The user query string.
        candidates: List of (chunk_id, chunk_text, retrieval_score) tuples.
            Typically the top-100 from RRF fusion.
        top_k: Number of results to return after reranking.
        threshold: Minimum rerank score to include (default 0 = keep all).

    Returns:
        List of :class:`RerankResult` sorted by rerank_score descending.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, text) for _, text, _ in candidates]

    try:
        raw_scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(raw_scores, float):
            raw_scores = [raw_scores]
        scores = [float(s) for s in raw_scores]
    except Exception as exc:
        log.warning("Reranker scoring failed, returning retrieval order: %s", exc)
        return [
            RerankResult(
                chunk_id=cid,
                retrieval_score=ret_score,
                rerank_score=ret_score,
                rank=i + 1,
            )
            for i, (cid, _, ret_score) in enumerate(candidates[:top_k])
        ]

    # Combine and sort
    combined = [
        (cid, ret_score, rr_score)
        for (cid, _, ret_score), rr_score in zip(candidates, scores)
        if rr_score >= threshold
    ]
    combined.sort(key=lambda x: x[2], reverse=True)

    return [
        RerankResult(
            chunk_id=cid,
            retrieval_score=ret_score,
            rerank_score=rr_score,
            rank=i + 1,
        )
        for i, (cid, ret_score, rr_score) in enumerate(combined[:top_k])
    ]
