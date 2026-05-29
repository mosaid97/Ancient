"""C1: Reciprocal Rank Fusion (RRF) to merge multiple retrieval legs (plan §0.6 C1).

RRF formula: score(d) = Σ_leg 1 / (k + rank(d, leg))
where k=60 is the standard constant from Cormack et al. 2009.

Usage::

    from apps.backend.retrieval.fuse import rrf_fuse

    bm25_hits  = [(chunk_id, score), ...]   # any ranked list
    dense_hits = [(chunk_id, score), ...]
    fused = rrf_fuse([bm25_hits, dense_hits], top_k=100)
"""
from __future__ import annotations

_RRF_K = 60


def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
    *,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Merge ranked lists via Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each element is a ranked list of (chunk_id, score) pairs
            already sorted descending by score. The score values are ignored for
            RRF (only rank position is used).
        top_k: Number of top results to return.

    Returns:
        Fused list of (chunk_id, rrf_score) sorted descending by rrf_score.
    """
    rrf_scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def normalize_scores(hits: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Min-max normalize scores to [0, 1] for display/comparison."""
    if not hits:
        return hits
    min_s = min(s for _, s in hits)
    max_s = max(s for _, s in hits)
    if max_s == min_s:
        return [(cid, 1.0) for cid, _ in hits]
    return [(cid, (s - min_s) / (max_s - min_s)) for cid, s in hits]
