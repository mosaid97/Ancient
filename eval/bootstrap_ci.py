"""Bootstrap confidence-interval utilities (plan §0.6 D2).

Every metric in the eval harness is reported as:
    {"mean": float, "ci_low": float, "ci_high": float, "n": int}

using 10,000 bootstrap resamples and a 95% two-sided CI.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Return {mean, ci_low, ci_high, n} for `stat` applied to `values`.

    Uses the percentile bootstrap method (Efron 1979).
    Falls back to NaN when values is empty.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        nan = float("nan")
        return {"mean": nan, "ci_low": nan, "ci_high": nan, "n": 0}

    observed = float(stat(arr))
    if n == 1:
        return {"mean": observed, "ci_low": observed, "ci_high": observed, "n": 1}

    rng = np.random.default_rng(seed)
    boot_stats = np.array(
        [stat(rng.choice(arr, size=n, replace=True)) for _ in range(n_resamples)]
    )

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.percentile(boot_stats, alpha * 100))
    ci_high = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return {"mean": observed, "ci_low": ci_low, "ci_high": ci_high, "n": n}


def ndcg_at_k(
    relevant_ids: set[str],
    ranked_ids: list[str],
    k: int,
) -> float:
    """Compute NDCG@k given a set of relevant chunk IDs and a ranked list."""
    if not relevant_ids or not ranked_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, cid in enumerate(ranked_ids[:k])
        if cid in relevant_ids
    )
    ideal = sum(
        1.0 / math.log2(rank + 2)
        for rank in range(min(len(relevant_ids), k))
    )
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not ranked_ids or k == 0:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in relevant_ids)
    return hits / min(k, len(ranked_ids))


def mrr(relevant_ids: set[str], ranked_ids: list[str]) -> float:
    """Mean Reciprocal Rank (single-query version)."""
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0
