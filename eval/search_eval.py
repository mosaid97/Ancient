"""Search quality evaluator (plan §0.6 D2).

Runs a labelled query set against the hybrid_search pipeline and reports:
  - NDCG@10, Recall@10, Precision@10, MRR (overall + per-tier + per-language)
  - Each metric with 10k-bootstrap 95% CI

Query set format (JSON Lines, one object per line):
  {
    "query_id": "q001",
    "query": "唐律疏議中的十惡是什麼",
    "relevant_chunk_ids": ["chunk_abc", "chunk_xyz"],
    "tier": "primary",          // optional filter hint
    "language": "zh-classical"  // optional
  }
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neo4j import Driver

from eval.bootstrap_ci import bootstrap_ci, mrr, ndcg_at_k, precision_at_k, recall_at_k

log = logging.getLogger(__name__)


@dataclass
class QueryResult:
    query_id: str
    query: str
    tier: str | None
    language: str | None
    relevant_chunk_ids: set[str]
    ranked_chunk_ids: list[str]
    duration_ms: float

    def metrics(self, k: int = 10) -> dict[str, float]:
        return {
            "ndcg": ndcg_at_k(self.relevant_chunk_ids, self.ranked_chunk_ids, k),
            "recall": recall_at_k(self.relevant_chunk_ids, self.ranked_chunk_ids, k),
            "precision": precision_at_k(self.relevant_chunk_ids, self.ranked_chunk_ids, k),
            "mrr": mrr(self.relevant_chunk_ids, self.ranked_chunk_ids),
            "duration_ms": self.duration_ms,
        }


def run_search_eval(
    driver: Driver,
    query_file: Path,
    *,
    k: int = 10,
    mode: str = "hybrid",
    top_k: int = 100,
    rerank_top_k: int = 20,
) -> dict[str, Any]:
    """Evaluate search against a JSONL labelled query file.

    Returns a metrics dict suitable for JSON serialisation.
    """
    from apps.backend.pipeline.search import hybrid_search
    from apps.backend.retrieval.bm25 import BM25Corpus

    queries = _load_queries(query_file)
    if not queries:
        log.warning("No queries found in %s", query_file)
        return {}

    bm25 = BM25Corpus.build(driver)
    results: list[QueryResult] = []

    for q in queries:
        try:
            resp = hybrid_search(
                driver,
                q["query"],
                bm25,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                use_community=True,
            )
            ranked = [r.chunk_id for r in resp.all_results]
            results.append(
                QueryResult(
                    query_id=q["query_id"],
                    query=q["query"],
                    tier=q.get("tier"),
                    language=q.get("language"),
                    relevant_chunk_ids=set(q.get("relevant_chunk_ids", [])),
                    ranked_chunk_ids=ranked,
                    duration_ms=resp.duration_ms,
                )
            )
        except Exception as exc:
            log.error("Search failed for query %s: %s", q.get("query_id"), exc)

    return _aggregate(results, k=k)


def _load_queries(path: Path) -> list[dict]:
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def _aggregate(results: list[QueryResult], k: int) -> dict[str, Any]:
    def _slice(
        name: str,
        items: list[QueryResult],
    ) -> dict[str, Any]:
        if not items:
            return {}
        all_m = [r.metrics(k) for r in items]
        return {
            "ndcg_at_k": bootstrap_ci([m["ndcg"] for m in all_m]),
            "recall_at_k": bootstrap_ci([m["recall"] for m in all_m]),
            "precision_at_k": bootstrap_ci([m["precision"] for m in all_m]),
            "mrr": bootstrap_ci([m["mrr"] for m in all_m]),
            "duration_ms": bootstrap_ci([m["duration_ms"] for m in all_m]),
            "n_queries": len(items),
            "k": k,
        }

    out: dict[str, Any] = {"overall": _slice("overall", results)}

    for tier in ("primary", "secondary"):
        sliced = [r for r in results if r.tier == tier]
        if sliced:
            out[f"tier_{tier}"] = _slice(tier, sliced)

    for lang in ("zh-classical", "zh-modern", "ja-kanbun", "ja-modern"):
        sliced = [r for r in results if r.language == lang]
        if sliced:
            out[f"lang_{lang}"] = _slice(lang, sliced)

    return out
