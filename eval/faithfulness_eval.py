"""Faithfulness / verifier-gate evaluator (plan §0.6 D2).

Measures the zero-hallucination guarantee:
  - verified_rate:   fraction of returned results with verifier.ok == True
  - insufficient_evidence_rate: fraction returned as INSUFFICIENT_EVIDENCE
  - together they must = 1.0 (no result slips through ungated)
  - false_negative_rate: INSUFFICIENT_EVIDENCE on a query where a faithful
    chunk IS in the corpus (measures normalization gap)

Faithfulness query format (JSONL, extends the search query format):
  {
    "query_id": "q001",
    "query": "...",
    "faithful_span": "十惡尤切不容首免",   // exact span that MUST verify
    "faithful_chunk_id": "chunk_abc",        // chunk that contains the span
    "relevant_chunk_ids": ["chunk_abc"]
  }
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neo4j import Driver

from eval.bootstrap_ci import bootstrap_ci

log = logging.getLogger(__name__)


def run_faithfulness_eval(
    driver: Driver,
    query_file: Path,
    *,
    top_k: int = 100,
    rerank_top_k: int = 20,
) -> dict[str, Any]:
    """Evaluate verifier faithfulness on a JSONL query file.

    Returns metrics JSON with bootstrap CIs.
    """
    from apps.backend.pipeline.search import hybrid_search
    from apps.backend.retrieval.bm25 import BM25Corpus

    queries = _load_queries(query_file)
    if not queries:
        return {}

    bm25 = BM25Corpus.build(driver)

    verified_flags: list[float] = []
    insuf_flags: list[float] = []
    false_neg_flags: list[float] = []

    for q in queries:
        try:
            resp = hybrid_search(
                driver,
                q["query"],
                bm25,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                verify_span=q.get("faithful_span"),
            )
            all_results = resp.all_results
            if not all_results:
                continue

            for r in all_results:
                if r.verifier.outcome == "insufficient_evidence":
                    insuf_flags.append(1.0)
                    verified_flags.append(0.0)
                else:
                    insuf_flags.append(0.0)
                    verified_flags.append(1.0 if r.verified else 0.0)

            # False negative: faithful chunk was retrieved but verifier rejected it
            faithful_id = q.get("faithful_chunk_id")
            if faithful_id:
                for r in all_results:
                    if r.chunk_id == faithful_id and not r.verified:
                        false_neg_flags.append(1.0)
                        break
                else:
                    false_neg_flags.append(0.0)

        except Exception as exc:
            log.error("Faithfulness eval failed for %s: %s", q.get("query_id"), exc)

    return {
        "verified_rate": bootstrap_ci(verified_flags),
        "insufficient_evidence_rate": bootstrap_ci(insuf_flags),
        "false_negative_rate": bootstrap_ci(false_neg_flags) if false_neg_flags else None,
        "n_result_rows": len(verified_flags),
        "n_queries": len(queries),
    }


def _load_queries(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
