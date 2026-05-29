"""Primary↔Secondary CITES linking evaluator (plan §0.6 D2).

Measures the entailment citation linker (B2):
  - precision@k:  fraction of top-k CITES edges that are correct
  - recall@k:     fraction of gold CITES pairs recovered in top-k
  - f1@k

Gold pairs format (JSONL):
  {
    "secondary_chunk_id": "chunk_sec_001",
    "primary_chunk_ids": ["chunk_pri_042", "chunk_pri_099"]
  }

The evaluator queries Neo4j for the actual CITES edges ordered by confidence
and compares to the gold set.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neo4j import Driver

from eval.bootstrap_ci import bootstrap_ci, precision_at_k, recall_at_k

log = logging.getLogger(__name__)

_CITES_QUERY = """
MATCH (sec:CHUNK {id: $chunk_id})-[r:CITES]->(pri:CHUNK)
RETURN pri.id AS primary_id, r.confidence AS confidence
ORDER BY r.confidence DESC
LIMIT $top_k
"""


def run_linking_eval(
    driver: Driver,
    gold_file: Path,
    *,
    k: int = 10,
) -> dict[str, Any]:
    """Evaluate CITES linking against a JSONL gold file."""
    gold_pairs = _load_gold(gold_file)
    if not gold_pairs:
        return {}

    prec_scores: list[float] = []
    rec_scores: list[float] = []
    f1_scores: list[float] = []

    for item in gold_pairs:
        sec_id = item["secondary_chunk_id"]
        gold_primary = set(item.get("primary_chunk_ids", []))
        if not gold_primary:
            continue

        try:
            with driver.session() as s:
                rows = s.run(_CITES_QUERY, chunk_id=sec_id, top_k=k).data()
            ranked = [r["primary_id"] for r in rows]
        except Exception as exc:
            log.error("CITES query failed for %s: %s", sec_id, exc)
            continue

        p = precision_at_k(gold_primary, ranked, k)
        r = recall_at_k(gold_primary, ranked, k)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        prec_scores.append(p)
        rec_scores.append(r)
        f1_scores.append(f1)

    return {
        f"precision_at_{k}": bootstrap_ci(prec_scores),
        f"recall_at_{k}": bootstrap_ci(rec_scores),
        f"f1_at_{k}": bootstrap_ci(f1_scores),
        "n_evaluated": len(prec_scores),
        "k": k,
    }


def _load_gold(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
