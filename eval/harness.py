"""D2: AncientChinaSearch-Bench evaluation harness (plan §0.6 D2).

Usage:
    uv run python -m eval.harness [--bench data/bench] [--out eval_out/] [--k 10]

Produces:
    eval_out/metrics.json   — all metrics with 10k-bootstrap 95% CIs
    eval_out/summary.txt    — human-readable summary table

Expected bench layout (data/bench/):
    queries.jsonl           — search queries with relevant_chunk_ids
    faithfulness.jsonl      — queries with faithful_span + faithful_chunk_id
    linking_gold.jsonl      — secondary_chunk_id → primary_chunk_ids gold pairs
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _connect_neo4j():
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    return GraphDatabase.driver(uri, auth=(user, password))


def _fmt_ci(d: dict | None, decimals: int = 4) -> str:
    if not d:
        return "n/a"
    m = round(d["mean"], decimals)
    lo = round(d["ci_low"], decimals)
    hi = round(d["ci_high"], decimals)
    return f"{m} [{lo}, {hi}]"


def _summary_table(metrics: dict[str, Any]) -> str:
    lines = ["AncientChinaSearch-Bench — Evaluation Summary", "=" * 60]
    for section, data in metrics.items():
        if not isinstance(data, dict):
            continue
        lines.append(f"\n[{section}]")
        for key, val in data.items():
            if isinstance(val, dict) and "mean" in val:
                lines.append(f"  {key:<30s} {_fmt_ci(val)}")
            elif not isinstance(val, dict):
                lines.append(f"  {key:<30s} {val}")
    return "\n".join(lines)


def run(
    bench_dir: Path,
    out_dir: Path,
    k: int = 10,
) -> dict[str, Any]:
    driver = _connect_neo4j()
    all_metrics: dict[str, Any] = {
        "eval_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "k": k,
    }

    # Search eval
    query_file = bench_dir / "queries.jsonl"
    if query_file.exists():
        log.info("Running search eval on %s …", query_file)
        from eval.search_eval import run_search_eval
        try:
            m = run_search_eval(driver, query_file, k=k)
            all_metrics["search"] = m
            log.info("Search eval done: %d query slices", len(m))
        except Exception as exc:
            log.error("Search eval failed: %s", exc)
            all_metrics["search"] = {"error": str(exc)}
    else:
        log.warning("No queries.jsonl in %s — skipping search eval", bench_dir)

    # Faithfulness eval
    faith_file = bench_dir / "faithfulness.jsonl"
    if faith_file.exists():
        log.info("Running faithfulness eval on %s …", faith_file)
        from eval.faithfulness_eval import run_faithfulness_eval
        try:
            m = run_faithfulness_eval(driver, faith_file)
            all_metrics["faithfulness"] = m
            log.info("Faithfulness eval done")
        except Exception as exc:
            log.error("Faithfulness eval failed: %s", exc)
            all_metrics["faithfulness"] = {"error": str(exc)}
    else:
        log.warning("No faithfulness.jsonl in %s — skipping", bench_dir)

    # Linking eval
    gold_file = bench_dir / "linking_gold.jsonl"
    if gold_file.exists():
        log.info("Running linking eval on %s …", gold_file)
        from eval.linking_eval import run_linking_eval
        try:
            m = run_linking_eval(driver, gold_file, k=k)
            all_metrics["linking"] = m
            log.info("Linking eval done")
        except Exception as exc:
            log.error("Linking eval failed: %s", exc)
            all_metrics["linking"] = {"error": str(exc)}
    else:
        log.warning("No linking_gold.jsonl in %s — skipping", bench_dir)

    driver.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    log.info("Metrics written → %s", metrics_path)

    summary = _summary_table(all_metrics)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary)
    log.info("Summary written → %s", summary_path)
    print("\n" + summary)

    return all_metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AncientChinaSearch-Bench harness")
    parser.add_argument("--bench", default="data/bench", help="Bench directory (JSONL files)")
    parser.add_argument("--out", default="eval_out", help="Output directory for metrics")
    parser.add_argument("--k", type=int, default=10, help="Cutoff k for ranking metrics")
    args = parser.parse_args(argv)

    metrics = run(Path(args.bench), Path(args.out), k=args.k)
    if not metrics:
        sys.exit(1)


if __name__ == "__main__":
    main()
