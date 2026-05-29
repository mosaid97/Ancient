#!/usr/bin/env python3
"""Background runner for full-corpus OCR page evaluation (Phase 4).

Runs inter-engine CER + LLM problem classification over fused OCR pages
and writes evaluationDecision / problemClass to Neo4j.

Usage:
    caffeinate -dimsu uv run python scripts/run_evaluator.py --log-file logs/evaluator.log
    uv run python scripts/run_evaluator.py --max-pages 50
    uv run python scripts/run_evaluator.py --recompute

Estimated runtime: ~2,900 pages; most pass on CER alone (~2 % LLM call rate)
→ roughly 1–3 hours depending on Silra latency.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from apps.backend.agents.evaluator import evaluate_pages
from apps.backend.graph.neo4j_client import get_driver
from apps.backend.llm.silra import get_silra_client


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-corpus OCR evaluator runner")
    p.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (default: all)")
    p.add_argument("--batch-size", type=int, default=50, help="Neo4j fetch batch (default: 50)")
    p.add_argument("--recompute", action="store_true", help="Re-evaluate already-evaluated pages")
    p.add_argument("--log-file", default="logs/evaluator.log", help="Log file path")
    p.add_argument("--report-file", default="logs/evaluator_report.json", help="JSON report path")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    def _handle_signal(sig: int, _frame: object) -> None:
        log.info("Signal %s received — will stop after current page", sig)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    driver = get_driver()
    client = get_silra_client()

    log.info(
        "Starting evaluator (max_pages=%s, batch_size=%d, recompute=%s)",
        args.max_pages,
        args.batch_size,
        args.recompute,
    )

    t0 = time.time()
    try:
        report = evaluate_pages(
            driver,
            max_pages=args.max_pages,
            recompute=args.recompute,
            batch_size=args.batch_size,
            client=client,
        )
    finally:
        driver.close()

    elapsed = time.time() - t0
    log.info(
        "=== DONE in %.1f s (%.1f min) === total=%d pass=%d review=%d failed=%d "
        "skipped=%d llm_calls=%d",
        elapsed,
        elapsed / 60,
        report.pages_total,
        report.pages_pass,
        report.pages_needs_review,
        report.pages_failed,
        report.pages_skipped,
        report.llm_calls,
    )

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    log.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
