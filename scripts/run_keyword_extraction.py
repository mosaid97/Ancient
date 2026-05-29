#!/usr/bin/env python3
"""Background runner for full-corpus keyword extraction (Phase 6).

Extracts 3-8 keywords per CHUNK via deepseek-chat, wires KEYWORD nodes
and (:CHUNK)-[:MENTION]->(:KEYWORD) relationships.

Usage:
    caffeinate -dimsu uv run python scripts/run_keyword_extraction.py
    uv run python scripts/run_keyword_extraction.py --max-chunks 500 --verbose

Estimated runtime: ~34,000 chunks × ~1.5 s/call ≈ 14–16 hours unattended.
Requires LLM_API_KEY, LLM_BASE_URL, CHAT_LLM_MODEL in .env.
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

import os

from neo4j import GraphDatabase

from apps.backend.pipeline.keywords import run_keyword_extraction


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-corpus keyword extraction runner")
    p.add_argument("--max-chunks", type=int, default=None, help="Stop after N chunks (default: all)")
    p.add_argument("--batch-size", type=int, default=200, help="Neo4j page size (default: 200)")
    p.add_argument("--min-chars", type=int, default=50, help="Min chunk chars to process (default: 50)")
    p.add_argument("--model", default=None, help="Override chat model (default: CHAT_LLM_MODEL)")
    p.add_argument("--workers", type=int, default=1,
                   help="Thread-pool size for parallel LLM calls (default: 1). "
                        "8–12 recommended for Silra; tune down if rate-limited.")
    p.add_argument("--recompute", action="store_true", help="Reprocess already-extracted chunks (mentionStatus='ok')")
    p.add_argument("--recompute-failed", action="store_true", dest="recompute_failed",
                   help="Retry only failed chunks (mentionStatus='failed')")
    p.add_argument("--recompute-skipped", action="store_true", dest="recompute_skipped",
                   help="Retry skipped chunks (mentionStatus='skipped'); bypasses language filter")
    p.add_argument("--log-file", default="logs/run_keyword_extraction.log", help="Log file path")
    p.add_argument("--report-file", default="logs/keyword_extraction_report.json", help="JSON report path")
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

    _stop = False

    def _handle_signal(sig: int, _frame: object) -> None:
        nonlocal _stop
        log.info("Signal %s received — will stop after current chunk", sig)
        _stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "AncientChina")),
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )

    model = args.model or os.getenv("CHAT_LLM_MODEL", "deepseek-chat")
    log.info(
        "Starting keyword extraction (max_chunks=%s, model=%s, recompute=%s, "
        "recompute_failed=%s, min_chars=%d, workers=%d)",
        args.max_chunks,
        model,
        args.recompute,
        args.recompute_failed,
        args.min_chars,
        args.workers,
    )

    t_start = time.time()
    try:
        report = run_keyword_extraction(
            driver,
            model=model,
            batch_size=args.batch_size,
            max_chunks=args.max_chunks,
            min_chars=args.min_chars,
            recompute=args.recompute,
            recompute_failed=args.recompute_failed,
            recompute_skipped=args.recompute_skipped,
            max_workers=args.workers,
        )
    finally:
        driver.close()

    elapsed = time.time() - t_start
    log.info("=== DONE in %.1f s ===", elapsed)
    log.info("  chunks_ok=%d  failed=%d  skipped=%d  keywords=%d  unique=%d",
             report.chunks_ok, report.chunks_failed, report.chunks_skipped,
             report.keywords_extracted, report.keywords_unique)

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"elapsed_seconds": round(elapsed, 2), **report.to_dict()},
                   ensure_ascii=False, indent=2)
    )
    log.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
