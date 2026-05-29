#!/usr/bin/env python3
"""Background runner for full-corpus chunking.

Usage:
    caffeinate -dimsu uv run python scripts/run_chunking.py
    uv run python scripts/run_chunking.py --max-pages 100 --recompute
    uv run python scripts/run_chunking.py --chunk-size 300 --overlap 30

Expected runtime: ~15–30 min for ~14,000 pages × ~5 chunks/page = ~70,000 chunks.
Text source priority per page: structuredMarkdown → textFused → text.
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

from apps.backend.pipeline.chunk import chunk_pages


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-corpus chunking runner")
    p.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (default: all)")
    p.add_argument("--chunk-size", type=int, default=500, help="Target chunk size in characters (default: 500)")
    p.add_argument("--overlap", type=int, default=50, help="Overlap size in characters (default: 50)")
    p.add_argument("--recompute", action="store_true", help="Re-chunk already-chunked pages")
    p.add_argument("--log-file", default="logs/chunking_run.log", help="Log file path")
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
        log.info("Signal %s received — will stop after current batch", sig)
        _stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "AncientChina")),
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )

    log.info(
        "Starting chunking (max_pages=%s, chunk_size=%d, overlap=%d, recompute=%s)",
        args.max_pages,
        args.chunk_size,
        args.overlap,
        args.recompute,
    )
    t0 = time.time()

    report = chunk_pages(
        driver,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_pages=args.max_pages,
        recompute=args.recompute,
    )

    elapsed = time.time() - t0
    log.info(
        "Chunking complete in %.1fs — chunked=%d skipped=%d failed=%d chunks=%d",
        elapsed,
        report.pages_chunked,
        report.pages_skipped,
        report.pages_failed,
        report.chunks_created,
    )
    if report.errors:
        log.warning("Errors (%d): %s", len(report.errors), report.errors[:5])

    report_path = Path("logs/chunking_report.json")
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    log.info("Report written to %s", report_path)

    driver.close()


if __name__ == "__main__":
    main()
