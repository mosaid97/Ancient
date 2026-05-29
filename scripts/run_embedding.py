#!/usr/bin/env python3
"""Background runner for full-corpus CHUNK embedding via Silra text-embedding-v4.

Usage:
    caffeinate -dimsu uv run python scripts/run_embedding.py
    uv run python scripts/run_embedding.py --max-chunks 1000 --recompute

Expected runtime: ~70,000 chunks × 32 per batch = ~2,200 API calls × ~1.5 s = ~55 min.
Requires EMBED_LLM_MODEL and LLM_API_KEY / LLM_BASE_URL in .env.
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

from apps.backend.pipeline.embed import embed_chunks


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-corpus CHUNK embedding runner")
    p.add_argument("--max-chunks", type=int, default=None, help="Stop after N chunks (default: all)")
    p.add_argument("--batch-size", type=int, default=10, help="API batch size (default: 10, Silra max)")
    p.add_argument("--model", default=None, help="Override embedding model (default: EMBED_LLM_MODEL)")
    p.add_argument("--recompute", action="store_true", help="Re-embed already-embedded chunks")
    p.add_argument("--log-file", default="logs/embedding_run.log", help="Log file path")
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

    model = args.model or os.getenv("EMBED_LLM_MODEL", "text-embedding-v4")
    log.info(
        "Starting embedding (max_chunks=%s, batch_size=%d, model=%s, recompute=%s)",
        args.max_chunks,
        args.batch_size,
        model,
        args.recompute,
    )
    t0 = time.time()

    report = embed_chunks(
        driver,
        model=model,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
        recompute=args.recompute,
    )

    elapsed = time.time() - t0
    log.info(
        "Embedding complete in %.1fs — embedded=%d failed=%d skipped=%d total=%d",
        elapsed,
        report.chunks_embedded,
        report.chunks_failed,
        report.chunks_skipped,
        report.chunks_total,
    )
    if report.errors:
        log.warning("Errors (%d): %s", len(report.errors), report.errors[:5])

    report_path = Path("logs/embedding_report.json")
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    log.info("Report written to %s", report_path)

    driver.close()


if __name__ == "__main__":
    main()
