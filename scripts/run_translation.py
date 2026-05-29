"""A1: Full-corpus translation runner — populates textCanonical + textVernacular (plan §0.6 A1).

Processes CHUNK nodes in batches:
  - primary tier: word → paragraph → review → textCanonical + textVernacular
  - secondary tier: normalization only → textCanonical (70% LLM cost saving)

Usage:
    caffeinate -dimsu uv run python scripts/run_translation.py --log-file logs/translation_run.log
    uv run python scripts/run_translation.py --tier primary --max-chunks 100   # smoke test
    uv run python scripts/run_translation.py --recompute                       # re-run all
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

from apps.backend.pipeline.translate import TranslationReport, translate_chunks


def _setup_logging(log_file: str) -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _connect():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "AncientChina")),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-corpus translation runner (A1)")
    ap.add_argument("--batch-size", type=int, default=50, help="Chunks per translate_chunks call")
    ap.add_argument("--max-chunks", type=int, default=None, help="Stop after N total chunks")
    ap.add_argument("--tier", choices=["primary", "secondary"], default=None, help="Limit to one tier")
    ap.add_argument("--recompute", action="store_true", help="Re-translate already-done chunks")
    ap.add_argument("--log-file", default="logs/translation_run.log")
    args = ap.parse_args()

    _setup_logging(args.log_file)
    log = logging.getLogger("run_translation")

    driver = _connect()
    log.info("Translation runner started — tier=%s batch=%d max=%s recompute=%s",
             args.tier, args.batch_size, args.max_chunks, args.recompute)

    t0 = time.time()
    total_ok = 0
    total_failed = 0
    total_skipped = 0
    batch_num = 0

    # Graceful shutdown on SIGINT/SIGTERM
    stop = {"flag": False}

    def _handle_signal(sig, _frame):
        log.info("Signal %d received — will stop after current batch", sig)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    while not stop["flag"]:
        report: TranslationReport = translate_chunks(
            driver,
            limit=args.batch_size,
            tier_filter=args.tier,
            recompute=args.recompute,
        )

        if report.total == 0:
            log.info("No more chunks to translate — done.")
            break

        total_ok += report.ok
        total_failed += report.failed
        total_skipped += report.skipped
        batch_num += 1

        elapsed = time.time() - t0
        log.info(
            "Batch %d done: ok=%d failed=%d skipped=%d | running total ok=%d failed=%d (%.0fs)",
            batch_num, report.ok, report.failed, report.skipped,
            total_ok, total_failed, elapsed,
        )

        if args.max_chunks and (total_ok + total_failed + total_skipped) >= args.max_chunks:
            log.info("Reached --max-chunks %d — stopping.", args.max_chunks)
            break

    elapsed = time.time() - t0
    log.info(
        "=== Translation complete in %.1fs — ok=%d failed=%d skipped=%d ===",
        elapsed, total_ok, total_failed, total_skipped,
    )

    report_path = Path("logs/translation_report.json")
    report_path.write_text(json.dumps({
        "total_ok": total_ok,
        "total_failed": total_failed,
        "total_skipped": total_skipped,
        "elapsed_seconds": round(elapsed, 1),
        "tier_filter": args.tier,
    }, indent=2))
    log.info("Report written to %s", report_path)
    driver.close()


if __name__ == "__main__":
    main()
