#!/usr/bin/env python3
"""Background runner for full-corpus 3-engine OCR fusion.

Usage:
    caffeinate -dimsu uv run python scripts/run_fusion.py
    uv run python scripts/run_fusion.py --max-pages 100 --recompute
    uv run python scripts/run_fusion.py --document 唐律疏議箋解__abc123

Expected runtime: ~30–60 min for 2,942 OCR pages (pure in-memory difflib, no API calls).
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

from apps.backend.pipeline.fusion import fuse_pages


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-corpus OCR fusion runner")
    p.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (default: all)")
    p.add_argument("--document", default=None, help="Limit to a single document_id")
    p.add_argument("--recompute", action="store_true", help="Re-fuse already-fused pages")
    p.add_argument("--no-lang-detect", action="store_true", help="Skip post-fusion language detection")
    p.add_argument("--log-file", default="logs/fusion_run.log", help="Log file path")
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
        "Starting full-corpus fusion (document=%s, max_pages=%s, recompute=%s, lang_detect=%s)",
        args.document or "ALL",
        args.max_pages,
        args.recompute,
        not args.no_lang_detect,
    )
    t0 = time.time()

    report = fuse_pages(
        driver,
        document_id=args.document,
        max_pages=args.max_pages,
        recompute_existing=args.recompute,
        run_language_detection=not args.no_lang_detect,
    )

    elapsed = time.time() - t0
    log.info(
        "Fusion complete in %.1fs — dual_fused=%d single=%d failed=%d total=%d",
        elapsed,
        report.pages_dual_fused,
        report.pages_single_engine,
        report.pages_failed,
        report.pages_total,
    )
    if report.avg_agreement_rate is not None:
        log.info("Avg agreement rate: %.3f", report.avg_agreement_rate)
    if report.errors:
        log.warning("Errors (%d): %s", len(report.errors), report.errors[:5])

    report_path = Path("logs/fusion_report.json")
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    log.info("Report written to %s", report_path)

    driver.close()


if __name__ == "__main__":
    main()
