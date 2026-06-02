"""Translate DOCUMENT.title into titleZh, titleEn, titleJa, titleAr.

Usage:
    uv run python scripts/run_title_translation.py
    uv run python scripts/run_title_translation.py --recompute   # re-translate all
    uv run python scripts/run_title_translation.py --dry-run     # show titles without translating
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import os

from neo4j import GraphDatabase

from apps.backend.pipeline.translate_titles import translate_document_titles


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
    ap = argparse.ArgumentParser(description="Translate DOCUMENT titles into 4 languages")
    ap.add_argument("--recompute", action="store_true", help="Re-translate already-done documents")
    ap.add_argument("--dry-run", action="store_true", help="List document titles without translating")
    ap.add_argument("--log-file", default="logs/title_translation.log")
    args = ap.parse_args()

    _setup_logging(args.log_file)
    log = logging.getLogger("run_title_translation")

    driver = _connect()

    if args.dry_run:
        with driver.session() as s:
            rows = s.run("MATCH (d:DOCUMENT) WHERE d.title IS NOT NULL RETURN d.id AS id, d.title AS title ORDER BY d.id").data()
        print(f"Found {len(rows)} documents with titles:")
        for r in rows:
            print(f"  {r['id']!r:50s}  {r['title']!r}")
        driver.close()
        return

    log.info("Title translation started — recompute=%s", args.recompute)
    t0 = time.time()

    report = translate_document_titles(driver, recompute=args.recompute)

    elapsed = time.time() - t0
    log.info(
        "=== Title translation complete in %.1fs — ok=%d failed=%d skipped=%d ===",
        elapsed, report.ok, report.failed, report.skipped,
    )

    # Print a summary table
    print(f"\n{'Title':40s}  {'English':40s}  {'Status'}")
    print("-" * 95)
    for r in report.results:
        title_short = r.original_title[:38] + ".." if len(r.original_title) > 40 else r.original_title
        en_short = r.title_en[:38] + ".." if len(r.title_en) > 40 else r.title_en
        print(f"{title_short:40s}  {en_short:40s}  {r.status}")

    report_path = Path("logs/title_translation_report.json")
    report_path.write_text(json.dumps({
        "total": report.total,
        "ok": report.ok,
        "failed": report.failed,
        "skipped": report.skipped,
        "elapsed_seconds": round(elapsed, 1),
        "results": [
            {
                "doc_id": r.doc_id,
                "original_title": r.original_title,
                "status": r.status,
                "title_zh": r.title_zh,
                "title_en": r.title_en,
                "title_ja": r.title_ja,
                "title_ar": r.title_ar,
            }
            for r in report.results
        ],
    }, ensure_ascii=False, indent=2))
    log.info("Report written to %s", report_path)
    driver.close()


if __name__ == "__main__":
    main()
