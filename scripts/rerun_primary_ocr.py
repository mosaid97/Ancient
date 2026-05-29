"""Re-run LLM OCR on primary-tier pages with the updated traditional-script prompts.

Usage (dry-run preview):
    uv run python scripts/rerun_primary_ocr.py --engine qwen --dry-run

Full Qwen re-run on all primary pages (including those already 'ok' to fix
any silent simplified-character conversion from the old prompt):
    caffeinate -dimsu uv run python scripts/rerun_primary_ocr.py --engine qwen --recompute

Run Qwen only on pages currently flagged empty (validation_failed:simplified_conversion…):
    caffeinate -dimsu uv run python scripts/rerun_primary_ocr.py --engine qwen

Also re-run DeepSeek on primary pages:
    caffeinate -dimsu uv run python scripts/rerun_primary_ocr.py --engine deepseek --recompute

Limit to one document for quick smoke test:
    uv run python scripts/rerun_primary_ocr.py --engine qwen --document 唐律疏議箋解__3fbb3392d0 --max 20

The script uses the exact same orchestrators that notebooks/03_ocr.ipynb uses so
behaviour is identical.  All output is logged to logs/rerun_primary_ocr_<engine>.log
and a summary JSON is written to logs/rerun_primary_ocr_<engine>_report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

# ── make imports work without PYTHONPATH= ─────────────────────────────────
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(repo_root / ".env")

from apps.backend.graph.neo4j_client import get_driver  # noqa: E402
from apps.backend.pipeline.extract import (  # noqa: E402
    ExtractRunReport,
    run_deepseek_pages,
    run_qwen_pages,
)
from apps.backend.storage.minio_client import get_minio_client  # noqa: E402

# ── logging ────────────────────────────────────────────────────────────────
logs_dir = repo_root / "logs"
logs_dir.mkdir(exist_ok=True)

_STOP_REQUESTED = False


def _handle_signal(sig: int, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"\n[signal {sig}] Finishing current page then stopping …", flush=True)


def _setup_logging(engine: str) -> logging.Logger:
    log_file = logs_dir / f"rerun_primary_ocr_{engine}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--engine",
        choices=["qwen", "deepseek", "both"],
        default="qwen",
        help="Which LLM OCR engine to re-run (default: qwen).",
    )
    p.add_argument(
        "--recompute",
        action="store_true",
        default=False,
        help=(
            "Re-run even pages that already have status='ok'.  Use this "
            "after a prompt change to fix previously 'ok' but silently-simplified outputs.  "
            "Without --recompute only pages with status IS NULL, 'failed', or 'empty' are processed."
        ),
    )
    p.add_argument(
        "--include-empty",
        action="store_true",
        default=False,
        help="Also re-run pages with status='empty' (implied by --recompute).",
    )
    p.add_argument(
        "--document",
        default=None,
        metavar="DOCUMENT_ID",
        help="Limit re-run to one document id (for smoke testing).",
    )
    p.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N pages (useful for quick smoke tests).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print how many pages would be processed, then exit without calling the API.",
    )
    p.add_argument(
        "--bucket",
        default="ancient-pages",
        help="MinIO bucket (default: ancient-pages).",
    )
    return p.parse_args()


# ── dry-run query ──────────────────────────────────────────────────────────

_DRY_RUN_CYPHER = """
MATCH (p:PAGE)
WHERE p.tier = 'primary'
  AND p.mode = 'ocr'
  AND p.preprocessedImageUri IS NOT NULL
  AND (p.role IS NULL OR p.role = 'body')
  AND ($document_id IS NULL OR p.documentId = $document_id)
RETURN count(p) AS total,
       count(CASE WHEN p.{status_col} IS NULL THEN 1 END) AS unset,
       count(CASE WHEN p.{status_col} = 'ok'     THEN 1 END) AS ok,
       count(CASE WHEN p.{status_col} = 'empty'  THEN 1 END) AS empty,
       count(CASE WHEN p.{status_col} = 'failed' THEN 1 END) AS failed
"""


def _dry_run_report(driver, engine: str, document_id: str | None, logger: logging.Logger) -> None:
    status_col = {"qwen": "qwenVlOcrStatus", "deepseek": "deepseekOcrStatus"}[engine]
    cypher = _DRY_RUN_CYPHER.replace("{status_col}", status_col)
    with driver.session() as s:
        row = s.run(cypher, document_id=document_id).single()
    logger.info(
        "DRY RUN — primary-tier pages eligible for %s re-run:\n"
        "  total=%d  unset=%d  ok=%d  empty=%d  failed=%d",
        engine, row["total"], row["unset"], row["ok"], row["empty"], row["failed"],
    )
    logger.info(
        "  With --recompute: all %d pages would be processed.", row["total"]
    )
    logger.info(
        "  Without --recompute: %d pages would be processed (unset + failed + empty).",
        row["unset"] + row["failed"] + row["empty"],
    )


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    engine_tag = args.engine if args.engine != "both" else "qwen_and_deepseek"
    logger = _setup_logging(engine_tag)
    logger.info("=== rerun_primary_ocr.py  engine=%s  recompute=%s ===", args.engine, args.recompute)

    driver = get_driver()
    minio_client = get_minio_client()

    engines_to_run: list[str] = (
        ["qwen", "deepseek"] if args.engine == "both" else [args.engine]
    )

    if args.dry_run:
        for eng in engines_to_run:
            _dry_run_report(driver, eng, args.document, logger)
        return

    all_reports: dict[str, dict] = {}

    for eng in engines_to_run:
        logger.info("--- Starting %s OCR on primary-tier pages ---", eng)
        t0 = time.monotonic()

        include_empty = args.include_empty or args.recompute

        try:
            if eng == "qwen":
                report: ExtractRunReport = run_qwen_pages(
                    driver=driver,
                    minio_client=minio_client,
                    document_id=args.document,
                    max_pages=args.max,
                    recompute_existing=args.recompute,
                    include_empty=include_empty,
                    bucket=args.bucket,
                    progress_every=10,
                )
            else:
                report = run_deepseek_pages(
                    driver=driver,
                    minio_client=minio_client,
                    document_id=args.document,
                    max_pages=args.max,
                    recompute_existing=args.recompute,
                    include_empty=include_empty,
                    bucket=args.bucket,
                    progress_every=10,
                )
        except KeyboardInterrupt:
            logger.warning("Interrupted during %s run — saving partial report.", eng)
            break

        elapsed = time.monotonic() - t0
        logger.info(
            "%s DONE: total=%d ok=%d empty=%d failed=%d skipped=%d  %.1fs",
            eng,
            report.pages_total,
            report.pages_processed,
            report.pages_empty,
            report.pages_failed,
            report.pages_skipped,
            elapsed,
        )
        if report.errors:
            logger.warning("First 10 errors:\n  %s", "\n  ".join(report.errors[:10]))

        all_reports[eng] = report.to_dict()

    report_path = logs_dir / f"rerun_primary_ocr_{engine_tag}_report.json"
    report_path.write_text(json.dumps(all_reports, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report saved to %s", report_path)

    driver.close()


if __name__ == "__main__":
    main()
