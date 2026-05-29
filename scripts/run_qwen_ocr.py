"""Standalone background runner: Qwen-VL-OCR over all eligible body pages.

Usage::

    # Dry-run (prints page count, exits)
    uv run python scripts/run_qwen_ocr.py --dry-run

    # Full corpus (background-safe; keep laptop awake with caffeinate)
    caffeinate -dimsu uv run python scripts/run_qwen_ocr.py

    # Single document
    uv run python scripts/run_qwen_ocr.py --document-id <doc_id>

    # Cap page count for smoke test
    uv run python scripts/run_qwen_ocr.py --max-pages 20

    # Force re-OCR of already-processed pages
    uv run python scripts/run_qwen_ocr.py --recompute

Exit codes: 0 = success, 1 = partial failure (some pages failed), 2 = fatal.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

# Prepend repo root so `uv run python scripts/run_qwen_ocr.py` works
# without PYTHONPATH= (mirrors scripts/run_paddle_ocr.py).
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv

load_dotenv(_REPO / ".env")

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.llm.silra import get_silra_client
from apps.backend.pipeline.extract import run_qwen_pages
from apps.backend.storage.minio_client import get_minio_client

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_REPO / "logs" / "qwen_ocr_run.log", mode="a"),
    ],
)
logger = logging.getLogger("run_qwen_ocr")

# ---------------------------------------------------------------------------
# SIGINT / SIGTERM: finish current page then exit cleanly.
# ---------------------------------------------------------------------------

_STOP = False


def _handle_signal(signum: int, _frame) -> None:  # noqa: ANN001
    global _STOP  # noqa: PLW0603
    _STOP = True
    logger.info("Signal %d received — will stop after current page.", signum)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen-VL-OCR corpus runner")
    parser.add_argument("--dry-run", action="store_true", help="Count pages and exit")
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--recompute", action="store_true", help="Re-OCR already-done pages")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-page API timeout (s)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    logs_dir = _REPO / "logs"
    logs_dir.mkdir(exist_ok=True)

    driver = get_driver()
    minio = get_minio_client()
    silra = get_silra_client(timeout=args.timeout)

    if args.dry_run:
        with driver.session() as s:
            n = s.run(
                """
                MATCH (p:PAGE)
                WHERE p.mode = 'ocr'
                  AND p.preprocessedImageUri IS NOT NULL
                  AND p.role IN ['body']
                  AND ($recompute = true OR p.qwenVlOcrStatus IS NULL
                       OR p.qwenVlOcrStatus = 'failed')
                  AND ($document_id IS NULL OR p.documentId = $document_id)
                RETURN count(p) AS n
                """,
                recompute=args.recompute,
                document_id=args.document_id,
            ).single()["n"]
        print(f"[DRY RUN] Pages eligible for Qwen-VL-OCR: {n}")
        driver.close()
        return 0

    logger.info(
        "Starting Qwen-VL-OCR | max_pages=%s | recompute=%s | doc=%s",
        args.max_pages, args.recompute, args.document_id,
    )

    t0 = time.monotonic()
    report = run_qwen_pages(
        driver=driver,
        minio_client=minio,
        silra_client=silra,
        document_id=args.document_id,
        max_pages=args.max_pages,
        recompute_existing=args.recompute,
        progress_every=args.progress_every,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    elapsed = time.monotonic() - t0

    # Persist report.
    report_path = logs_dir / "qwen_ocr_report.json"
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    logger.info("=== Qwen-VL-OCR complete in %.1fs ===", elapsed)
    logger.info("  total=%d  ok=%d  empty=%d  failed=%d",
                report.pages_total, report.pages_processed,
                report.pages_empty, report.pages_failed)
    logger.info("  avg_s/page=%.2f", report.avg_seconds_per_page)
    logger.info("  report → %s", report_path)

    if report.errors:
        logger.warning("First 5 errors:")
        for e in report.errors[:5]:
            logger.warning("  %s", e)

    driver.close()
    return 1 if report.pages_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
