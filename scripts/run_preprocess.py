"""Standalone background runner for Phase-2 preprocessing over missed pages.

Usage::

    caffeinate -dimsu uv run python scripts/run_preprocess.py

Idempotent — only processes pages where ``preprocessingStatus IS NULL`` or
``preprocessingStatus='failed'``. Safe to re-run at any time.

Flags
-----
- ``--document-id ID``  Only preprocess one document.
- ``--max-pages N``     Cap the number of pages processed.
- ``--recompute``       Re-run even pages that already have a result.
- ``--log-file PATH``   Log file path (default ``logs/preprocess_retry.log``).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.pipeline.preprocess import preprocess_pages
from apps.backend.storage.minio_client import get_minio_client

_SHUTDOWN = False


def _handle_sigint(sig, frame):  # noqa: ANN001
    global _SHUTDOWN
    print("\n[interrupted] finishing current page then stopping…", flush=True)
    _SHUTDOWN = True


def _setup_logging(log_file: str) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    for noisy in ("urllib3", "botocore", "boto3", "s3transfer", "minio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-2 preprocessing background runner")
    parser.add_argument("--document-id", default=None, help="Only process one document")
    parser.add_argument("--max-pages", type=int, default=None, help="Page cap")
    parser.add_argument("--recompute", action="store_true", help="Re-run already-preprocessed pages")
    parser.add_argument("--log-file", default="logs/preprocess_retry.log")
    args = parser.parse_args()

    _setup_logging(args.log_file)
    signal.signal(signal.SIGINT, _handle_sigint)

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting preprocessing at {started}")
    print(f"  document : {args.document_id or 'ALL'}")
    print(f"  max_pages: {args.max_pages or 'unlimited'}")
    print(f"  recompute: {args.recompute}")
    print(f"  log      : {args.log_file}", flush=True)

    driver = get_driver()
    minio_client = get_minio_client()

    report = preprocess_pages(
        driver=driver,
        minio_client=minio_client,
        document_id=args.document_id,
        max_pages=args.max_pages,
        recompute_existing=args.recompute,
    )

    avg = report.duration_seconds / max(report.pages_processed, 1)
    print(
        f"\nDONE — {report.pages_processed}/{report.pages_total} processed in "
        f"{report.duration_seconds / 60:.1f} min "
        f"({avg:.2f}s/page avg)"
    )
    print(f"       skipped={report.pages_skipped} failed={report.pages_failed}")
    if report.errors:
        print("ERRORS (first 10):")
        for e in report.errors[:10]:
            print(f"  {e}")

    return 0 if report.pages_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
