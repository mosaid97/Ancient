"""Standalone background runner for PaddleOCR over the entire corpus.

Designed for the offline / flight scenario described in the project:

::

    caffeinate -dimsu uv run python scripts/run_paddle_ocr.py \\
        --log-file logs/paddle_ocr.log \\
        --resume

The script is idempotent — it picks up where it left off using the
``paddleOcrStatus`` flag, so killing it mid-flight (e.g. low battery)
and resuming on the next reboot just works.

Flags
-----

- ``--max-pages N``    Cap (default: unlimited).
- ``--document-id ID`` Only process one document.
- ``--recompute``      Re-OCR pages that already have a paddle result.
- ``--roles ROLE,..``  Default ``body``; pass ``body,marginalia`` to
  OCR the marginal annotations too.
- ``--log-file PATH``  Tee logs to this file (default ``logs/paddle_ocr.log``).
- ``--no-progress``    Suppress per-page lines (only periodic summaries).

The script keeps a single PaddleOCR engine in memory so model load
(~30 s) is paid once.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.ocr.paddle import PaddleOCREngine
from apps.backend.pipeline.extract import run_paddle_pages
from apps.backend.storage.minio_client import get_minio_client


def _setup_logging(log_file: Path | None, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    # Pipeline orchestrator emits progress at INFO under apps.backend.pipeline.extract
    logging.getLogger("apps.backend.pipeline.extract").setLevel(logging.INFO)


_INTERRUPTED = False


def _handle_sigint(signum, frame):  # noqa: ARG001
    global _INTERRUPTED
    print("\nReceived SIGINT — finishing current page, then exiting.", flush=True)
    _INTERRUPTED = True


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Background PaddleOCR runner")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--document-id", type=str, default=None)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--roles", type=str, default="body",
        help="Comma-separated PAGE roles to OCR (default: body)",
    )
    parser.add_argument("--bucket", type=str, default=None)
    parser.add_argument("--log-file", type=str, default="logs/paddle_ocr.log")
    parser.add_argument("--report-file", type=str, default="logs/paddle_ocr_report.json")
    parser.add_argument("--model-size", type=str, default="mobile", choices=["mobile", "server"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_file = Path(args.log_file).expanduser().resolve() if args.log_file else None
    _setup_logging(log_file, verbose=args.verbose or True)

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    bucket = args.bucket or os.getenv("MINIO_BUCKET_PAGES", "ancient-pages")
    roles = tuple(r.strip() for r in args.roles.split(",") if r.strip())

    print(f"Starting PaddleOCR background run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  bucket   : {bucket}")
    print(f"  roles    : {roles}")
    print(f"  max_pages: {args.max_pages}")
    print(f"  document : {args.document_id}")
    print(f"  recompute: {args.recompute}")
    print(f"  model    : PP-OCRv5-{args.model_size}, device={args.device}")
    print(f"  log      : {log_file}", flush=True)

    driver = get_driver()
    minio_client = get_minio_client()
    engine = PaddleOCREngine(model_size=args.model_size, device=args.device)
    t0 = time.monotonic()
    engine.warmup(langs=["ch"])  # Preload the Chinese head; japan loads on demand.
    print(f"Engine warmup done in {time.monotonic() - t0:.1f}s", flush=True)

    started = time.monotonic()
    report = run_paddle_pages(
        driver=driver,
        minio_client=minio_client,
        engine=engine,
        bucket=bucket,
        document_id=args.document_id,
        max_pages=args.max_pages,
        recompute_existing=args.recompute,
        roles=roles,
        progress_every=10,
    )
    elapsed = time.monotonic() - started

    payload = report.to_dict()
    payload["interrupted"] = _INTERRUPTED
    payload["wall_clock_seconds"] = round(elapsed, 1)

    report_file = Path(args.report_file).expanduser().resolve()
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print()
    print(
        f"DONE — processed {report.pages_processed}/{report.pages_total} "
        f"in {elapsed/60:.1f} min "
        f"({report.avg_seconds_per_page:.2f}s/page avg), "
        f"empty={report.pages_empty}, failed={report.pages_failed}",
        flush=True,
    )
    print(f"Report written to {report_file}", flush=True)
    return 0 if report.pages_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
