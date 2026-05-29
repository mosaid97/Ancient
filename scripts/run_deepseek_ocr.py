"""Standalone background runner for DeepSeek-OCR (Silra) over the corpus.

Companion to :mod:`scripts.run_paddle_ocr`. Both writers target distinct
``(:PAGE)`` properties (``paddleOcr*`` vs ``deepseekOcr*``) so the two
runners can execute concurrently without contention.

Typical usage — after landing, with internet::

    caffeinate -dimsu uv run python scripts/run_deepseek_ocr.py \\
        --log-file logs/deepseek_ocr.log

The script is idempotent — it picks up where it left off using the
``deepseekOcrStatus`` flag, so killing it mid-run (e.g. flaky Wi-Fi)
and resuming later just works. The selector also includes
``deepseekOcrStatus = 'failed'``, so transient API errors are
automatically retried on the next invocation.

Flags
-----

- ``--max-pages N``    Cap (default: unlimited).
- ``--document-id ID`` Only process one document.
- ``--recompute``      Re-OCR pages that already have a deepseek result.
- ``--roles ROLE,..``  Default ``body``; pass ``body,marginalia`` to
  OCR the marginal annotations too.
- ``--concurrency N``  Number of Silra requests in flight at once
  (default: 1 — safe for free-tier rate limits; bump to 2-4 for
  paid tiers).
- ``--max-tokens N``   Per-response cap (default: 4096).
- ``--timeout S``      Per-request timeout (default: 180s — DeepSeek-OCR
  on dense pages can legitimately take 60+ s).
- ``--log-file PATH``  Tee logs to this file.

Cost note: at ~7-12s/page and 2300 pages, a full pass is a 5-8 hour
job. Run overnight rather than in a flight window.
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
from apps.backend.llm.silra import get_silra_client
from apps.backend.pipeline.extract import run_deepseek_pages
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
    parser = argparse.ArgumentParser(description="Background DeepSeek-OCR runner (Silra)")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--document-id", type=str, default=None)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--roles", type=str, default="body",
        help="Comma-separated PAGE roles to OCR (default: body)",
    )
    parser.add_argument("--bucket", type=str, default=None)
    parser.add_argument("--log-file", type=str, default="logs/deepseek_ocr.log")
    parser.add_argument("--report-file", type=str, default="logs/deepseek_ocr_report.json")
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Per-response cap (default 4096 — long enough for dense pages).",
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0,
        help="Per-request timeout in seconds (default 180).",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_file = Path(args.log_file).expanduser().resolve() if args.log_file else None
    _setup_logging(log_file, verbose=args.verbose or True)

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    bucket = args.bucket or os.getenv("MINIO_BUCKET_PAGES", "ancient-pages")
    roles = tuple(r.strip() for r in args.roles.split(",") if r.strip())

    print(f"Starting DeepSeek-OCR background run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  bucket     : {bucket}")
    print(f"  roles      : {roles}")
    print(f"  max_pages  : {args.max_pages}")
    print(f"  document   : {args.document_id}")
    print(f"  recompute  : {args.recompute}")
    print(f"  max_tokens : {args.max_tokens}")
    print(f"  timeout    : {args.timeout}s")
    print(f"  log        : {log_file}", flush=True)

    driver = get_driver()
    minio_client = get_minio_client()
    silra_client = get_silra_client(timeout=args.timeout)

    started = time.monotonic()
    report = run_deepseek_pages(
        driver=driver,
        minio_client=minio_client,
        silra_client=silra_client,
        bucket=bucket,
        document_id=args.document_id,
        max_pages=args.max_pages,
        recompute_existing=args.recompute,
        roles=roles,
        progress_every=args.progress_every,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
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
