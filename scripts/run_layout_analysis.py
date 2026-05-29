"""Standalone background runner for PP-StructureV2 layout analysis.

Designed for the long-running full-corpus pass:

::

    caffeinate -dimsu uv run python scripts/run_layout_analysis.py --verbose

The script is idempotent — it picks up where it left off using
``PAGE.layoutStatus``, so killing it mid-run and resuming later just
works. Manuscript pages are pre-classified without inference in <0.1s
each; typeset pages take ~2–5s on Apple-Silicon CPU.

Estimated total runtime: ~2–3 hours for ~2,300 OCR pages.

Flags
-----

- ``--max-pages N``       Hard cap (default: unlimited — full corpus).
- ``--document-id ID``    Only process one document (debugging).
- ``--recompute``         Re-process pages that already have layoutStatus set.
- ``--roles ROLE,...``    Default ``body``; pass ``body,cover`` to include covers.
- ``--log-file PATH``     Tee logs here (default ``logs/layout_analysis.log``).
- ``--report-file PATH``  JSON report path (default ``logs/layout_report.json``).
- ``--verbose``           Set log level to INFO (default: WARNING + layout INFO).
- ``--device CPU/GPU``    PaddlePaddle inference device (default ``cpu``).
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
from apps.backend.ocr.structure import StructureEngine
from apps.backend.pipeline.layout import run_layout_pages
from apps.backend.storage.minio_client import get_minio_client


def _get_paddle_version() -> str:
    try:
        import paddleocr
        return getattr(paddleocr, "__version__", "?")
    except Exception:  # noqa: BLE001
        return "?"


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
    # Always show layout orchestrator progress at INFO.
    logging.getLogger("apps.backend.pipeline.layout").setLevel(logging.INFO)


_INTERRUPTED = False


def _handle_sigint(signum, frame):  # noqa: ARG001
    global _INTERRUPTED
    print("\nReceived SIGINT — finishing current page, then exiting.", flush=True)
    _INTERRUPTED = True


def main() -> int:
    load_dotenv(_REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Background PP-StructureV2 layout analysis runner"
    )
    parser.add_argument("--max-pages", type=int, default=None, help="Hard page cap (default: all)")
    parser.add_argument("--document-id", type=str, default=None, help="Process one document only")
    parser.add_argument("--recompute", action="store_true", help="Re-process already-processed pages")
    parser.add_argument(
        "--roles", type=str, default="body",
        help="Comma-separated PAGE roles (default: body)",
    )
    parser.add_argument("--log-file", type=str, default="logs/layout_analysis.log")
    parser.add_argument("--report-file", type=str, default="logs/layout_report.json")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device (cpu/gpu)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_file = Path(args.log_file).expanduser().resolve() if args.log_file else None
    _setup_logging(log_file, verbose=args.verbose)

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    print(f"Starting layout analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  model    : PP-StructureV2/PP-StructureV2 (paddleocr {_get_paddle_version()})")
    print(f"  device   : {args.device}")
    print(f"  max_pages: {args.max_pages}")
    print(f"  document : {args.document_id}")
    print(f"  recompute: {args.recompute}")
    print(f"  log      : {log_file}", flush=True)

    driver = get_driver()
    minio_client = get_minio_client()

    # Load model once — this triggers PP-DocLayout_plus-L download on first run.
    print("\nLoading PP-StructureV2 pipeline (first run downloads models)...", flush=True)
    engine = StructureEngine(device=args.device)
    t0 = time.monotonic()
    engine._load()
    print(f"Model loaded in {time.monotonic() - t0:.1f}s\n", flush=True)

    started = time.monotonic()
    report = run_layout_pages(
        driver=driver,
        minio_client=minio_client,
        engine=engine,
        recompute_existing=args.recompute,
        document_id=args.document_id,
        max_pages=args.max_pages,
        progress_every=20,
    )
    elapsed = time.monotonic() - started

    payload = report.to_dict()
    payload["interrupted"] = _INTERRUPTED
    payload["wall_clock_seconds"] = round(elapsed, 1)
    payload["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    report_file = Path(args.report_file).expanduser().resolve()
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print()
    print(
        f"DONE — {report.pages_ok}/{report.pages_total} ok "
        f"in {elapsed/60:.1f} min "
        f"({report.avg_seconds_per_page:.2f}s/page avg)"
    )
    print(
        f"       empty={report.pages_empty} "
        f"failed={report.pages_failed} "
        f"manuscript={report.pages_manuscript}"
    )
    print(f"Report: {report_file}", flush=True)

    return 0 if report.pages_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
