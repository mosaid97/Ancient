"""Re-run DeepSeek-OCR on pages blocked as empty/garbage, with the new
inline hallucination validator active.

Context:
    After two rounds of validation-aware re-runs, 913 pages remain at
    ``deepseekOcrStatus='empty'``.  827 of those have PaddleOCR content
    (``paddleOcrCharCount > 10``), confirming real text exists on the page.
    The dominant error reasons are:

    * ``validation_failed:html_table_markup`` (455) — bibliography/reference
      sections that DeepSeek consistently formats as HTML.  The updated
      system prompt now explicitly bans HTML and instructs plain-text output.
    * ``validation_failed:low_cjk_ratio:0.000`` (214) — mix of English
      abstract pages (expected; validator will still block) and column-label
      garbage tables (worth a retry).
    * ``(no error set)`` (119) — manually cleaned before the validator existed.
    * ``internal_token_leakage``, ``sequential_number_table``, etc. — worth
      retrying; most are model artefacts that may not recur.

    With ``--require-paddle-text 11`` (the default), the 86 genuinely blank
    pages (Paddle also got ≤10 chars) are skipped, saving ~86 API calls.

    After this re-run, run the fusion notebook (``03b_fusion.ipynb``) to
    incorporate improved DeepSeek outputs into the 3-way ``textFused``.

Usage::

    # Dry-run: shows page count then exits
    uv run python scripts/rerun_deepseek_empty.py --dry-run

    # Full re-run (background-safe, keeps mac awake)
    caffeinate -dimsu uv run python scripts/rerun_deepseek_empty.py

    # Retry including blank pages (all 913)
    uv run python scripts/rerun_deepseek_empty.py --require-paddle-text 0

    # Single document
    uv run python scripts/rerun_deepseek_empty.py --document-id <doc_id>

    # Smoke test (first 20 pages)
    uv run python scripts/rerun_deepseek_empty.py --max-pages 20
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv
load_dotenv(_REPO / ".env")

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.llm.silra import get_silra_client
from apps.backend.pipeline.extract import run_deepseek_pages
from apps.backend.storage.minio_client import get_minio_client

_logs_dir = _REPO / "logs"
_logs_dir.mkdir(exist_ok=True)

# INFO → stdout + main log; DEBUG (raw OCR responses) → separate file.
_root_handler_stdout = logging.StreamHandler(sys.stdout)
_root_handler_stdout.setLevel(logging.INFO)
_root_handler_file = logging.FileHandler(_logs_dir / "deepseek_rerun_empty.log", mode="a")
_root_handler_file.setLevel(logging.INFO)
_debug_handler = logging.FileHandler(_logs_dir / "deepseek_rerun_empty_debug.log", mode="a")
_debug_handler.setLevel(logging.DEBUG)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[_root_handler_stdout, _root_handler_file, _debug_handler],
)
logger = logging.getLogger("rerun_deepseek_empty")

_STOP = False


def _handle_signal(signum: int, _frame) -> None:  # noqa: ANN001
    global _STOP  # noqa: PLW0603
    _STOP = True
    logger.info("Signal %d — finishing current page then stopping.", signum)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run DeepSeek-OCR on empty/cleaned pages (with validator)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--require-paddle-text", type=int, default=11, metavar="MIN_CHARS",
        help=(
            "Only retry pages where paddleOcrCharCount >= MIN_CHARS. "
            "Default 11 skips genuinely blank pages. Set 0 to retry all 913."
        ),
    )
    args = parser.parse_args()

    logs_dir = _REPO / "logs"
    logs_dir.mkdir(exist_ok=True)

    driver = get_driver()

    if args.dry_run:
        with driver.session() as s:
            paddle_clause = (
                f"AND p.paddleOcrCharCount >= {args.require_paddle_text} "
                if args.require_paddle_text > 0 else ""
            )
            n = s.run(
                f"""
                MATCH (p:PAGE)
                WHERE p.mode = 'ocr'
                  AND p.preprocessedImageUri IS NOT NULL
                  AND p.role IN ['body']
                  AND p.deepseekOcrStatus = 'empty'
                  AND ($document_id IS NULL OR p.documentId = $document_id)
                  {paddle_clause}
                RETURN count(p) AS n
                """,
                document_id=args.document_id,
            ).single()["n"]
        print(
            f"[DRY RUN] DeepSeek empty pages eligible for re-run: {n}"
            + (f" (require_paddle_text >= {args.require_paddle_text})" if args.require_paddle_text > 0 else " (all)")
        )
        driver.close()
        return 0

    minio = get_minio_client()
    silra = get_silra_client(timeout=args.timeout)

    logger.info(
        "Re-running DeepSeek-OCR on empty pages | max_pages=%s | doc=%s | "
        "require_paddle_text=%d",
        args.max_pages, args.document_id, args.require_paddle_text,
    )

    t0 = time.monotonic()
    report = run_deepseek_pages(
        driver=driver,
        minio_client=minio,
        silra_client=silra,
        document_id=args.document_id,
        max_pages=args.max_pages,
        recompute_existing=False,
        include_empty=True,
        require_paddle_text=args.require_paddle_text,
        progress_every=args.progress_every,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    elapsed = time.monotonic() - t0

    report_path = logs_dir / "deepseek_rerun_empty_report.json"
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    logger.info("=== DeepSeek re-run complete in %.1fs ===", elapsed)
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
