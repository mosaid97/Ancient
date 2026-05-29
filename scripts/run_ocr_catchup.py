"""Watch for preprocessing to finish, then run PaddleOCR + DeepSeek-OCR catch-up.

Usage::

    caffeinate -dimsu uv run python scripts/run_ocr_catchup.py

Polls Neo4j every 60 s until no body PAGE nodes have ``preprocessingStatus IS NULL``,
then runs PaddleOCR followed by DeepSeek-OCR on any pages that still lack OCR results.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.ocr.paddle import PaddleOCREngine
from apps.backend.pipeline.extract import run_paddle_pages, run_deepseek_pages
from apps.backend.storage.minio_client import get_minio_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/ocr_catchup.log"),
    ],
)
for noisy in ("urllib3", "botocore", "boto3", "s3transfer", "minio", "httpx"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 60


def _count_pending_preprocess(driver) -> int:
    with driver.session() as s:
        row = s.run(
            "MATCH (p:PAGE) WHERE p.mode='ocr' AND p.role='body' "
            "AND p.preprocessingStatus IS NULL RETURN count(p) AS n"
        ).single()
        return row["n"] if row else 0


def _count_needs_ocr(driver) -> tuple[int, int]:
    """Returns (needs_paddle, needs_deepseek)."""
    with driver.session() as s:
        row = s.run("""
            MATCH (p:PAGE)
            WHERE p.mode='ocr' AND p.role='body'
              AND p.preprocessedImageUri IS NOT NULL
            RETURN
              count(CASE WHEN p.paddleOcrStatus IS NULL OR p.paddleOcrStatus='failed' THEN 1 END) AS paddle,
              count(CASE WHEN p.deepseekOcrStatus IS NULL OR p.deepseekOcrStatus='failed' THEN 1 END) AS deepseek
        """).single()
        return (row["paddle"] if row else 0), (row["deepseek"] if row else 0)


def main() -> int:
    driver = get_driver()
    minio_client = get_minio_client()

    # ── 1. Wait for preprocessing to finish ──────────────────────────────────
    pending = _count_pending_preprocess(driver)
    if pending > 0:
        logger.info("Waiting for preprocessing to finish (%d pages remaining)...", pending)
        while True:
            time.sleep(POLL_INTERVAL_S)
            pending = _count_pending_preprocess(driver)
            logger.info("Preprocessing remaining: %d", pending)
            if pending == 0:
                break
        logger.info("Preprocessing complete — starting OCR catch-up.")
    else:
        logger.info("Preprocessing already complete — starting OCR immediately.")

    needs_paddle, needs_deepseek = _count_needs_ocr(driver)
    logger.info("Pages needing PaddleOCR: %d  |  DeepSeek-OCR: %d", needs_paddle, needs_deepseek)

    # ── 2. PaddleOCR ─────────────────────────────────────────────────────────
    if needs_paddle > 0:
        logger.info("Loading PaddleOCR engine...")
        paddle_engine = PaddleOCREngine()
        paddle_report = run_paddle_pages(
            driver=driver,
            minio_client=minio_client,
            engine=paddle_engine,
        )
        logger.info(
            "PaddleOCR done: %d ok, %d empty, %d failed",
            paddle_report.pages_ok,
            paddle_report.pages_empty,
            paddle_report.pages_failed,
        )
        print(
            f"\nPaddleOCR — {paddle_report.pages_ok}/{paddle_report.pages_total} ok "
            f"({paddle_report.pages_empty} empty, {paddle_report.pages_failed} failed)"
        )
    else:
        logger.info("PaddleOCR: nothing to do.")

    # ── 3. DeepSeek-OCR ──────────────────────────────────────────────────────
    if needs_deepseek > 0:
        logger.info("Starting DeepSeek-OCR...")
        deepseek_report = run_deepseek_pages(
            driver=driver,
            minio_client=minio_client,
        )
        logger.info(
            "DeepSeek done: %d ok, %d empty, %d failed",
            deepseek_report.pages_ok,
            deepseek_report.pages_empty,
            deepseek_report.pages_failed,
        )
        print(
            f"DeepSeek-OCR — {deepseek_report.pages_ok}/{deepseek_report.pages_total} ok "
            f"({deepseek_report.pages_empty} empty, {deepseek_report.pages_failed} failed)"
        )
    else:
        logger.info("DeepSeek-OCR: nothing to do.")

    logger.info("OCR catch-up complete.")
    driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
