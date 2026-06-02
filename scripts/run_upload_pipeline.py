#!/usr/bin/env python3
"""Upload pipeline runner (Track E).

Spawned by ``apps/backend/api/routers/upload.py`` for each website upload.
Chains the existing orchestrators scoped to one ``document_id`` and updates
the ``UPLOAD_JOB`` node so the web UI can poll progress:

    ingest -> (preprocess -> Paddle + {qwen|deepseek|custom} -> fuse -> layout)

OCR is skipped entirely when ``--ocr-required 0`` (native text path). At the
end ``DOCUMENT.status`` is set to ``awaiting_review`` and the job is marked
``awaiting_review`` so the HITL fusion screen can pick it up.

Custom OCR credentials are read from the environment
(``CUSTOM_OCR_BASE_URL`` / ``CUSTOM_OCR_API_KEY`` / ``CUSTOM_OCR_MODEL``) and
are never persisted.

Usage (normally invoked by the API, but runnable manually):
    uv run python scripts/run_upload_pipeline.py \\
        --job-id abc123 --path staging/abc123/scan.pdf \\
        --doc-type book --ocr-model qwen --language chinese \\
        --tier primary --ocr-required 1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.pipeline import upload_job
from apps.backend.pipeline.ingest import _document_id_from_path, ingest_path, upsert_topic
from apps.backend.storage import get_minio_client

log = logging.getLogger("run_upload_pipeline")


_SET_DOC_STATUS = """
MATCH (d:DOCUMENT {id: $doc_id})
SET d.status = $status, d.statusUpdatedAt = timestamp()
RETURN d.id AS id
"""

_COUNT_OCR_PAGES = """
MATCH (p:PAGE {documentId: $doc_id})
WHERE p.mode = 'ocr'
RETURN count(p) AS n
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Process one website upload")
    p.add_argument("--job-id", required=True)
    p.add_argument("--path", required=True, help="Path to the staged upload file")
    p.add_argument("--doc-type", default="book")
    p.add_argument("--ocr-model", default="qwen", choices=["qwen", "deepseek", "custom"])
    p.add_argument("--language", default="chinese")
    p.add_argument("--tier", default="primary", choices=["primary", "secondary"])
    p.add_argument("--ocr-required", default="1", choices=["0", "1"])
    return p


def _set_doc_status(driver, doc_id: str, status: str) -> None:
    with driver.session() as s:
        s.run(_SET_DOC_STATUS, doc_id=doc_id, status=status).consume()


def main() -> None:  # noqa: C901 — sequential pipeline
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    # Catch-all so an unexpected crash (or a failure in a code path without its
    # own try/except) marks the job 'failed' instead of leaving it stuck in
    # 'running' forever, which the UI would poll indefinitely.
    try:
        _run(args)
    except Exception as exc:  # noqa: BLE001
        log.exception("upload pipeline crashed")
        try:
            driver = get_driver()
            upload_job.update_job(
                driver, args.job_id, status="failed",
                error=f"pipeline crashed: {type(exc).__name__}: {exc}",
            )
        except Exception:  # noqa: BLE001 — best-effort; nothing else we can do
            log.exception("could not mark job failed")


def _run(args) -> None:  # noqa: C901 — sequential pipeline
    job_id = args.job_id
    ocr_required = args.ocr_required == "1"
    repo_root = Path(__file__).resolve().parent.parent
    src_path = Path(args.path)
    if not src_path.is_absolute():
        src_path = repo_root / src_path

    driver = get_driver()

    if not src_path.exists():
        log.error("Upload file not found: %s", src_path)
        upload_job.update_job(driver, job_id, status="failed", error=f"file not found: {src_path}")
        return

    doc_id = _document_id_from_path(src_path)
    log.info("job=%s doc_id=%s path=%s ocr_required=%s model=%s",
             job_id, doc_id, src_path.name, ocr_required, args.ocr_model)
    upload_job.update_job(driver, job_id, document_id=doc_id, stage="ingest", pct=5.0)

    t_total = time.time()
    minio = get_minio_client()

    # -- Ingest ------------------------------------------------------------
    try:
        upsert_topic(driver)
        report = ingest_path(
            src_path,
            driver=driver,
            minio_client=minio,
            tier=args.tier,
            document_id=doc_id,
            repo_root=repo_root,
        )
        log.info("ingested %d pages (%d ocr, %d native)",
                 report.pages_total, report.pages_ocr, report.pages_native)
    except Exception as exc:  # noqa: BLE001
        log.exception("ingest failed")
        upload_job.update_job(driver, job_id, status="failed", stage="ingest",
                              error=f"ingest failed: {exc}")
        return

    upload_job.update_job(driver, job_id, stage="ingest", pct=20.0)

    # How many OCR-bound pages do we actually have?
    with driver.session() as s:
        rec = s.run(_COUNT_OCR_PAGES, doc_id=doc_id).single()
    n_ocr = rec["n"] if rec else 0

    if ocr_required and n_ocr > 0:
        try:
            _run_ocr_stages(driver, minio, doc_id, job_id, args)
        except Exception as exc:  # noqa: BLE001
            log.exception("OCR pipeline failed")
            upload_job.update_job(driver, job_id, status="failed", stage="ocr",
                                  error=f"OCR pipeline failed: {exc}")
            return
    else:
        log.info("Skipping OCR (ocr_required=%s, ocr_pages=%d)", ocr_required, n_ocr)
        upload_job.update_job(driver, job_id, stage="layout", pct=95.0)

    # -- Gate for human review --------------------------------------------
    _set_doc_status(driver, doc_id, "awaiting_review")
    elapsed = time.time() - t_total
    upload_job.update_job(driver, job_id, status="awaiting_review", stage="done", pct=100.0)
    log.info("=== job %s done in %.1f s — awaiting_review ===", job_id, elapsed)


def _run_ocr_stages(driver, minio, doc_id: str, job_id: str, args) -> None:
    """Preprocess -> Paddle + LLM (+custom) -> fuse -> layout for one document."""
    from apps.backend.ocr.paddle import PaddleOCREngine
    from apps.backend.pipeline.extract import (
        run_custom_pages,
        run_deepseek_pages,
        run_paddle_pages,
        run_qwen_pages,
    )
    from apps.backend.pipeline.fusion import fuse_pages
    from apps.backend.pipeline.layout import run_layout_pages
    from apps.backend.pipeline.preprocess import preprocess_pages

    # -- Preprocess --------------------------------------------------------
    upload_job.update_job(driver, job_id, stage="preprocess", pct=30.0)
    t0 = time.time()
    pre = preprocess_pages(driver=driver, minio_client=minio, document_id=doc_id)
    log.info("preprocessed=%d failed=%d in %.1fs",
             pre.pages_processed, pre.pages_failed, time.time() - t0)

    # -- PaddleOCR (alignment anchor) -------------------------------------
    upload_job.update_job(driver, job_id, stage="ocr", pct=45.0)
    paddle_engine = PaddleOCREngine()
    # Route the Paddle model by upload language (japanese pages need the
    # 'japan' model; everything else uses the Chinese model).
    paddle_lang = "japan" if args.language == "japanese" else "ch"
    paddle_engine.warmup(langs=[paddle_lang])
    pr = run_paddle_pages(driver=driver, minio_client=minio, engine=paddle_engine, document_id=doc_id)
    log.info("paddle ok=%d failed=%d", pr.pages_processed, pr.pages_failed)

    # -- LLM OCR engine ----------------------------------------------------
    upload_job.update_job(driver, job_id, stage="ocr", pct=60.0)
    if args.ocr_model == "deepseek":
        dr = run_deepseek_pages(driver=driver, minio_client=minio, document_id=doc_id)
        log.info("deepseek ok=%d failed=%d", dr.pages_processed, dr.pages_failed)
    elif args.ocr_model == "custom":
        base_url = os.getenv("CUSTOM_OCR_BASE_URL", "")
        api_key = os.getenv("CUSTOM_OCR_API_KEY", "")
        model = os.getenv("CUSTOM_OCR_MODEL", "")
        if base_url and api_key and model:
            # Run Qwen too so there's always a built-in LLM column to fuse with.
            qr = run_qwen_pages(driver=driver, minio_client=minio, document_id=doc_id)
            log.info("qwen ok=%d failed=%d", qr.pages_processed, qr.pages_failed)
            cr = run_custom_pages(
                driver=driver, minio_client=minio,
                base_url=base_url, api_key=api_key, model=model, document_id=doc_id,
            )
            log.info("custom ok=%d failed=%d", cr.pages_processed, cr.pages_failed)
        else:
            log.warning("custom OCR requested but credentials missing — falling back to qwen")
            qr = run_qwen_pages(driver=driver, minio_client=minio, document_id=doc_id)
            log.info("qwen ok=%d failed=%d", qr.pages_processed, qr.pages_failed)
    else:  # qwen (default)
        qr = run_qwen_pages(driver=driver, minio_client=minio, document_id=doc_id)
        log.info("qwen ok=%d failed=%d", qr.pages_processed, qr.pages_failed)

    # -- Fusion ------------------------------------------------------------
    upload_job.update_job(driver, job_id, stage="fuse", pct=80.0)
    fr = fuse_pages(driver, document_id=doc_id)
    log.info("fused dual=%d single=%d failed=%d",
             fr.pages_dual_fused, fr.pages_single_engine, fr.pages_failed)

    # -- Layout ------------------------------------------------------------
    upload_job.update_job(driver, job_id, stage="layout", pct=92.0)
    lr = run_layout_pages(driver, minio, document_id=doc_id)
    log.info("layout ok=%d", lr.pages_ok)


if __name__ == "__main__":
    main()
