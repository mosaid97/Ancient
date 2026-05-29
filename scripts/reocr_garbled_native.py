#!/usr/bin/env python3
"""Re-pipeline a PDF whose native text layer is garbled (broken font CMap).

Runs the full Phase 1→6 pipeline on a single document, forcing OCR mode
even on pages that have an embedded (but corrupted) text layer.

Steps:
  1. Delete existing PAGE, CHUNK, KEYWORD MENTION edges for this document
     (DOCUMENT and CHAPTER/SECTION spine are preserved and re-stamped).
  2. Re-ingest with force_ocr=True → all pages become OCR-bound.
  3. Run preprocessing (Phase 2).
  4. Run PaddleOCR (Phase 3a) + Qwen-VL-OCR (Phase 3c).
  5. Run 3-way fusion (Phase 3d).
  6. Run layout analysis (Phase 4).
  7. Re-chunk (Phase 5a).
  8. Re-embed (Phase 5b).
  9. Re-run keyword extraction (Phase 6).

Usage:
    uv run python scripts/reocr_garbled_native.py \\
        --path "raw/Secondary/specific_卫官/爱宕元_唐代的官荫入仕：以卫官之路为中心_日本中青年学者论中国史六朝隋唐卷.pdf" \\
        --dry-run

    caffeinate -dimsu uv run python scripts/reocr_garbled_native.py \\
        --path "raw/Secondary/specific_卫官/爱宕元_唐代的官荫入仕：以卫官之路为中心_日本中青年学者论中国史六朝隋唐卷.pdf"

Estimated runtime: ~530 pages × (preprocess ~2 s + OCR ~25 s + ...) ≈ 5–8 hours.
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

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.ocr.paddle import PaddleOCREngine
from apps.backend.pipeline.chunk import chunk_pages
from apps.backend.pipeline.embed import embed_chunks
from apps.backend.pipeline.extract import run_paddle_pages, run_qwen_pages
from apps.backend.pipeline.fusion import fuse_pages
from apps.backend.pipeline.ingest import _document_id_from_path, ingest_path, upsert_topic
from apps.backend.pipeline.keywords import run_keyword_extraction
from apps.backend.pipeline.lang_detect import detect_pages
from apps.backend.pipeline.layout import run_layout_pages
from apps.backend.pipeline.preprocess import preprocess_pages
from apps.backend.storage import get_minio_client


# ---------------------------------------------------------------------------
# Cypher: delete stale PAGE→CHUNK→MENTION nodes for one document
# ---------------------------------------------------------------------------

_DELETE_CHUNKS_AND_MENTIONS = """
MATCH (doc:DOCUMENT {id: $doc_id})-[:CONSIST_OF]->(:CHAPTER)-[:INCLUDE]->(:SECTION)
      -[:INCLUDE]->(:PAGE)-[:HAS]->(c:CHUNK)
DETACH DELETE c
"""

_DELETE_PAGES = """
MATCH (doc:DOCUMENT {id: $doc_id})-[:CONSIST_OF]->(:CHAPTER)-[:INCLUDE]->(:SECTION)
      -[:INCLUDE]->(p:PAGE)
DETACH DELETE p
"""

_COUNT_PAGES = """
MATCH (doc:DOCUMENT {id: $doc_id})-[:CONSIST_OF]->(:CHAPTER)-[:INCLUDE]->(:SECTION)
      -[:INCLUDE]->(p:PAGE)
RETURN count(p) AS n
"""


def _get_minio():
    return get_minio_client()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Re-OCR a garbled-native-text PDF")
    p.add_argument("--path", required=True,
                   help="Relative path to the PDF (e.g. raw/Secondary/…/foo.pdf)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan and counts; make no changes")
    p.add_argument("--skip-ocr-paddle", action="store_true",
                   help="Skip PaddleOCR pass (e.g. already done)")
    p.add_argument("--skip-ocr-qwen", action="store_true",
                   help="Skip Qwen-VL-OCR pass")
    p.add_argument("--skip-fusion", action="store_true")
    p.add_argument("--skip-layout", action="store_true")
    p.add_argument("--skip-chunk", action="store_true")
    p.add_argument("--skip-embed", action="store_true")
    p.add_argument("--skip-keywords", action="store_true")
    p.add_argument("--skip-ingest", action="store_true",
                   help="Skip steps 1-3 (delete/re-ingest/lang-detect) — use when pages are already ingested")
    p.add_argument("--log-file", default="logs/reocr_garbled_native.log")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:  # noqa: C901 — sequential pipeline, intentionally long
    args = _build_parser().parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    repo_root = Path(__file__).resolve().parent.parent
    pdf_path = repo_root / args.path
    if not pdf_path.exists():
        log.error("File not found: %s", pdf_path)
        sys.exit(1)

    doc_id = _document_id_from_path(pdf_path)
    log.info("Document: %s", pdf_path.name)
    log.info("Document ID: %s", doc_id)
    log.info("Dry run: %s", args.dry_run)

    driver = get_driver()
    minio = _get_minio()

    # -----------------------------------------------------------------------
    # 0. Pre-flight counts
    # -----------------------------------------------------------------------
    with driver.session() as s:
        existing = s.run(_COUNT_PAGES, doc_id=doc_id).single()
        page_count = existing["n"] if existing else 0
    log.info("Existing PAGE nodes: %d", page_count)

    if args.dry_run:
        log.info("[DRY RUN] Would delete %d PAGE nodes + their CHUNKs and MENTIONs", page_count)
        log.info("[DRY RUN] Would re-ingest %s with force_ocr=True", pdf_path.name)
        log.info("[DRY RUN] Exiting.")
        driver.close()
        return

    t_total = time.time()

    if args.skip_ingest:
        log.info("Steps 1-3 skipped (--skip-ingest).")
    else:
        # -------------------------------------------------------------------
        # 1. Delete stale CHUNK/MENTION then PAGE nodes
        # -------------------------------------------------------------------
        log.info("Step 1: Deleting stale CHUNK + MENTION nodes…")
        with driver.session() as s:
            s.run(_DELETE_CHUNKS_AND_MENTIONS, doc_id=doc_id).consume()
        log.info("Step 1: Deleting PAGE nodes…")
        with driver.session() as s:
            s.run(_DELETE_PAGES, doc_id=doc_id).consume()
        log.info("Step 1: Done.")

        # -------------------------------------------------------------------
        # 2. Re-ingest with force_ocr=True
        # -------------------------------------------------------------------
        log.info("Step 2: Re-ingesting with force_ocr=True…")
        upsert_topic(driver)
        report = ingest_path(
            pdf_path,
            driver=driver,
            minio_client=minio,
            repo_root=repo_root,
            reader_kwargs={"force_ocr": True},
        )
        log.info("Step 2: ingested %d pages (%d ocr, %d native)",
                 report.pages_total, report.pages_ocr, report.pages_native)

        # -------------------------------------------------------------------
        # 3. Language detection (native pages — OCR pages get it post-fusion)
        # -------------------------------------------------------------------
        log.info("Step 3: Language detection for native pages…")
        detect_pages(driver)

    # -----------------------------------------------------------------------
    # 4. Preprocessing (Phase 2)
    # -----------------------------------------------------------------------
    log.info("Step 4: Preprocessing OCR pages…")
    t0 = time.time()
    preproc = preprocess_pages(driver=driver, minio_client=minio, document_id=doc_id)
    log.info("Step 4: preprocessed=%d skipped=%d failed=%d in %.1f s",
             preproc.pages_processed, preproc.pages_skipped, preproc.pages_failed, time.time() - t0)

    # -----------------------------------------------------------------------
    # 5. PaddleOCR (Phase 3a)
    # -----------------------------------------------------------------------
    if not args.skip_ocr_paddle:
        log.info("Step 5: PaddleOCR (loading engine)…")
        paddle_engine = PaddleOCREngine()
        paddle_engine.warmup(langs=["ch"])
        t0 = time.time()
        paddle_rep = run_paddle_pages(
            driver=driver,
            minio_client=minio,
            engine=paddle_engine,
            document_id=doc_id,
        )
        log.info("Step 5: paddle ok=%d failed=%d in %.1f s",
                 paddle_rep.pages_ok, paddle_rep.pages_failed, time.time() - t0)

    # -----------------------------------------------------------------------
    # 6. Qwen-VL-OCR (Phase 3c)
    # -----------------------------------------------------------------------
    if not args.skip_ocr_qwen:
        log.info("Step 6: Qwen-VL-OCR…")
        t0 = time.time()
        qwen_rep = run_qwen_pages(
            driver=driver,
            minio_client=minio,
            document_id=doc_id,
        )
        log.info("Step 6: qwen ok=%d failed=%d in %.1f s",
                 qwen_rep.pages_ok, qwen_rep.pages_failed, time.time() - t0)

    # -----------------------------------------------------------------------
    # 7. Fusion (Phase 3d)
    # -----------------------------------------------------------------------
    if not args.skip_fusion:
        log.info("Step 7: Fusion…")
        t0 = time.time()
        fuse_rep = fuse_pages(driver, document_id=doc_id)
        log.info("Step 7: fused ok=%d in %.1f s", fuse_rep.pages_ok, time.time() - t0)

    # -----------------------------------------------------------------------
    # 8. Layout analysis (Phase 4)
    # -----------------------------------------------------------------------
    if not args.skip_layout:
        log.info("Step 8: Layout analysis…")
        t0 = time.time()
        layout_rep = run_layout_pages(driver, minio, document_id=doc_id)
        log.info("Step 8: layout ok=%d in %.1f s", layout_rep.pages_ok, time.time() - t0)

    # -----------------------------------------------------------------------
    # 9. Chunking (Phase 5a)
    # -----------------------------------------------------------------------
    if not args.skip_chunk:
        log.info("Step 9: Chunking…")
        t0 = time.time()
        chunk_rep = chunk_pages(driver)
        log.info("Step 9: chunked %d pages → %d chunks in %.1f s",
                 chunk_rep.pages_chunked, chunk_rep.chunks_created, time.time() - t0)

    # -----------------------------------------------------------------------
    # 10. Embedding (Phase 5b)
    # -----------------------------------------------------------------------
    if not args.skip_embed:
        log.info("Step 10: Embedding new chunks…")
        t0 = time.time()
        embed_rep = embed_chunks(driver)
        log.info("Step 10: embedded %d chunks in %.1f s",
                 embed_rep.chunks_embedded, time.time() - t0)

    # -----------------------------------------------------------------------
    # 11. Keyword extraction (Phase 6)
    # -----------------------------------------------------------------------
    if not args.skip_keywords:
        log.info("Step 11: Keyword extraction (5 workers)…")
        t0 = time.time()
        kw_rep = run_keyword_extraction(driver, max_workers=5)
        log.info("Step 11: kw ok=%d failed=%d keywords=%d unique=%d in %.1f s",
                 kw_rep.chunks_ok, kw_rep.chunks_failed,
                 kw_rep.keywords_extracted, kw_rep.keywords_unique, time.time() - t0)

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_total
    log.info("=== DONE in %.1f s (%.1f min) ===", elapsed, elapsed / 60)

    report_path = Path("logs/reocr_garbled_native_report.json")
    report_path.write_text(json.dumps({
        "document_id": doc_id,
        "pdf_path": str(args.path),
        "elapsed_seconds": round(elapsed, 1),
    }, ensure_ascii=False, indent=2))
    log.info("Report: %s", report_path)

    driver.close()


if __name__ == "__main__":
    main()
