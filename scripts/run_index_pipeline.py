#!/usr/bin/env python3
"""Index pipeline runner (Track E).

Spawned by ``POST /api/hitl/ocr/approve`` after a reviewer approves an
uploaded document's OCR/fusion. Chains the indexing stages:

    chunk -> embed -> keyword extraction

then sets ``DOCUMENT.status='indexed'`` so ``--document-id`` becomes
searchable and selectable in the interactive hover view.

IMPORTANT — scope: the underlying orchestrators (``chunk_pages`` /
``embed_chunks`` / ``run_keyword_extraction``) currently operate
**corpus-wide** — they do not accept a ``document_id`` filter. They are
idempotency-gated (chunk: pages without CHUNKs; embed: chunks without an
embedding; keywords: chunks with ``mentionStatus IS NULL``), so in steady
state a run only *processes* the newly-approved document's pending items,
but it *scans* the whole corpus to find them. Two near-simultaneous
approvals therefore run overlapping corpus-wide sweeps with no lock; the
idempotency gates make this correct but wasteful. Adding a real per-document
filter to those three orchestrators is tracked as future work. The
``--document-id`` here only controls the final ``status='indexed'`` write.

Usage:
    uv run python scripts/run_index_pipeline.py --document-id <doc_id>
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from apps.backend.graph.neo4j_client import get_driver

log = logging.getLogger("run_index_pipeline")

_SET_STATUS = """
MATCH (d:DOCUMENT {id: $doc_id})
SET d.status = $status, d.statusUpdatedAt = timestamp()
RETURN d.id AS id
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Index one approved document")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--skip-keywords", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    from apps.backend.pipeline.chunk import chunk_pages
    from apps.backend.pipeline.embed import embed_chunks
    from apps.backend.pipeline.keywords import run_keyword_extraction

    doc_id = args.document_id
    driver = get_driver()
    t_total = time.time()
    log.info("Indexing document %s", doc_id)

    try:
        t0 = time.time()
        chunk_rep = chunk_pages(driver)
        log.info("chunked: %s in %.1fs", getattr(chunk_rep, "chunks_created", "?"), time.time() - t0)

        t0 = time.time()
        embed_rep = embed_chunks(driver)
        log.info("embedded: %s in %.1fs", getattr(embed_rep, "chunks_embedded", "?"), time.time() - t0)

        if not args.skip_keywords:
            t0 = time.time()
            kw_rep = run_keyword_extraction(driver, max_workers=5)
            log.info("keywords: ok=%s in %.1fs", getattr(kw_rep, "chunks_ok", "?"), time.time() - t0)
    except Exception:
        log.exception("index pipeline failed")
        with driver.session() as s:
            s.run(_SET_STATUS, doc_id=doc_id, status="approved").consume()
        return

    with driver.session() as s:
        s.run(_SET_STATUS, doc_id=doc_id, status="indexed").consume()
    log.info("=== document %s indexed in %.1fs ===", doc_id, time.time() - t_total)


if __name__ == "__main__":
    main()
