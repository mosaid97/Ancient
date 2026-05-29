"""B2 orchestrator: run the entailment-based Secondary→Primary citation linker.

Usage:
    uv run python scripts/run_citation_linker.py               # full corpus
    uv run python scripts/run_citation_linker.py --max-chunks 50  # smoke test
    uv run python scripts/run_citation_linker.py --threshold 0.90  # stricter

Requires:
    - FlagEmbedding installed (uv add FlagEmbedding)
    - CHUNK.embedding populated for secondary chunks (embed pipeline)
    - chunk_embedding_classical vector index populated with primary chunks
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase

from apps.backend.pipeline.citation_linker import run_citation_linker


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="B2: Entailment-based citation linker")
    p.add_argument("--threshold", type=float, default=0.85, help="Cross-encoder score threshold")
    p.add_argument("--top-k", type=int, default=20, help="Dense retrieval top-K per span")
    p.add_argument("--max-chunks", type=int, default=None, help="Cap for smoke tests")
    p.add_argument("--batch-size", type=int, default=100, help="Secondary chunks per batch")
    p.add_argument("--log-file", default="logs/citation_linker.log")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s %(name)-35s %(message)s",
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )

    interrupted = False
    def _sig(sig, frame):
        nonlocal interrupted
        interrupted = True
        log.warning("Interrupted.")
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    log.info(
        "Starting citation linker (threshold=%.2f, top_k=%d, max_chunks=%s)",
        args.threshold, args.top_k, args.max_chunks,
    )
    t0 = time.time()
    report = run_citation_linker(
        driver,
        threshold=args.threshold,
        dense_top_k=args.top_k,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
    )

    out = Path("logs/citation_linker_report.json")
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    log.info(
        "Done in %.1fs: %d chunks, %d spans, %d CITES edges. Report: %s",
        time.time() - t0, report.chunks_processed, report.spans_extracted,
        report.edges_created, out,
    )
    driver.close()


if __name__ == "__main__":
    main()
