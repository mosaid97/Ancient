"""B1 orchestrator: embed KEYWORD nodes + build RELATED edges.

Phase 1 — embed_keywords(): writes KEYWORD.embedding for all nodes lacking one.
Phase 2 — link_related_keywords(): queries ANN index and MERGEs RELATED edges.

Usage:
    uv run python scripts/run_keyword_relate.py              # full run
    uv run python scripts/run_keyword_relate.py --embed-only # Phase 1 only
    uv run python scripts/run_keyword_relate.py --link-only  # Phase 2 only (needs Phase 1)
    uv run python scripts/run_keyword_relate.py --max-keywords 1000 --embed-only  # smoke test
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

from apps.backend.pipeline.keyword_relate import (
    RelateReport,
    EmbedReport,
    embed_keywords,
    link_related_keywords,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="B1: Keyword embedding + RELATED edge builder")
    p.add_argument("--embed-only", action="store_true", help="Only run Phase 1 (embedding)")
    p.add_argument("--link-only", action="store_true", help="Only run Phase 2 (RELATED edges)")
    p.add_argument("--max-keywords", type=int, default=None, help="Cap for smoke tests")
    p.add_argument("--threshold", type=float, default=0.85, help="Cosine threshold for RELATED edges")
    p.add_argument("--top-k", type=int, default=50, help="ANN candidates per keyword")
    p.add_argument("--batch-size", type=int, default=10, help="Embed API batch size")
    p.add_argument("--log-file", default="logs/keyword_relate.log")
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

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "")
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

    interrupted = False
    def _sighandler(sig, frame):
        nonlocal interrupted
        interrupted = True
        log.warning("Interrupted — will save partial report.")
    signal.signal(signal.SIGINT, _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    report: dict = {}
    t_start = time.time()

    run_embed = not args.link_only
    run_link = not args.embed_only

    if run_embed:
        log.info("=== Phase 1: Embedding keywords ===")
        embed_rep = embed_keywords(
            driver,
            batch_size=args.batch_size,
            max_keywords=args.max_keywords,
        )
        report["embed"] = embed_rep.to_dict()
        log.info("Phase 1 done: %d embedded, %d failed", embed_rep.keywords_embedded, embed_rep.keywords_failed)

    if run_link and not interrupted:
        log.info("=== Phase 2: Building RELATED edges (threshold=%.2f) ===", args.threshold)
        relate_rep = link_related_keywords(
            driver,
            threshold=args.threshold,
            top_k=args.top_k,
            max_keywords=args.max_keywords,
        )
        report["relate"] = relate_rep.to_dict()
        log.info(
            "Phase 2 done: %d keywords processed, %d RELATED edges created",
            relate_rep.keywords_processed, relate_rep.edges_created,
        )

    report["duration_seconds"] = round(time.time() - t_start, 2)
    out = Path("logs/keyword_relate_report.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("Report saved to %s", out)
    driver.close()


if __name__ == "__main__":
    main()
