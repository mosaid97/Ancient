"""B3 orchestrator: Leiden community detection + LLM summaries.

Usage:
    uv run python scripts/run_communities.py              # full corpus
    uv run python scripts/run_communities.py --limit 10000 --min-size 3  # smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase

from apps.backend.pipeline.community_summarize import build_community_summaries


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="B3: Leiden community summaries")
    p.add_argument("--resolution", type=float, default=3.0, help="Leiden resolution (higher=more communities)")
    p.add_argument("--min-size", type=int, default=5, help="Min community size (chunks)")
    p.add_argument("--max-kw-freq", type=int, default=500, help="Exclude keywords appearing in more than N chunks")
    p.add_argument("--limit", type=int, default=500_000, help="Max edges to load (smoke: 10000)")
    p.add_argument("--max-summary-chunks", type=int, default=10)
    p.add_argument("--log-file", default="logs/community_summarize.log")
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
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )

    log.info("Starting B3 community pipeline (resolution=%.2f, min_size=%d, max_kw_freq=%d, limit=%d)",
             args.resolution, args.min_size, args.max_kw_freq, args.limit)
    t0 = time.time()

    report = build_community_summaries(
        driver,
        resolution=args.resolution,
        min_size=args.min_size,
        max_kw_freq=args.max_kw_freq,
        limit=args.limit,
        max_summary_chunks=args.max_summary_chunks,
    )

    out = Path("logs/community_report.json")
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    log.info("Done in %.1fs. Report: %s", time.time() - t0, out)
    driver.close()


if __name__ == "__main__":
    main()
