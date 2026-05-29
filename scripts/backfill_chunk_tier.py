"""Backfill CHUNK.tier from the parent PAGE.tier.

chunk.py previously omitted the tier property when creating CHUNK nodes,
leaving ~40K chunks with tier=NULL. This script propagates the tier from
each chunk's parent PAGE via the (:PAGE)-[:HAS]->(:CHUNK) edge.

Run once after the fix to chunk.py; subsequent chunk_pages() runs will
write tier correctly for new chunks.

Usage:
    uv run python scripts/backfill_chunk_tier.py
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Propagate tier from parent PAGE to child CHUNK where tier is missing.
# Batched via SKIP/LIMIT to avoid a single giant transaction.
_COUNT_QUERY = """
MATCH (p:PAGE)-[:HAS]->(c:CHUNK)
WHERE c.tier IS NULL AND p.tier IS NOT NULL
RETURN count(c) AS n
"""

_BACKFILL_BATCH = """
MATCH (p:PAGE)-[:HAS]->(c:CHUNK)
WHERE c.tier IS NULL AND p.tier IS NOT NULL
WITH c, p LIMIT $batch
SET c.tier = p.tier
RETURN count(c) AS updated
"""


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    t_start = time.time()

    with driver.session() as s:
        total_missing = s.run(_COUNT_QUERY).single()["n"]

    log.info("Chunks missing tier: %d", total_missing)
    if total_missing == 0:
        log.info("Nothing to backfill — all chunks already have tier set.")
        driver.close()
        return

    updated = 0
    batch = 5_000
    while True:
        with driver.session() as s:
            n = s.run(_BACKFILL_BATCH, batch=batch).single()["updated"]
        updated += n
        log.info("Backfilled %d / %d chunks", updated, total_missing)
        if n == 0:
            break

    with driver.session() as s:
        remaining = s.run(_COUNT_QUERY).single()["n"]

    elapsed = time.time() - t_start
    report = {
        "chunks_backfilled": updated,
        "chunks_remaining_null": remaining,
        "duration_seconds": round(elapsed, 2),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    out = Path("logs/backfill_chunk_tier_report.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("Done in %.1fs. Report: %s", elapsed, out)
    driver.close()


if __name__ == "__main__":
    main()
