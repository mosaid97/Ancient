"""
Mark persistently-failing chunks as 'skipped' so run_translation can advance.

Usage:
    uv run python scripts/skip_stuck_chunks.py [--dry-run]
"""
import argparse
import os
from datetime import datetime, timezone

from neo4j import GraphDatabase

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "AncientChina")

# Chunks confirmed stuck across ≥30 consecutive all-fail batches
STUCK_PAGES = [
    "旧唐书__4f94404c88::p00047",
    "旧唐书__4f94404c88::p00048",
    "旧唐书__4f94404c88::p00049",
]

INSPECT_Q = """
MATCH (ch:CHUNK)-[:HAS]-(p:PAGE {id: $page_id})
WHERE ch.translationStatus = 'failed'
RETURN ch.id AS chunk_id,
       size(coalesce(ch.text, '')) AS text_len,
       ch.translationError AS error,
       ch.text AS text
ORDER BY ch.id
"""

SKIP_Q = """
MATCH (ch:CHUNK)-[:HAS]-(p:PAGE {id: $page_id})
WHERE ch.translationStatus = 'failed'
SET ch.translationStatus = 'skipped',
    ch.translationAt     = $ts,
    ch.translationError  = 'skipped_stuck: persistent connection_error across >30 batches'
RETURN count(ch) AS skipped
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    ts = datetime.now(timezone.utc).isoformat()

    with driver.session() as s:
        for page_id in STUCK_PAGES:
            rows = s.run(INSPECT_Q, page_id=page_id).data()
            print(f"\n=== {page_id} — {len(rows)} failed chunks ===")
            for r in rows[:3]:
                print(f"  {r['chunk_id']}  len={r['text_len']}  err={r['error']}")
                if r['text']:
                    print(f"  text[:120]: {r['text'][:120]!r}")
            if len(rows) > 3:
                print(f"  ... and {len(rows)-3} more")

            if not args.dry_run and rows:
                result = s.run(SKIP_Q, page_id=page_id, ts=ts).single()
                print(f"  → marked {result['skipped']} chunks as skipped")
            elif args.dry_run:
                print(f"  [dry-run] would skip {len(rows)} chunks")

    driver.close()
    print("\nDone. Re-running translate will now skip these pages and process the rest.")


if __name__ == "__main__":
    main()
