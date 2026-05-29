"""Seed all language dictionaries into Neo4j DICTIONARY_ENTRY nodes.

Seeds zh (Kangxi/唐律疏議), ja (JMdict), en (Hucker/Mathews), ar (Arabic Wikipedia).

Usage:
    uv run python scripts/seed_dictionaries.py
    uv run python scripts/seed_dictionaries.py --lang ja     # one language only
    uv run python scripts/seed_dictionaries.py --dry-run     # count only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import os

from neo4j import GraphDatabase

from apps.backend.kb.dictionary import _SEED_PATHS_BY_LANG, seed_all_dictionaries, seed_dictionary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("seed_dictionaries")


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed multilingual dictionaries into Neo4j")
    ap.add_argument("--lang", choices=list(_SEED_PATHS_BY_LANG), default=None,
                    help="Seed only this language (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Count entries without writing to Neo4j")
    args = ap.parse_args()

    if args.dry_run:
        for lang, path in _SEED_PATHS_BY_LANG.items():
            if args.lang and lang != args.lang:
                continue
            if not path.exists():
                log.warning("Missing seed file: %s", path)
                continue
            with open(path, encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
            log.info("[dry-run] %s → %d entries in %s", lang, count, path.name)
        return

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "AncientChina")),
    )

    try:
        if args.lang:
            path = _SEED_PATHS_BY_LANG[args.lang]
            n = seed_dictionary(driver, path)
            log.info("Seeded %s: %d entries", args.lang, n)
        else:
            totals = seed_all_dictionaries(driver)
            for lang, n in totals.items():
                log.info("Seeded %s: %d entries", lang, n)
            log.info("Total: %d entries across %d languages", sum(totals.values()), len(totals))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
