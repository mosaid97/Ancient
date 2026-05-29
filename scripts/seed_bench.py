"""Seed data/bench/ relevant_chunk_ids via hybrid search (plan §0.6 D3).

For each query in queries.jsonl that has labeling_status='needs_labeling',
runs hybrid_search and writes the top-k chunk IDs as candidate
relevant_chunk_ids (labeling_status → 'auto_labeled', needs human review).

For faithfulness.jsonl, also resolves faithful_chunk_id via BM25 match on
the faithful_span.

Usage:
    uv run python scripts/seed_bench.py [--top-k 5] [--bench data/bench]
    uv run python scripts/seed_bench.py --dry-run   # print without writing
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
log = logging.getLogger(__name__)


def _connect():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
    )


def _update_queries(driver, bench_dir: Path, top_k: int, dry_run: bool) -> None:
    from apps.backend.pipeline.search import hybrid_search
    from apps.backend.retrieval.bm25 import BM25Corpus

    qfile = bench_dir / "queries.jsonl"
    if not qfile.exists():
        log.warning("No queries.jsonl at %s", qfile)
        return

    log.info("Building BM25 corpus …")
    bm25 = BM25Corpus.build(driver)

    queries = [json.loads(l) for l in qfile.open() if l.strip()]
    updated = 0

    for q in queries:
        if q.get("labeling_status") != "needs_labeling":
            continue
        try:
            resp = hybrid_search(driver, q["query"], bm25, top_k=100, rerank_top_k=top_k)
            chunk_ids = [r.chunk_id for r in resp.all_results[:top_k]]
            q["relevant_chunk_ids"] = chunk_ids
            q["labeling_status"] = "auto_labeled"
            updated += 1
            log.info("  %s → %d candidates", q["query_id"], len(chunk_ids))
        except Exception as exc:
            log.error("  %s failed: %s", q["query_id"], exc)

    if not dry_run:
        with qfile.open("w") as f:
            for q in queries:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        log.info("Written %d updated queries → %s", updated, qfile)
    else:
        log.info("[dry-run] Would update %d queries", updated)


def _update_faithfulness(driver, bench_dir: Path, dry_run: bool) -> None:
    """Resolve faithful_chunk_id by BM25 span search."""
    ffile = bench_dir / "faithfulness.jsonl"
    if not ffile.exists():
        return

    from apps.backend.retrieval.bm25 import BM25Corpus
    bm25 = BM25Corpus.build(driver)

    items = [json.loads(l) for l in ffile.open() if l.strip()]
    updated = 0

    for item in items:
        if item.get("labeling_status") != "needs_labeling":
            continue
        span = item.get("faithful_span", "")
        if not span:
            continue
        try:
            hits = bm25.query(span, top_k=1)
            if hits:
                item["faithful_chunk_id"] = hits[0][0]
                item["relevant_chunk_ids"] = [h[0] for h in bm25.query(span, top_k=5)]
                item["labeling_status"] = "auto_labeled"
                updated += 1
        except Exception as exc:
            log.error("  %s span lookup failed: %s", item["query_id"], exc)

    if not dry_run:
        with ffile.open("w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log.info("Written %d updated faithfulness entries → %s", updated, ffile)


def _update_linking(driver, bench_dir: Path, dry_run: bool) -> None:
    """Resolve secondary/primary chunk IDs in linking_gold via Neo4j."""
    lfile = bench_dir / "linking_gold.jsonl"
    if not lfile.exists():
        return

    _SECONDARY_QUERY = """
    MATCH (c:CHUNK)-[:HAS|CONSIST_OF|INCLUDE*]-(d:DOCUMENT)
    WHERE c.tier = 'secondary' AND d.title CONTAINS $doc_title
    RETURN c.id AS chunk_id LIMIT 1
    """
    _CITES_QUERY = """
    MATCH (sec:CHUNK {id: $cid})-[r:CITES]->(pri:CHUNK)
    RETURN pri.id AS primary_id ORDER BY r.confidence DESC LIMIT 5
    """

    items = [json.loads(l) for l in lfile.open() if l.strip()]
    updated = 0

    for item in items:
        if item.get("labeling_status") != "needs_labeling":
            continue
        doc_title = item.get("secondary_doc", "")
        if not doc_title:
            continue
        try:
            with driver.session() as s:
                rows = s.run(_SECONDARY_QUERY, doc_title=doc_title).data()
            if not rows:
                continue
            sec_id = rows[0]["chunk_id"]
            item["secondary_chunk_id"] = sec_id

            with driver.session() as s:
                pri_rows = s.run(_CITES_QUERY, cid=sec_id).data()
            item["primary_chunk_ids"] = [r["primary_id"] for r in pri_rows]
            item["labeling_status"] = "auto_labeled"
            updated += 1
        except Exception as exc:
            log.error("  %s linking failed: %s", item["pair_id"], exc)

    if not dry_run:
        with lfile.open("w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log.info("Written %d updated linking entries → %s", updated, lfile)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/bench")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bench_dir = Path(args.bench)
    driver = _connect()

    _update_queries(driver, bench_dir, args.top_k, args.dry_run)
    _update_faithfulness(driver, bench_dir, args.dry_run)
    _update_linking(driver, bench_dir, args.dry_run)

    driver.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
