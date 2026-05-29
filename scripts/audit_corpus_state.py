#!/usr/bin/env python3
"""Corpus state census — prints a full Neo4j snapshot to stdout and saves
logs/corpus_state.json.

Usage:
    uv run python scripts/audit_corpus_state.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)


def _run(session, query: str, **params) -> list[dict]:
    return session.run(query, **params).data()


def collect_census(driver) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    census: dict = {"timestamp": ts}

    with driver.session() as s:

        # ── Node counts ──────────────────────────────────────────────────────
        rows = _run(s, "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC")
        census["node_counts"] = {r["label"]: r["cnt"] for r in rows}

        # ── DOCUMENT by tier ─────────────────────────────────────────────────
        rows = _run(s, """
            MATCH (d:DOCUMENT)
            RETURN d.tier AS tier, count(d) AS cnt
            ORDER BY tier
        """)
        census["documents_by_tier"] = {r["tier"] or "unknown": r["cnt"] for r in rows}

        # ── PAGE summary ─────────────────────────────────────────────────────
        row = _run(s, """
            MATCH (p:PAGE)
            RETURN
              count(p)                                                         AS total,
              count(CASE WHEN p.mode = 'ocr'         THEN 1 END)              AS ocr,
              count(CASE WHEN p.mode = 'native_text' THEN 1 END)              AS native,
              count(CASE WHEN p.fusionStatus = 'ok'    THEN 1 END)            AS fused_ok,
              count(CASE WHEN p.fusionStatus = 'single' THEN 1 END)           AS fused_single,
              count(CASE WHEN p.fusionStatus IS NULL
                          AND p.mode = 'ocr'         THEN 1 END)              AS fusion_pending,
              count(CASE WHEN p.textFused IS NOT NULL THEN 1 END)             AS has_text_fused,
              count(CASE WHEN p.layoutStatus = 'ok'   THEN 1 END)             AS layout_ok,
              count(CASE WHEN p.layoutStatus = 'manuscript' THEN 1 END)       AS layout_manuscript,
              count(CASE WHEN p.layoutStatus IS NULL
                          AND p.mode = 'ocr'         THEN 1 END)              AS layout_pending,
              count(CASE WHEN p.chunkingAt IS NOT NULL THEN 1 END)            AS chunked,
              count(CASE WHEN p.paddleOcrStatus = 'ok'  THEN 1 END)           AS paddle_ok,
              count(CASE WHEN p.deepseekOcrStatus = 'ok' THEN 1 END)          AS deepseek_ok,
              count(CASE WHEN p.qwenVlOcrStatus = 'ok'  THEN 1 END)           AS qwen_ok
        """)[0]
        census["pages"] = dict(row)

        # ── CHUNK summary ────────────────────────────────────────────────────
        row = _run(s, """
            MATCH (c:CHUNK)
            RETURN
              count(c)                                                         AS total,
              count(CASE WHEN c.embeddingStatus = 'ok'      THEN 1 END)       AS embedded_ok,
              count(CASE WHEN c.embeddingStatus = 'pending' THEN 1 END)       AS embed_pending,
              count(CASE WHEN c.embeddingStatus = 'failed'  THEN 1 END)       AS embed_failed,
              count(CASE WHEN c.textCanonical IS NOT NULL   THEN 1 END)       AS has_canonical,
              count(CASE WHEN c.textVernacular IS NOT NULL  THEN 1 END)       AS has_vernacular,
              count(CASE WHEN c.translationStatus = 'ok'   THEN 1 END)       AS translated_ok,
              count(CASE WHEN c.translationStatus = 'failed' THEN 1 END)      AS translated_failed,
              count(CASE WHEN c.mentionStatus = 'ok'        THEN 1 END)       AS keywords_ok,
              count(CASE WHEN c.tier = 'primary'            THEN 1 END)       AS primary_chunks,
              count(CASE WHEN c.tier = 'secondary'          THEN 1 END)       AS secondary_chunks
        """)[0]
        census["chunks"] = dict(row)

        # ── KEYWORD ──────────────────────────────────────────────────────────
        row = _run(s, """
            MATCH (k:KEYWORD)
            RETURN count(k) AS total,
                   sum(k.frequency) AS total_mentions
        """)[0]
        census["keywords"] = dict(row)

        # ── Relationship counts ───────────────────────────────────────────────
        rows = _run(s, """
            MATCH ()-[r]->()
            RETURN type(r) AS rel, count(r) AS cnt
            ORDER BY cnt DESC
        """)
        census["relationship_counts"] = {r["rel"]: r["cnt"] for r in rows}

        # ── Coverage pcts ────────────────────────────────────────────────────
        pages = census["pages"]
        chunks = census["chunks"]
        ocr = pages["ocr"] or 1
        total_chunks = chunks["total"] or 1
        census["coverage_pct"] = {
            "ocr_fused": round(100 * (pages["fused_ok"] + pages["fused_single"]) / ocr, 1),
            "ocr_layout": round(100 * pages["layout_ok"] / ocr, 1),
            "chunks_embedded": round(100 * chunks["embedded_ok"] / total_chunks, 1),
            "chunks_translated": round(100 * chunks["translated_ok"] / total_chunks, 1),
            "chunks_has_canonical": round(100 * chunks["has_canonical"] / total_chunks, 1),
            "chunks_keywords": round(100 * chunks["keywords_ok"] / total_chunks, 1),
        }

    return census


def main() -> None:
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "AncientChina")),
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )

    log.info("Collecting corpus census…")
    census = collect_census(driver)
    driver.close()

    # pretty-print to stdout
    print(json.dumps(census, indent=2, ensure_ascii=False))

    # persist
    out = Path("logs/corpus_state.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(census, indent=2, ensure_ascii=False))
    log.info("Saved → %s", out)


if __name__ == "__main__":
    main()
