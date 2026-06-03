"""Resilient pipeline orchestrator — runs remaining steps to v3 milestone.

Steps (in order):
  1. Translation A1   — primary tier until ≥95% coverage (auto-restarts on crash)
  2. Embed recompute  — re-embed all chunks over textCanonical
  3. Citation linker  — B2 Secondary→Primary CITES edges
  4. Communities      — B3 Leiden re-run at resolution=3.0
  5. Corpus audit     — final snapshot

Each step retries up to MAX_RETRIES times with exponential backoff.
Run with:
    caffeinate -dimsu uv run python scripts/run_pipeline_orchestrator.py 2>&1 | tee logs/orchestrator2.log
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("orchestrator")

MAX_RETRIES = 20
BASE_BACKOFF = 30   # seconds; doubles each retry up to MAX_BACKOFF
MAX_BACKOFF = 300


def _neo4j_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "AncientChina")),
        max_connection_pool_size=5,
        connection_acquisition_timeout=60,
    )


def _translation_coverage() -> tuple[int, int]:
    """Return (translated_primary, total_primary), retrying on auth rate limit."""
    backoff = 30
    for attempt in range(10):
        try:
            driver = _neo4j_driver()
            try:
                with driver.session() as s:
                    total = s.run(
                        'MATCH (c:CHUNK {tier:"primary"}) RETURN count(c) AS n'
                    ).single()["n"]
                    done = s.run(
                        'MATCH (c:CHUNK {tier:"primary"}) WHERE c.textCanonical IS NOT NULL RETURN count(c) AS n'
                    ).single()["n"]
                return done, total
            finally:
                driver.close()
        except Exception as exc:
            log.warning("_translation_coverage attempt %d failed (%s) — retry in %ds", attempt + 1, exc, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
    raise RuntimeError("_translation_coverage: all retries exhausted")


def _run(cmd: list[str], label: str) -> bool:
    """Run a subprocess, return True on success."""
    log.info("[%s] running: %s", label, " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    if result.returncode == 0:
        log.info("[%s] finished OK.", label)
        return True
    log.warning("[%s] exited with code %d.", label, result.returncode)
    return False


def run_with_retry(cmd: list[str], label: str, check_done=None) -> None:
    """Run cmd, retrying up to MAX_RETRIES on failure.

    check_done: optional callable() -> bool; if it returns True the step is
    considered complete even if the process exited non-zero (e.g. after a
    SIGINT mid-run that still made progress).
    """
    backoff = BASE_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        ok = _run(cmd, label)
        if ok:
            return
        if check_done and check_done():
            log.info("[%s] target condition met despite non-zero exit — continuing.", label)
            return
        if attempt < MAX_RETRIES:
            log.warning("[%s] attempt %d/%d failed — retrying in %ds…",
                        label, attempt, MAX_RETRIES, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
        else:
            log.error("[%s] all %d attempts failed — aborting orchestrator.", label, MAX_RETRIES)
            sys.exit(1)


# ── Step 1: Translation A1 — loop until ≥95% primary coverage ────────────────

def step_translation():
    TARGET_PCT = 0.95
    done, total = _translation_coverage()
    log.info("STEP 1 — Translation A1: %d/%d primary chunks translated (%.1f%%)",
             done, total, 100 * done / max(total, 1))

    if done / max(total, 1) >= TARGET_PCT:
        log.info("STEP 1 — Already at ≥95%%. Skipping.")
        return

    # Run in serial (workers=1) to avoid Neo4j auth rate limit
    cmd = [
        "uv", "run", "python", "scripts/run_translation.py",
        "--tier", "primary",
        "--workers", "1",
        "--batch-size", "100",
        "--log-file", "logs/translation_auto.log",
    ]

    loop = 0
    while True:
        loop += 1
        done, total = _translation_coverage()
        pct = done / max(total, 1)
        log.info("STEP 1 loop %d — %d/%d (%.1f%%)", loop, done, total, 100 * pct)
        if pct >= TARGET_PCT:
            log.info("STEP 1 — Target reached.")
            break

        backoff = BASE_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            ok = _run(cmd, f"translation-loop{loop}-attempt{attempt}")
            if ok:
                break
            # Check if we made progress despite non-zero exit
            new_done, _ = _translation_coverage()
            if new_done > done:
                log.info("STEP 1 — made progress (%d → %d), continuing.", done, new_done)
                break
            log.warning("STEP 1 — attempt %d/%d failed, retry in %ds", attempt, MAX_RETRIES, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
        else:
            log.error("STEP 1 — translation stuck after %d attempts. Continuing with partial coverage.", MAX_RETRIES)
            break

    done, total = _translation_coverage()
    log.info("STEP 1 — Final coverage: %d/%d (%.1f%%)", done, total, 100 * done / max(total, 1))


# ── Step 2: Re-embed over textCanonical ──────────────────────────────────────

def step_embed_recompute():
    log.info("STEP 2 — Embed recompute (textCanonical-based vectors)")
    run_with_retry(
        ["uv", "run", "python", "scripts/run_embedding.py",
         "--recompute",
         "--log-file", "logs/embedding_recompute.log"],
        "embed-recompute",
    )


# ── Step 3: Citation linker B2 ────────────────────────────────────────────────

def step_citation_linker():
    log.info("STEP 3 — Citation linker B2 (Secondary→Primary CITES)")
    run_with_retry(
        ["uv", "run", "python", "scripts/run_citation_linker.py",
         "--log-file", "logs/citation_linker_auto.log"],
        "citation-linker",
    )


# ── Step 4: Community detection B3 ───────────────────────────────────────────

def step_communities():
    log.info("STEP 4 — Community detection B3 (resolution=3.0)")
    run_with_retry(
        ["uv", "run", "python", "scripts/run_communities.py",
         "--resolution", "3.0",
         "--log-file", "logs/community_auto.log"],
        "communities",
    )


# ── Step 5: Corpus audit ──────────────────────────────────────────────────────

def step_audit():
    log.info("STEP 5 — Final corpus audit")
    run_with_retry(
        ["uv", "run", "python", "scripts/audit_corpus_state.py"],
        "audit",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Pipeline orchestrator started")
    log.info("=" * 60)

    step_translation()
    step_embed_recompute()
    step_citation_linker()
    step_communities()
    step_audit()

    log.info("=" * 60)
    log.info("ALL STEPS COMPLETE — v3 milestone pipeline done.")
    log.info("=" * 60)
