#!/usr/bin/env python3
"""Resumable Python DAG orchestrator — drives the full Ancient pipeline.

Replaces the brittle bash ``run_pipeline_orchestrator.sh``.  Each stage:
  - checks a census gate (reads the live Neo4j graph)
  - skips itself when already complete
  - writes a per-stage report to ``logs/<stage>_report.json``
  - logs failures but continues independent downstream stages

Usage::

    # Dry-run: print outstanding counts per stage, no work done
    uv run python scripts/run_pipeline.py --dry-run

    # Full unattended run (recommended with caffeinate on macOS)
    caffeinate -dimsu uv run python scripts/run_pipeline.py \\
        --workers 8 --log-file logs/pipeline.log

    # Resume after a partial run (skips already-complete stages automatically)
    uv run python scripts/run_pipeline.py --workers 8

    # Run only a single stage
    uv run python scripts/run_pipeline.py --only fusion

Stage order (dependency-ordered):
    fusion → layout → chunk → embed → translate → keywords →
    keyword_relate → citation_linker → communities → audit
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("run_pipeline")

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"

# ---------------------------------------------------------------------------
# Stage definition
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    name: str
    status: str = "pending"   # pending | skipped | ok | failed
    elapsed_seconds: float = 0.0
    skip_reason: str = ""
    error: str = ""
    extra: dict = field(default_factory=dict)


def _run_stage(
    cmd: list[str],
    *,
    name: str,
    dry_run: bool,
    log_file: Path,
) -> tuple[int, str]:
    """Run a pipeline stage as a subprocess, teeing output to log_file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log.info("[%s] running: %s", name, " ".join(cmd))
    if dry_run:
        log.info("[%s] DRY-RUN — skipping actual execution", name)
        return 0, ""

    with log_file.open("a") as lf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lf.write(proc.stdout or "")

    # Echo tail to orchestrator log
    tail = (proc.stdout or "").splitlines()[-5:]
    for line in tail:
        log.debug("[%s] %s", name, line)

    return proc.returncode, (proc.stdout or "")


def _collect_census(driver) -> dict:
    """Import and run the census collector."""
    # Import here to avoid loading neo4j at module import time
    from scripts.audit_corpus_state import collect_census  # noqa: PLC0415
    return collect_census(driver)


def _get_driver():
    from neo4j import GraphDatabase  # noqa: PLC0415
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "AncientChina"),
        ),
    )


# ---------------------------------------------------------------------------
# Gate predicates — return (outstanding_count, skip_if_zero)
# ---------------------------------------------------------------------------

def gate_fusion(census: dict) -> tuple[int, str]:
    n = census.get("pages", {}).get("fusion_pending", 0)
    return n, f"{n} OCR pages pending fusion"


def gate_layout(census: dict) -> tuple[int, str]:
    n = census.get("pages", {}).get("layout_pending", 0)
    return n, f"{n} OCR pages pending layout"


def gate_chunk(census: dict) -> tuple[int, str]:
    pages = census.get("pages", {})
    total_ocr = pages.get("ocr", 0)
    total_native = pages.get("native", 0)
    chunked = pages.get("chunked", 0)
    total = total_ocr + total_native
    outstanding = max(0, total - chunked)
    return outstanding, f"{outstanding} pages not yet chunked"


def gate_embed(census: dict) -> tuple[int, str]:
    chunks = census.get("chunks", {})
    n = chunks.get("embed_pending", 0) + chunks.get("embed_failed", 0)
    return n, f"{n} chunks pending/failed embedding"


def gate_translate(census: dict) -> tuple[int, str]:
    chunks = census.get("chunks", {})
    total = chunks.get("total", 0)
    ok = chunks.get("translated_ok", 0)
    skipped = chunks.get("total", 0) - chunks.get("primary_chunks", 0)  # secondary is fast
    outstanding = max(0, total - ok - skipped)
    # Rough: any chunk without translationStatus=ok or skipped
    outstanding = max(0, total - ok)
    return outstanding, f"{outstanding} chunks not yet translated"


def gate_keywords(census: dict) -> tuple[int, str]:
    chunks = census.get("chunks", {})
    total = chunks.get("total", 0)
    done = chunks.get("keywords_ok", 0)
    n = max(0, total - done)
    return n, f"{n} chunks missing keyword extraction"


def gate_keyword_relate(census: dict) -> tuple[int, str]:
    kw_total = census.get("keywords", {}).get("total", 0)
    related = census.get("relationship_counts", {}).get("RELATED", 0)
    # Heuristic: if RELATED << keywords we haven't run relate yet
    # Each keyword gets ~5 avg RELATED edges when complete
    expected_min = kw_total * 2
    n = max(0, expected_min - related) if related < expected_min else 0
    return n, f"RELATED edges {related} vs keywords {kw_total} (need re-run if low)"


def gate_citation_linker(census: dict) -> tuple[int, str]:
    cites = census.get("relationship_counts", {}).get("CITES", 0)
    secondary = census.get("chunks", {}).get("secondary_chunks", 0)
    # Expect at least 5% of secondary chunks to have CITES edges when complete
    threshold = max(1, secondary // 20)
    n = max(0, threshold - cites)
    return n, f"CITES edges {cites} vs secondary chunks {secondary}"


def gate_communities(census: dict) -> tuple[int, str]:
    n_communities = census.get("node_counts", {}).get("COMMUNITY", 0)
    # Expect at least 50 communities with the fixed resolution=3.0 setting
    n = max(0, 50 - n_communities)
    return n, f"{n_communities} COMMUNITY nodes (need ≥50 for synthesis routing)"


# ---------------------------------------------------------------------------
# Stage table
# ---------------------------------------------------------------------------

def _build_stage_table(workers: int, recompute_embed: bool) -> list[dict]:
    """Return ordered list of stage descriptors."""
    uv = ["uv", "run", "python"]
    return [
        {
            "name": "fusion",
            "gate": gate_fusion,
            "cmd": uv + ["scripts/run_fusion.py"],
            "log": LOGS_DIR / "fusion_run.log",
        },
        {
            "name": "layout",
            "gate": gate_layout,
            "cmd": uv + ["scripts/run_layout_analysis.py"],
            "log": LOGS_DIR / "layout_run.log",
        },
        {
            "name": "chunk",
            "gate": gate_chunk,
            "cmd": uv + ["scripts/run_chunking.py"],
            "log": LOGS_DIR / "chunking_run.log",
        },
        {
            "name": "embed",
            "gate": gate_embed,
            "cmd": (
                uv + ["scripts/run_embedding.py", "--recompute"]
                if recompute_embed
                else uv + ["scripts/run_embedding.py"]
            ),
            "log": LOGS_DIR / "embedding_run.log",
        },
        {
            "name": "translate",
            "gate": gate_translate,
            "cmd": uv + [
                "scripts/run_translation.py",
                "--workers", str(workers),
                "--tier", "primary",
            ],
            "log": LOGS_DIR / "translation_primary.log",
        },
        {
            "name": "translate_secondary",
            "gate": gate_translate,
            "cmd": uv + [
                "scripts/run_translation.py",
                "--workers", str(workers),
                "--tier", "secondary",
            ],
            "log": LOGS_DIR / "translation_secondary.log",
        },
        {
            "name": "embed_vernacular",
            "gate": gate_embed,
            "cmd": uv + ["scripts/run_embedding.py"],
            "log": LOGS_DIR / "embedding_vernacular.log",
        },
        {
            "name": "keywords",
            "gate": gate_keywords,
            "cmd": uv + ["scripts/run_keyword_extraction.py"],
            "log": LOGS_DIR / "keywords_run.log",
        },
        {
            "name": "keyword_relate",
            "gate": gate_keyword_relate,
            "cmd": uv + ["scripts/run_keyword_relate.py"],
            "log": LOGS_DIR / "keyword_relate_run.log",
        },
        {
            "name": "citation_linker",
            "gate": gate_citation_linker,
            "cmd": uv + ["scripts/run_citation_linker.py"],
            "log": LOGS_DIR / "citation_linker_run.log",
            "needs_hf": True,
        },
        {
            "name": "communities",
            "gate": gate_communities,
            "cmd": uv + ["scripts/run_communities.py"],
            "log": LOGS_DIR / "communities_run.log",
        },
        {
            "name": "audit",
            "gate": lambda _: (1, "always re-audit at end"),
            "cmd": uv + ["scripts/audit_corpus_state.py"],
            "log": LOGS_DIR / "audit_final.log",
        },
    ]


# ---------------------------------------------------------------------------
# HF model cache check
# ---------------------------------------------------------------------------

def _check_hf_model(model_name: str = "BAAI/bge-reranker-v2-gemma") -> bool:
    """Return True if the model is already in the HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
        result = try_to_load_from_cache(model_name, "config.json")
        return result is not None and result != "not_found"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _setup_logging(log_file: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Resumable Ancient pipeline DAG orchestrator")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print per-stage outstanding counts; do not execute any stage",
    )
    ap.add_argument(
        "--workers", type=int, default=8,
        help="Parallel translation workers passed to run_translation.py (default 8)",
    )
    ap.add_argument(
        "--only", metavar="STAGE",
        help="Run only this stage name, ignoring all others",
    )
    ap.add_argument(
        "--recompute-embed", action="store_true",
        help="Pass --recompute to run_embedding.py (re-embed all chunks)",
    )
    ap.add_argument(
        "--log-file", default="logs/pipeline_run.log",
        help="Orchestrator log file (default logs/pipeline_run.log)",
    )
    args = ap.parse_args()

    _setup_logging(args.log_file)
    log.info("=== Ancient pipeline orchestrator started (dry_run=%s workers=%d) ===",
             args.dry_run, args.workers)

    # Connect and take a census snapshot
    log.info("Taking corpus census...")
    try:
        driver = _get_driver()
        from scripts.audit_corpus_state import collect_census  # noqa: PLC0415
        census = collect_census(driver)
        driver.close()
    except Exception as exc:
        log.error("Failed to collect census: %s — aborting", exc)
        sys.exit(1)

    log.info(
        "Census: docs=%s pages=%s chunks=%s keywords=%s communities=%s",
        sum(census.get("documents_by_tier", {}).values()),
        census.get("pages", {}).get("total", "?"),
        census.get("chunks", {}).get("total", "?"),
        census.get("keywords", {}).get("total", "?"),
        census.get("node_counts", {}).get("COMMUNITY", 0),
    )

    stages = _build_stage_table(workers=args.workers, recompute_embed=args.recompute_embed)
    if args.only:
        stages = [s for s in stages if s["name"] == args.only]
        if not stages:
            log.error("Unknown stage '%s'", args.only)
            sys.exit(1)

    results: list[StageResult] = []
    t_global = time.time()

    for stage in stages:
        name = stage["name"]
        r = StageResult(name=name)

        # Gate check
        outstanding, reason = stage["gate"](census)
        if outstanding == 0 and name != "audit":
            r.status = "skipped"
            r.skip_reason = reason
            log.info("[%s] SKIP — %s", name, reason)
            results.append(r)
            continue

        log.info("[%s] outstanding=%s — %s", name, outstanding, reason)

        # HF cache pre-flight
        if stage.get("needs_hf") and not args.dry_run:
            if not _check_hf_model():
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    log.warning(
                        "[%s] bge-reranker-v2-gemma not in HF cache and HF_TOKEN not set. "
                        "Stage may hang downloading. Set HF_TOKEN in .env to enable auth.",
                        name,
                    )

        t0 = time.time()
        rc, output = _run_stage(
            stage["cmd"],
            name=name,
            dry_run=args.dry_run,
            log_file=stage["log"],
        )
        r.elapsed_seconds = time.time() - t0

        if args.dry_run:
            r.status = "skipped"
            r.skip_reason = "dry-run"
        elif rc == 0:
            r.status = "ok"
            log.info("[%s] OK in %.0fs", name, r.elapsed_seconds)
        else:
            r.status = "failed"
            r.error = f"exit code {rc}"
            log.error("[%s] FAILED (exit %d) in %.0fs — continuing pipeline", name, rc, r.elapsed_seconds)

        # Write per-stage mini-report
        report_path = LOGS_DIR / f"{name}_stage_report.json"
        report_path.write_text(json.dumps({
            "name": name,
            "status": r.status,
            "elapsed_seconds": round(r.elapsed_seconds, 1),
            "skip_reason": r.skip_reason,
            "error": r.error,
        }, indent=2))

        results.append(r)

    # Final summary
    elapsed = time.time() - t_global
    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = [r for r in results if r.status == "failed"]

    log.info(
        "=== Pipeline done in %.0fs — stages: %d ok / %d skipped / %d failed ===",
        elapsed, ok, skipped, len(failed),
    )
    for r in failed:
        log.error("  FAILED: %s — %s", r.name, r.error)

    summary = {
        "elapsed_seconds": round(elapsed, 1),
        "stages_ok": ok,
        "stages_skipped": skipped,
        "stages_failed": len(failed),
        "failed_names": [r.name for r in failed],
        "details": [{"name": r.name, "status": r.status, "elapsed_seconds": r.elapsed_seconds,
                     "skip_reason": r.skip_reason, "error": r.error}
                    for r in results],
    }
    (LOGS_DIR / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Summary written to logs/pipeline_summary.json")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
