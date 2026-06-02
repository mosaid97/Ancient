"""Admin stats & pipeline monitoring API router."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from neo4j import Driver

from apps.backend.api.deps import get_driver

log = logging.getLogger(__name__)
router = APIRouter()

_LOGS_DIR = Path(__file__).resolve().parents[4] / "logs"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _run(driver: Driver, query: str, **params) -> list[dict]:
    with driver.session() as s:
        return [dict(r) for r in s.run(query, **params)]


@router.get("/stats")
async def stats(driver: Driver = Depends(get_driver)) -> dict[str, Any]:
    try:
        # Documents
        doc_rows = _run(driver, "MATCH (d:DOCUMENT) RETURN d.tier AS tier, count(*) AS n")
        doc_counts: dict[str, int] = {}
        for r in doc_rows:
            doc_counts[r["tier"] or "unknown"] = r["n"]

        # Pages
        page_rows = _run(driver, """
            MATCH (p:PAGE)
            RETURN p.mode AS mode, p.fusionStatus AS fstat, count(*) AS n
        """)
        page_total = sum(r["n"] for r in page_rows)
        page_by_mode: dict[str, int] = {}
        page_by_fusion: dict[str, int] = {}
        for r in page_rows:
            page_by_mode[r["mode"] or "unknown"] = page_by_mode.get(r["mode"] or "unknown", 0) + r["n"]
            page_by_fusion[r["fstat"] or "pending"] = page_by_fusion.get(r["fstat"] or "pending", 0) + r["n"]

        # OCR evaluation counts
        eval_rows = _run(driver, """
            MATCH (p:PAGE) WHERE p.mode = 'ocr'
            RETURN p.evaluationDecision AS dec, count(*) AS n
        """)
        ocr_eval: dict[str, int] = {}
        for r in eval_rows:
            ocr_eval[r["dec"] or "unevaluated"] = r["n"]

        # Chunks
        chunk_rows = _run(driver, """
            MATCH (c:CHUNK)
            RETURN c.tier AS tier, c.translationStatus AS ts, count(*) AS n
        """)
        chunk_total = sum(r["n"] for r in chunk_rows)
        chunk_primary = sum(r["n"] for r in chunk_rows if r["tier"] == "primary")
        chunk_secondary = sum(r["n"] for r in chunk_rows if r["tier"] == "secondary")
        chunk_translated = sum(r["n"] for r in chunk_rows if r["ts"] == "ok")
        chunk_failed = sum(r["n"] for r in chunk_rows if r["ts"] == "failed")
        chunk_untranslated = chunk_total - chunk_translated - chunk_failed

        # Embedding coverage
        embed_row = _run(driver, """
            MATCH (c:CHUNK)
            RETURN
              sum(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded,
              count(*) AS total
        """)
        embedded = embed_row[0]["embedded"] if embed_row else 0
        embed_total = embed_row[0]["total"] if embed_row else 0

        # Keywords
        kw_row = _run(driver, "MATCH (k:KEYWORD) RETURN count(*) AS n")
        kw_count = kw_row[0]["n"] if kw_row else 0

        # Communities
        comm_row = _run(driver, "MATCH (c:COMMUNITY) RETURN count(*) AS n")
        comm_count = comm_row[0]["n"] if comm_row else 0

        trans_pct = round(chunk_translated / chunk_total * 100, 1) if chunk_total else 0
        embed_pct = round(embedded / embed_total * 100, 1) if embed_total else 0

        return {
            "documents": {
                "total": sum(doc_counts.values()),
                "primary": doc_counts.get("primary", 0),
                "secondary": doc_counts.get("secondary", 0),
            },
            "pages": {
                "total": page_total,
                "ocr": page_by_mode.get("ocr", 0),
                "native": page_by_mode.get("native", 0),
                "by_fusion_status": page_by_fusion,
                "ocr_evaluation": ocr_eval,
            },
            "chunks": {
                "total": chunk_total,
                "primary": chunk_primary,
                "secondary": chunk_secondary,
                "translated": chunk_translated,
                "failed": chunk_failed,
                "untranslated": chunk_untranslated,
                "embedded": embedded,
            },
            "keywords": {"total": kw_count},
            "communities": {"total": comm_count},
            "translation_progress": {
                "ok": chunk_translated,
                "failed": chunk_failed,
                "total": chunk_total,
                "pct": trans_pct,
            },
            "embedding_progress": {
                "ok": embedded,
                "total": embed_total,
                "pct": embed_pct,
            },
        }
    except Exception as exc:
        log.exception("Stats query failed: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Pipeline jobs
# ---------------------------------------------------------------------------


def _translation_job() -> dict[str, Any]:
    report_path = _LOGS_DIR / "translation_report.json"
    log_path = _LOGS_DIR / "translation_run.log"

    ok = total = 0
    elapsed = 0.0
    status = "pending"

    if report_path.exists():
        try:
            r = json.loads(report_path.read_text())
            ok = r.get("total_ok", 0)
            elapsed = r.get("elapsed_seconds", 0.0)
        except Exception:
            pass

    # Try to get total from Neo4j would be slow here; use known corpus size
    # (recorded in AGENTS.md: 25,999 primary + 14,240 secondary = 40,239)
    total = 40239

    # Check if process is running
    try:
        import subprocess
        out = subprocess.check_output(
            ["pgrep", "-f", "run_translation.py"], text=True
        ).strip()
        status = "running" if out else ("complete" if ok >= total else "idle")
    except Exception:
        status = "idle" if ok < total else "complete"

    pct = round(ok / total * 100, 2) if total else 0
    eta_s = None
    if ok > 0 and elapsed > 0 and ok < total:
        rate = ok / elapsed  # chunks per second
        eta_s = int((total - ok) / rate) if rate > 0 else None

    return {
        "name": "翻譯 Translation",
        "key": "translation",
        "status": status,
        "ok": ok,
        "total": total,
        "pct": pct,
        "elapsed_s": elapsed,
        "eta_s": eta_s,
        "log_file": str(log_path),
    }


def _embedding_job() -> dict[str, Any]:
    report_path = _LOGS_DIR / "embedding_report.json"
    ok = total = 0
    elapsed = 0.0
    if report_path.exists():
        try:
            r = json.loads(report_path.read_text())
            ok = r.get("total_ok", 0)
            total = r.get("total_chunks", 0) or 40239
            elapsed = r.get("elapsed_seconds", 0.0)
        except Exception:
            pass
    total = total or 40239
    status = "complete" if ok >= total and total > 0 else ("running" if ok > 0 else "pending")
    return {
        "name": "嵌入向量 Embedding",
        "key": "embedding",
        "status": status,
        "ok": ok,
        "total": total,
        "pct": round(ok / total * 100, 2) if total else 0,
        "elapsed_s": elapsed,
        "eta_s": None,
    }


def _keyword_job() -> dict[str, Any]:
    report_path = _LOGS_DIR / "keyword_report.json"
    ok = total = 0
    elapsed = 0.0
    if report_path.exists():
        try:
            r = json.loads(report_path.read_text())
            ok = r.get("total_ok", 0)
            total = r.get("total_chunks", ok) or 40239
            elapsed = r.get("elapsed_seconds", 0.0)
        except Exception:
            pass
    status = "complete" if ok > 0 and ok >= total else ("running" if ok > 0 else "pending")
    return {
        "name": "關鍵詞提取 Keywords",
        "key": "keywords",
        "status": status,
        "ok": ok,
        "total": total or 40239,
        "pct": round(ok / total * 100, 2) if total else 0,
        "elapsed_s": elapsed,
        "eta_s": None,
    }


def _community_job() -> dict[str, Any]:
    report_path = _LOGS_DIR / "community_report.json"
    ok = total = 0
    if report_path.exists():
        try:
            r = json.loads(report_path.read_text())
            ok = r.get("total_ok", 0)
        except Exception:
            pass
    status = "complete" if ok > 0 else "pending"
    return {
        "name": "社區檢測 Communities",
        "key": "communities",
        "status": status,
        "ok": ok,
        "total": ok or 0,
        "pct": 100.0 if ok > 0 else 0.0,
        "elapsed_s": 0.0,
        "eta_s": None,
    }


def _citation_job() -> dict[str, Any]:
    report_path = _LOGS_DIR / "citation_report.json"
    ok = total = 0
    if report_path.exists():
        try:
            r = json.loads(report_path.read_text())
            ok = r.get("total_ok", 0)
            total = r.get("total_spans", 0)
        except Exception:
            pass
    status = "complete" if ok > 0 else "pending"
    return {
        "name": "引文連結 Citation Linking",
        "key": "citations",
        "status": status,
        "ok": ok,
        "total": total,
        "pct": round(ok / total * 100, 2) if total else 0.0,
        "elapsed_s": 0.0,
        "eta_s": None,
    }


@router.get("/pipeline")
async def pipeline_status(driver: Driver = Depends(get_driver)) -> dict[str, Any]:
    # Pull live counts from Neo4j — report files are only used for elapsed time
    try:
        chunk_rows = _run(driver, """
            MATCH (c:CHUNK)
            RETURN
                sum(CASE WHEN c.translationStatus = 'ok'      THEN 1 ELSE 0 END) AS trans_ok,
                sum(CASE WHEN c.translationStatus = 'failed'   THEN 1 ELSE 0 END) AS trans_fail,
                sum(CASE WHEN c.embedding IS NOT NULL          THEN 1 ELSE 0 END) AS embedded,
                sum(CASE WHEN c.mentionStatus = 'ok'           THEN 1 ELSE 0 END) AS kw_ok,
                count(*) AS total
        """)
        live = chunk_rows[0] if chunk_rows else {}
    except Exception:
        live = {}

    total = live.get("total") or 40239
    trans_ok = live.get("trans_ok") or 0
    embedded = live.get("embedded") or 0
    kw_ok = live.get("kw_ok") or 0

    # Check if translation runner process is live
    try:
        import subprocess
        out = subprocess.check_output(["pgrep", "-f", "run_translation.py"], text=True).strip()
        trans_status = "running" if out else ("complete" if trans_ok >= total else "idle")
    except Exception:
        trans_status = "complete" if trans_ok >= total else "idle"

    # Read elapsed from last report
    trans_elapsed = 0.0
    trans_report = _LOGS_DIR / "translation_report.json"
    if trans_report.exists():
        try:
            trans_elapsed = json.loads(trans_report.read_text()).get("elapsed_seconds", 0.0)
        except Exception:
            pass

    trans_eta = None
    if trans_ok > 5 and trans_elapsed > 0 and trans_ok < total:
        rate = trans_ok / trans_elapsed
        trans_eta = int((total - trans_ok) / rate) if rate > 0 else None

    # Community and citation counts from Neo4j
    try:
        comm_row = _run(driver, "MATCH (c:COMMUNITY) RETURN count(*) AS n")
        comm_ok = comm_row[0]["n"] if comm_row else 0
        cite_row = _run(driver, "MATCH ()-[r:CITES]->() RETURN count(r) AS n")
        cite_ok = cite_row[0]["n"] if cite_row else 0
    except Exception:
        comm_ok = cite_ok = 0

    embed_status = "complete" if embedded >= total else ("running" if embedded > 0 else "pending")
    kw_status = "complete" if kw_ok >= total else ("running" if kw_ok > 0 else "pending")

    jobs = [
        {
            "name": "翻譯 Translation", "key": "translation",
            "status": trans_status,
            "ok": trans_ok, "total": total,
            "pct": round(trans_ok / total * 100, 2) if total else 0,
            "elapsed_s": trans_elapsed, "eta_s": trans_eta,
        },
        {
            "name": "嵌入向量 Embedding", "key": "embedding",
            "status": embed_status,
            "ok": embedded, "total": total,
            "pct": round(embedded / total * 100, 2) if total else 0,
            "elapsed_s": 0.0, "eta_s": None,
        },
        {
            "name": "關鍵詞提取 Keywords", "key": "keywords",
            "status": kw_status,
            "ok": kw_ok, "total": total,
            "pct": round(kw_ok / total * 100, 2) if total else 0,
            "elapsed_s": 0.0, "eta_s": None,
        },
        {
            "name": "社區檢測 Communities", "key": "communities",
            "status": "complete" if comm_ok > 0 else "pending",
            "ok": comm_ok, "total": comm_ok or 0,
            "pct": 100.0 if comm_ok > 0 else 0.0,
            "elapsed_s": 0.0, "eta_s": None,
        },
        {
            "name": "引文連結 Citation Linking", "key": "citations",
            "status": "complete" if cite_ok > 0 else "pending",
            "ok": cite_ok, "total": cite_ok or 0,
            "pct": 100.0 if cite_ok > 0 else 0.0,
            "elapsed_s": 0.0, "eta_s": None,
        },
    ]
    return {"jobs": jobs}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@router.get("/health")
async def health(driver: Driver = Depends(get_driver)) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # Neo4j
    try:
        with driver.session() as s:
            s.run("RETURN 1").single()
        results["neo4j"] = "ok"
    except Exception as exc:
        results["neo4j"] = f"error: {exc}"

    # MinIO
    try:
        from apps.backend.storage import get_minio_client
        mc = get_minio_client()
        mc.list_buckets()
        results["minio"] = "ok"
    except Exception as exc:
        results["minio"] = f"error: {exc}"

    # Silra (just check env var)
    silra_key = os.getenv("LLM_API_KEY") or os.getenv("SILRA_API_KEY") or os.getenv("OPENAI_API_KEY")
    results["silra"] = "configured" if silra_key else "missing API key"

    return results


# ---------------------------------------------------------------------------
# Recent log lines
# ---------------------------------------------------------------------------


@router.get("/logs/{job_key}")
async def job_logs(job_key: str, lines: int = 50) -> dict[str, Any]:
    log_map = {
        "translation": "translation_run.log",
        "embedding": "embedding_run.log",
        "keywords": "keyword_run.log",
        "communities": "community_run.log",
        "citations": "citation_run.log",
    }
    filename = log_map.get(job_key)
    if not filename:
        return {"error": "unknown job", "lines": []}
    log_path = _LOGS_DIR / filename
    if not log_path.exists():
        return {"lines": [], "path": str(log_path)}
    all_lines = log_path.read_text(errors="replace").splitlines()
    return {"lines": all_lines[-lines:], "total_lines": len(all_lines)}
