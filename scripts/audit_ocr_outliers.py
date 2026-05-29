"""Per-engine outlier audit: hallucination signatures and char-count outliers.

Companion to ``audit_ocr_engines.py``.

Computes the *outlier rate* per engine — percent of pages whose
char-count exceeds 3x the cohort median, since LLM hallucinations
typically inflate char count 5-20x. Also reports:

- Page-level "engine-vs-cohort" disagreements (pages where Paddle is
  consistent with Qwen but DeepSeek diverges, and vice versa).
- Specific markdown / HTML / repetitive-heading hallucination
  signatures per LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from apps.backend.graph.neo4j_client import get_driver  # noqa: E402

logger = logging.getLogger("ocr_outliers")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
)


_MD_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_REPEATED_LINE_RE = re.compile(r"^(.+)$\n(?:\1$\n){2,}", re.MULTILINE)
_HTML_TABLE_RE = re.compile(r"<\s*(table|tr|td)[^>]*>", re.IGNORECASE)
_PIPE_TABLE_RE = re.compile(r"^\s*\|.*\|.*\|", re.MULTILINE)


def hallucination_signatures(text: str) -> dict[str, int]:
    if not text:
        return {
            "md_headers": 0,
            "repeated_lines": 0,
            "html_tables": 0,
            "pipe_tables": 0,
        }
    return {
        "md_headers": len(_MD_HEADER_RE.findall(text)),
        "repeated_lines": len(_REPEATED_LINE_RE.findall(text)),
        "html_tables": len(_HTML_TABLE_RE.findall(text)),
        "pipe_tables": len(_PIPE_TABLE_RE.findall(text)),
    }


def fetch(driver) -> list[dict[str, Any]]:
    cypher = """
    MATCH (p:PAGE)
    WHERE p.mode = 'ocr'
    RETURN
      p.id AS page_id,
      p.paddleOcrStatus AS p_status, p.paddleOcrCharCount AS p_chars, p.paddleOcrText AS p_text,
      p.deepseekOcrStatus AS d_status, p.deepseekOcrCharCount AS d_chars, p.deepseekOcrText AS d_text,
      p.qwenVlOcrStatus AS q_status, p.qwenVlOcrCharCount AS q_chars, p.qwenVlOcrText AS q_text
    """
    with driver.session() as session:
        return [dict(rec) for rec in session.run(cypher)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("logs/ocr_outliers_report.json"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    driver = get_driver()
    try:
        rows = fetch(driver)
    finally:
        driver.close()

    p_chars = [int(r["p_chars"] or 0) for r in rows if r.get("p_status") == "ok"]
    d_chars = [int(r["d_chars"] or 0) for r in rows if r.get("d_status") == "ok"]
    q_chars = [int(r["q_chars"] or 0) for r in rows if r.get("q_status") == "ok"]

    p_med = median(p_chars) if p_chars else 0
    d_med = median(d_chars) if d_chars else 0
    q_med = median(q_chars) if q_chars else 0

    # Outlier-rate using two thresholds:
    # (a) absolute: > 3x cohort median
    # (b) cross-engine: this engine's chars > 3x Paddle's chars (when both succeeded)
    out_abs = Counter()
    out_vs_paddle = Counter()
    sig_totals = defaultdict(int)
    sig_pages = Counter()

    for r in rows:
        for tag, status, chars, text, cohort_med in [
            ("paddle", r.get("p_status"), r.get("p_chars"), r.get("p_text"), p_med),
            ("deepseek", r.get("d_status"), r.get("d_chars"), r.get("d_text"), d_med),
            ("qwen", r.get("q_status"), r.get("q_chars"), r.get("q_text"), q_med),
        ]:
            if status != "ok":
                continue
            chars = int(chars or 0)
            if cohort_med and chars > 3 * cohort_med:
                out_abs[tag] += 1
            sigs = hallucination_signatures(text or "")
            for k, v in sigs.items():
                if v > 0:
                    sig_totals[(tag, k)] += v
                    sig_pages[(tag, k)] += 1

        if r.get("p_status") == "ok":
            p = int(r.get("p_chars") or 0)
            if p > 0:
                if r.get("d_status") == "ok":
                    d = int(r.get("d_chars") or 0)
                    if d > 3 * p:
                        out_vs_paddle["deepseek_3x_paddle"] += 1
                if r.get("q_status") == "ok":
                    q = int(r.get("q_chars") or 0)
                    if q > 3 * p:
                        out_vs_paddle["qwen_3x_paddle"] += 1

    by_engine = {}
    for tag, n in [
        ("paddle", len(p_chars)),
        ("deepseek", len(d_chars)),
        ("qwen", len(q_chars)),
    ]:
        by_engine[tag] = {
            "n_succeeded": n,
            "median_chars": [p_med, d_med, q_med][["paddle","deepseek","qwen"].index(tag)],
            "outlier_3x_cohort_median_count": out_abs[tag],
            "outlier_3x_cohort_median_pct": round(100 * out_abs[tag] / max(n, 1), 2),
            "hallucination_signature_pages": {
                k[1]: sig_pages[k] for k in sig_pages if k[0] == tag
            },
            "hallucination_signature_total_hits": {
                k[1]: sig_totals[k] for k in sig_totals if k[0] == tag
            },
        }

    report = {
        "page_count": len(rows),
        "by_engine": by_engine,
        "outliers_vs_paddle": dict(out_vs_paddle),
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"# Outlier audit ({len(rows)} pages)")
    print()
    for tag, s in by_engine.items():
        print(f"## {tag}")
        for k, v in s.items():
            print(f"  - {k}: {v}")
        print()
    print(f"## outliers vs paddle (engine chars > 3x paddle chars on same page)")
    for k, v in out_vs_paddle.items():
        print(f"  - {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
