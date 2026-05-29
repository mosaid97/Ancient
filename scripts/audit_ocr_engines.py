"""Corpus-wide OCR engine audit (PaddleOCR vs DeepSeek-OCR vs Qwen-VL-OCR).

Goal: produce hard numbers that decide whether Paddle should be the
**benchmark** (alignment anchor) or the **fallback** (safety net) in
the fusion stage.

Reports:

1. Coverage matrix: who succeeds on which pages, who is the only
   engine that read each page, etc.
2. Confidence distribution per engine.
3. Agreement / disagreement: pairwise SequenceMatcher ratios on pages
   where both engines produced text.
4. Quality proxies on Paddle text and on each LLM:
   - CJK ratio (legit-looking East Asian content)
   - mean line length
   - avg per-char confidence (for Paddle)
   - structural-noise score (HTML / markdown table / digit-stream
     contamination — already-known LLM hallucination patterns)
5. Per-document slices (so we can see which corpora are LLM-friendly
   vs which collapse to single-engine).

Output: ``logs/ocr_audit_report.json`` + a Markdown summary printed to
stdout.

Usage::

    uv run python scripts/audit_ocr_engines.py
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from apps.backend.graph.neo4j_client import get_driver  # noqa: E402

logger = logging.getLogger("ocr_audit")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
)


CJK_RANGES = [
    (0x3400, 0x4DBF),   # CJK Ext A
    (0x4E00, 0x9FFF),   # CJK
    (0x20000, 0x2A6DF),  # CJK Ext B
    (0x2A700, 0x2EBEF),  # CJK Ext C-F
    (0xF900, 0xFAFF),   # CJK Compat Ideographs
]


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in CJK_RANGES)


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for c in text if is_cjk(c))
    return cjk / max(len(text), 1)


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_DIGIT_STREAM_RE = re.compile(r"\b\d{2,}\b")
_TABLE_PIPE_RE = re.compile(r"\|[^|\n]*\|")
_SEQ_NUMS_RE = re.compile(r"\|\s*\d+\s*\|\s*\d+\s*\|")


def structural_noise_score(text: str) -> dict[str, Any]:
    """Heuristics for known LLM-OCR hallucination patterns.

    Higher = noisier (worse).
    """
    if not text:
        return {
            "html_tag_count": 0,
            "table_pipe_count": 0,
            "seq_num_table_count": 0,
            "digit_run_chars": 0,
            "digit_run_ratio": 0.0,
            "noise_index": 0.0,
        }
    html = len(_HTML_TAG_RE.findall(text))
    pipes = len(_TABLE_PIPE_RE.findall(text))
    seq = len(_SEQ_NUMS_RE.findall(text))
    digit_chars = sum(len(m.group(0)) for m in _DIGIT_STREAM_RE.finditer(text))
    digit_ratio = digit_chars / max(len(text), 1)
    noise_index = (
        0.4 * (1 - cjk_ratio(text))
        + 0.2 * min(html / 5, 1)
        + 0.2 * min(pipes / 10, 1)
        + 0.2 * min(seq / 3, 1)
    )
    return {
        "html_tag_count": html,
        "table_pipe_count": pipes,
        "seq_num_table_count": seq,
        "digit_run_chars": digit_chars,
        "digit_run_ratio": round(digit_ratio, 4),
        "noise_index": round(noise_index, 4),
    }


def line_count(text: str) -> int:
    return sum(1 for line in text.split("\n") if line.strip())


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return float(s[max(0, min(k, len(s) - 1))])


def fetch_pages(driver, sample_size: int | None) -> list[dict[str, Any]]:
    """Pull every OCR page that has at least one engine result."""
    cypher = """
    MATCH (p:PAGE)
    WHERE p.mode = 'ocr'
      AND (
        p.paddleOcrStatus IN ['ok','empty','failed']
        OR p.deepseekOcrStatus IN ['ok','empty','failed']
        OR p.qwenVlOcrStatus IN ['ok','empty','failed']
      )
    OPTIONAL MATCH (p)<-[:INCLUDE]-(:SECTION)<-[:INCLUDE]-(:CHAPTER)<-[:CONSIST_OF]-(d:DOCUMENT)
    RETURN
      p.id AS page_id,
      coalesce(d.id, 'unknown') AS document_id,
      coalesce(d.tier, 'unknown') AS tier,
      coalesce(p.language, 'unknown') AS language,
      p.paddleOcrStatus AS p_status,
      p.paddleOcrText AS p_text,
      p.paddleOcrConfidence AS p_conf,
      p.paddleOcrCharCount AS p_chars,
      p.deepseekOcrStatus AS d_status,
      p.deepseekOcrText AS d_text,
      p.deepseekOcrConfidence AS d_conf,
      p.deepseekOcrCharCount AS d_chars,
      p.deepseekOcrError AS d_error,
      p.qwenVlOcrStatus AS q_status,
      p.qwenVlOcrText AS q_text,
      p.qwenVlOcrConfidence AS q_conf,
      p.qwenVlOcrCharCount AS q_chars,
      p.qwenVlOcrError AS q_error
    """
    if sample_size:
        cypher += f"\nLIMIT {sample_size}"
    with driver.session() as session:
        result = session.run(cypher)
        return [dict(record) for record in result]


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.monotonic()
    n = len(rows)
    if n == 0:
        return {"page_count": 0, "note": "no rows"}

    coverage = Counter()
    only = Counter()
    status_p, status_d, status_q = Counter(), Counter(), Counter()
    by_doc_engines: dict[str, Counter] = defaultdict(Counter)
    by_tier_engines: dict[str, Counter] = defaultdict(Counter)
    by_lang_engines: dict[str, Counter] = defaultdict(Counter)

    p_confs, d_confs, q_confs = [], [], []
    p_chars_l, d_chars_l, q_chars_l = [], [], []
    p_lines_l, d_lines_l, q_lines_l = [], [], []
    p_cjk_l, d_cjk_l, q_cjk_l = [], [], []
    d_noise_l, q_noise_l, p_noise_l = [], [], []

    pair_ratios_pd, pair_ratios_pq, pair_ratios_dq = [], [], []
    pair_count_pd = pair_count_pq = pair_count_dq = 0

    p_low_conf_q_high = 0
    q_low_conf_p_high = 0
    d_low_conf_p_high = 0

    pd_disagree_d_better, pd_disagree_p_better = 0, 0
    pq_disagree_q_better, pq_disagree_p_better = 0, 0

    sample_disagreements: list[dict[str, Any]] = []

    for r in rows:
        p_ok = (r.get("p_status") == "ok") and bool(r.get("p_text"))
        d_ok = (r.get("d_status") == "ok") and bool(r.get("d_text"))
        q_ok = (r.get("q_status") == "ok") and bool(r.get("q_text"))

        engines = []
        if p_ok:
            engines.append("p")
        if d_ok:
            engines.append("d")
        if q_ok:
            engines.append("q")
        coverage[",".join(engines) or "none"] += 1

        if engines == ["p"]:
            only["paddle"] += 1
        elif engines == ["d"]:
            only["deepseek"] += 1
        elif engines == ["q"]:
            only["qwen"] += 1
        elif len(engines) == 3:
            only["all_three"] += 1
        elif "d" not in engines and "q" not in engines:
            only["paddle_only_path"] += 0
        elif "p" not in engines and len(engines) >= 1:
            only["llm_only"] += 1

        status_p[r.get("p_status") or "(null)"] += 1
        status_d[r.get("d_status") or "(null)"] += 1
        status_q[r.get("q_status") or "(null)"] += 1

        doc_id = r["document_id"]
        tier = r["tier"]
        lang = r["language"]
        for tag, ok in (("p", p_ok), ("d", d_ok), ("q", q_ok)):
            if ok:
                by_doc_engines[doc_id][tag] += 1
                by_tier_engines[tier][tag] += 1
                by_lang_engines[lang][tag] += 1
        by_doc_engines[doc_id]["total"] += 1
        by_tier_engines[tier]["total"] += 1
        by_lang_engines[lang]["total"] += 1

        if p_ok:
            txt = r["p_text"] or ""
            p_confs.append(float(r["p_conf"] or 0.0))
            p_chars_l.append(int(r.get("p_chars") or len(txt)))
            p_lines_l.append(line_count(txt))
            p_cjk_l.append(cjk_ratio(txt))
            p_noise_l.append(structural_noise_score(txt)["noise_index"])
        if d_ok:
            txt = r["d_text"] or ""
            d_confs.append(float(r["d_conf"] or 0.0))
            d_chars_l.append(int(r.get("d_chars") or len(txt)))
            d_lines_l.append(line_count(txt))
            d_cjk_l.append(cjk_ratio(txt))
            d_noise_l.append(structural_noise_score(txt)["noise_index"])
        if q_ok:
            txt = r["q_text"] or ""
            q_confs.append(float(r["q_conf"] or 0.0))
            q_chars_l.append(int(r.get("q_chars") or len(txt)))
            q_lines_l.append(line_count(txt))
            q_cjk_l.append(cjk_ratio(txt))
            q_noise_l.append(structural_noise_score(txt)["noise_index"])

        # ---------- pairwise alignment ratios (sample limit 600 per pair) ----------
        if p_ok and d_ok and pair_count_pd < 600:
            ratio = difflib.SequenceMatcher(
                a=r["p_text"][:2000], b=r["d_text"][:2000], autojunk=False
            ).ratio()
            pair_ratios_pd.append(ratio)
            pair_count_pd += 1
            if ratio < 0.5:
                d_noise = structural_noise_score(r["d_text"])
                p_noise = structural_noise_score(r["p_text"])
                if d_noise["noise_index"] > p_noise["noise_index"] + 0.1:
                    pd_disagree_p_better += 1
                elif p_noise["noise_index"] > d_noise["noise_index"] + 0.1:
                    pd_disagree_d_better += 1
                if len(sample_disagreements) < 30:
                    sample_disagreements.append({
                        "page_id": r["page_id"],
                        "pair": "paddle_vs_deepseek",
                        "ratio": round(ratio, 3),
                        "p_chars": len(r["p_text"]),
                        "d_chars": len(r["d_text"]),
                        "p_conf": r["p_conf"],
                        "d_conf": r["d_conf"],
                        "p_noise": p_noise["noise_index"],
                        "d_noise": d_noise["noise_index"],
                        "p_cjk_ratio": round(cjk_ratio(r["p_text"]), 3),
                        "d_cjk_ratio": round(cjk_ratio(r["d_text"]), 3),
                        "p_text_head": r["p_text"][:120],
                        "d_text_head": r["d_text"][:120],
                    })
        if p_ok and q_ok and pair_count_pq < 600:
            ratio = difflib.SequenceMatcher(
                a=r["p_text"][:2000], b=r["q_text"][:2000], autojunk=False
            ).ratio()
            pair_ratios_pq.append(ratio)
            pair_count_pq += 1
            if ratio < 0.5:
                q_noise = structural_noise_score(r["q_text"])
                p_noise = structural_noise_score(r["p_text"])
                if q_noise["noise_index"] > p_noise["noise_index"] + 0.1:
                    pq_disagree_p_better += 1
                elif p_noise["noise_index"] > q_noise["noise_index"] + 0.1:
                    pq_disagree_q_better += 1
                if len(sample_disagreements) < 60:
                    sample_disagreements.append({
                        "page_id": r["page_id"],
                        "pair": "paddle_vs_qwen",
                        "ratio": round(ratio, 3),
                        "p_chars": len(r["p_text"]),
                        "q_chars": len(r["q_text"]),
                        "p_conf": r["p_conf"],
                        "q_conf": r["q_conf"],
                        "p_noise": p_noise["noise_index"],
                        "q_noise": q_noise["noise_index"],
                        "p_cjk_ratio": round(cjk_ratio(r["p_text"]), 3),
                        "q_cjk_ratio": round(cjk_ratio(r["q_text"]), 3),
                        "p_text_head": r["p_text"][:120],
                        "q_text_head": r["q_text"][:120],
                    })
        if d_ok and q_ok and pair_count_dq < 600:
            ratio = difflib.SequenceMatcher(
                a=r["d_text"][:2000], b=r["q_text"][:2000], autojunk=False
            ).ratio()
            pair_ratios_dq.append(ratio)
            pair_count_dq += 1

        # confidence-discord patterns
        if r.get("p_conf") is not None and r.get("q_conf") is not None and p_ok and q_ok:
            if (r["p_conf"] or 0) < 0.6 and (r["q_conf"] or 0) >= 0.85:
                p_low_conf_q_high += 1
            if (r["q_conf"] or 0) < 0.6 and (r["p_conf"] or 0) >= 0.85:
                q_low_conf_p_high += 1
        if r.get("p_conf") is not None and r.get("d_conf") is not None and p_ok and d_ok:
            if (r["d_conf"] or 0) < 0.6 and (r["p_conf"] or 0) >= 0.85:
                d_low_conf_p_high += 1

    def stats(label: str, vals: list[float]) -> dict[str, Any]:
        if not vals:
            return {label: None}
        return {
            "n": len(vals),
            "mean": round(mean(vals), 4),
            "median": round(median(vals), 4),
            "p10": round(percentile(vals, 10), 4),
            "p90": round(percentile(vals, 90), 4),
        }

    out = {
        "page_count": n,
        "duration_seconds": round(time.monotonic() - started, 2),
        "coverage_matrix": dict(coverage),
        "single_engine_breakdown": dict(only),
        "status_per_engine": {
            "paddle": dict(status_p),
            "deepseek": dict(status_d),
            "qwen": dict(status_q),
        },
        "confidence_stats": {
            "paddle": stats("paddle_conf", p_confs),
            "deepseek": stats("deepseek_conf", d_confs),
            "qwen": stats("qwen_conf", q_confs),
        },
        "char_count_stats": {
            "paddle": stats("paddle_chars", p_chars_l),
            "deepseek": stats("deepseek_chars", d_chars_l),
            "qwen": stats("qwen_chars", q_chars_l),
        },
        "line_count_stats": {
            "paddle": stats("paddle_lines", p_lines_l),
            "deepseek": stats("deepseek_lines", d_lines_l),
            "qwen": stats("qwen_lines", q_lines_l),
        },
        "cjk_ratio_stats": {
            "paddle": stats("paddle_cjk", p_cjk_l),
            "deepseek": stats("deepseek_cjk", d_cjk_l),
            "qwen": stats("qwen_cjk", q_cjk_l),
        },
        "structural_noise_stats": {
            "paddle": stats("paddle_noise", p_noise_l),
            "deepseek": stats("deepseek_noise", d_noise_l),
            "qwen": stats("qwen_noise", q_noise_l),
        },
        "pair_alignment_ratio": {
            "paddle_vs_deepseek": stats("ratio_pd", pair_ratios_pd),
            "paddle_vs_qwen":     stats("ratio_pq", pair_ratios_pq),
            "deepseek_vs_qwen":   stats("ratio_dq", pair_ratios_dq),
        },
        "low_alignment_judgement": {
            "paddle_vs_deepseek_p_better": pd_disagree_p_better,
            "paddle_vs_deepseek_d_better": pd_disagree_d_better,
            "paddle_vs_qwen_p_better": pq_disagree_p_better,
            "paddle_vs_qwen_q_better": pq_disagree_q_better,
        },
        "confidence_discord": {
            "p_low_conf_but_q_high": p_low_conf_q_high,
            "q_low_conf_but_p_high": q_low_conf_p_high,
            "d_low_conf_but_p_high": d_low_conf_p_high,
        },
        "by_tier_engines": {
            t: dict(c) for t, c in by_tier_engines.items()
        },
        "by_language_engines": {
            ln: dict(c) for ln, c in by_lang_engines.items()
        },
        "by_document_top_paddle_only": [],
        "by_document_top_qwen_only": [],
        "sample_disagreements": sample_disagreements,
    }

    # Top docs where Paddle did the heavy lifting (vs LLMs failing)
    paddle_only_per_doc = []
    qwen_only_per_doc = []
    for doc, c in by_doc_engines.items():
        if c["total"] < 5:
            continue
        only_p = c["p"] - max(c["d"], c["q"])
        only_q = c["q"] - max(c["p"], c["d"])
        paddle_only_per_doc.append((doc, only_p, c["total"]))
        qwen_only_per_doc.append((doc, only_q, c["total"]))
    paddle_only_per_doc.sort(key=lambda x: x[1], reverse=True)
    qwen_only_per_doc.sort(key=lambda x: x[1], reverse=True)
    out["by_document_top_paddle_only"] = [
        {"document_id": d, "paddle_advantage": adv, "total_pages": t}
        for d, adv, t in paddle_only_per_doc[:15]
    ]
    out["by_document_top_qwen_only"] = [
        {"document_id": d, "qwen_advantage": adv, "total_pages": t}
        for d, adv, t in qwen_only_per_doc[:15]
    ]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None,
                   help="Optional LIMIT for fast iteration.")
    p.add_argument("--out", type=Path,
                   default=Path("logs/ocr_audit_report.json"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    driver = get_driver()
    try:
        logger.info("fetching OCR pages from Neo4j ...")
        rows = fetch_pages(driver, args.sample)
        logger.info("fetched %d pages, analyzing ...", len(rows))
        report = analyze(rows)
    finally:
        driver.close()

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info("wrote %s", args.out)

    # Markdown summary on stdout
    print()
    print(f"# OCR engine audit ({report['page_count']} pages)")
    print()
    print("## Coverage matrix (engines that succeeded per page)")
    for k, v in sorted(report["coverage_matrix"].items(), key=lambda x: -x[1]):
        print(f"- `{k}` → {v} pages")
    print()
    print("## Single-engine outcomes")
    for k, v in sorted(report["single_engine_breakdown"].items(), key=lambda x: -x[1]):
        print(f"- {k}: {v}")
    print()
    print("## Confidence stats (succeeded pages only)")
    for eng, s in report["confidence_stats"].items():
        print(f"- {eng}: {s}")
    print()
    print("## Pairwise alignment ratio (SequenceMatcher; 1.0 = identical)")
    for k, s in report["pair_alignment_ratio"].items():
        print(f"- {k}: {s}")
    print()
    print("## Structural-noise index (lower is cleaner)")
    for eng, s in report["structural_noise_stats"].items():
        print(f"- {eng}: {s}")
    print()
    print("## CJK ratio (legitimate CJK content)")
    for eng, s in report["cjk_ratio_stats"].items():
        print(f"- {eng}: {s}")
    print()
    print("## Low-alignment judgement (pairs whose ratio < 0.5)")
    print(f"  {report['low_alignment_judgement']}")
    print()
    print("## Confidence-discord cases")
    print(f"  {report['confidence_discord']}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
