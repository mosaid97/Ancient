"""Evaluator — Phase 4 (plan §6 Stage 4).

Runs over OCR pages (``PAGE.mode = 'ocr'``) after fusion to:
1. Compute inter-engine Character Error Rate (CER) between PaddleOCR and fused output.
2. Compute CJK validity ratio (fraction of characters in CJK Unicode blocks).
3. Classify the page's OCR quality via :func:`classify_page` when CER exceeds
   the cheap-call threshold (avoids LLM cost on clean pages).
4. Route the page to one of three states:
   - ``'pass'``        — CER < 5 % AND class is OK/POLYSEMY/CULTURAL_REFERENCE
   - ``'needs_review'``— 5 % ≤ CER < 15 % OR class in review_classes OR JP class
   - ``'failed'``      — CER ≥ 15 % OR class in failed_classes (RARE_GLYPH/DEGRADATION)

Results are written back to Neo4j on the PAGE node:
  evaluationStatus, evaluationDecision, problemClass, problemClassConfidence,
  problemClassReasoning, interEngineCer, cjkValidityRatio, evaluatedAt

The evaluator also creates ``(:PROBLEM_CLASS)`` nodes (seeded once via
:func:`seed_problem_class_nodes`) and ``(:PAGE)-[:CLASSIFIED_AS]->(:PROBLEM_CLASS)``
relationships.

Public API
----------
EvaluatorResult         — per-page evaluation output
EvaluatorRunReport      — summary of an evaluation pass
evaluate_pages(driver, *, max_pages, recompute, client) -> EvaluatorRunReport
seed_problem_class_nodes(driver) -> None
"""
from __future__ import annotations

import logging
import os
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neo4j import Driver
from openai import OpenAI

from apps.backend.agents.problem_classifier import (
    ClassificationResult,
    ProblemClassDef,
    classify_page,
    load_problem_classes,
)
from apps.backend.llm.silra import get_silra_client

log = logging.getLogger(__name__)

# CER thresholds (plan §6 Stage 4)
_CER_PASS_MAX = 0.05          # < 5 % → PASS (no LLM call needed)
_CER_REVIEW_MAX = 0.15        # 5–15 % → NEEDS_REVIEW
_CER_LLM_CALL_MIN = 0.02      # < 2 % → skip expensive classifier call entirely

# Routing sets
_PASS_CLASSES = frozenset({"OK", "POLYSEMY", "CULTURAL_REFERENCE"})
_REVIEW_CLASSES = frozenset({
    "LAYOUT_AMBIGUITY", "READING_ORDER", "ANNOTATION",
    "EDITORIAL_VS_SOURCE", "KANBUN_KUNTEN", "OKURIGANA",
    "HENTAIGANA", "MIXED_SCRIPT",
})
_FAILED_CLASSES = frozenset({"RARE_GLYPH", "DEGRADATION"})

# CJK Unicode ranges used for validity ratio
_CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Ext-A
    (0x20000, 0x2A6DF), # CJK Ext-B
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0x31F0, 0x31FF),   # Katakana Phonetic Extensions
    (0xFF65, 0xFF9F),   # Half-width Katakana
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvaluatorResult:
    """Per-page evaluation output."""

    page_id: str
    evaluation_decision: str        # 'pass' | 'needs_review' | 'failed'
    problem_class: str              # PROBLEM_CLASS code
    problem_class_confidence: float
    problem_class_reasoning: str
    inter_engine_cer: float         # Paddle vs fused
    cjk_validity_ratio: float
    evaluated_at: str               # ISO-8601
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "evaluation_decision": self.evaluation_decision,
            "problem_class": self.problem_class,
            "problem_class_confidence": self.problem_class_confidence,
            "problem_class_reasoning": self.problem_class_reasoning,
            "inter_engine_cer": round(self.inter_engine_cer, 4),
            "cjk_validity_ratio": round(self.cjk_validity_ratio, 4),
            "evaluated_at": self.evaluated_at,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass
class EvaluatorRunReport:
    """Summary of an evaluation pass."""

    pages_total: int = 0
    pages_pass: int = 0
    pages_needs_review: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0
    llm_calls: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages_total": self.pages_total,
            "pages_pass": self.pages_pass,
            "pages_needs_review": self.pages_needs_review,
            "pages_failed": self.pages_failed,
            "pages_skipped": self.pages_skipped,
            "llm_calls": self.llm_calls,
            "duration_seconds": round(self.duration_seconds, 3),
            "errors": self.errors[:20],
        }


# ---------------------------------------------------------------------------
# CER + CJK helpers
# ---------------------------------------------------------------------------


def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance (character-level)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Use two-row DP for memory efficiency
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len(b)]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate: edit_distance / max(len(ref), 1).

    Args:
        reference: Ground-truth text (fused output).
        hypothesis: OCR engine text to compare against.

    Returns:
        CER in [0, ∞). Normally ≤ 1 for similar-length texts.
    """
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0
    ed = _edit_distance(ref, hyp)
    return ed / len(ref)


def compute_cjk_validity(text: str) -> float:
    """Return the fraction of non-space characters that are CJK.

    A low ratio (< 0.3) on a classical Chinese page suggests garbled OCR.

    Args:
        text: OCR output text.

    Returns:
        Float in [0, 1].
    """
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    cjk_count = sum(
        1
        for c in chars
        if any(lo <= ord(c) <= hi for lo, hi in _CJK_RANGES)
    )
    return cjk_count / len(chars)


def _route_decision(cer: float, problem_class: str) -> str:
    """Apply the plan §6 Stage 4 routing rules."""
    if problem_class in _FAILED_CLASSES or cer >= _CER_REVIEW_MAX:
        return "failed"
    if problem_class in _REVIEW_CLASSES or _CER_PASS_MAX <= cer < _CER_REVIEW_MAX:
        return "needs_review"
    return "pass"


# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

_PAGE_QUERY = """
MATCH (p:PAGE)
WHERE p.mode = 'ocr'
  AND p.fusionStatus IN ['ok', 'single']
  AND ($recompute OR p.evaluationStatus IS NULL)
RETURN
    p.id                AS page_id,
    p.textFused         AS text_fused,
    p.paddleOcrText     AS paddle_text,
    p.qwenVlOcrText     AS qwen_text,
    p.deepseekOcrText   AS deepseek_text,
    p.language          AS language,
    p.documentId        AS document_id,
    p.sectionId         AS section_id,
    p.tier              AS tier,
    p.fusionAgreementRate AS fusion_agreement_rate
ORDER BY p.id
LIMIT $batch
"""

_DOCUMENT_CONTEXT_QUERY = """
MATCH (d:DOCUMENT {id: $doc_id})
OPTIONAL MATCH (d)<-[:CONTAIN]-(t:TOPIC)
RETURN d.title AS title, d.tier AS tier, t.name AS topic
LIMIT 1
"""

_EVAL_WRITE = """
MATCH (p:PAGE {id: $page_id})
SET p.evaluationStatus        = 'evaluated',
    p.evaluationDecision      = $decision,
    p.problemClass            = $problem_class,
    p.problemClassConfidence  = $confidence,
    p.problemClassReasoning   = $reasoning,
    p.interEngineCer          = $cer,
    p.cjkValidityRatio        = $cjk_ratio,
    p.evaluatedAt             = $ts
"""

_CLASSIFIED_AS_REL = """
MATCH (p:PAGE {id: $page_id})
MATCH (pc:PROBLEM_CLASS {code: $code})
MERGE (p)-[:CLASSIFIED_AS]->(pc)
"""

_PROBLEM_CLASS_UPSERT = """
UNWIND $classes AS c
MERGE (pc:PROBLEM_CLASS {code: c.code})
ON CREATE SET pc.label = c.label, pc.description = c.description,
              pc.language = c.language, pc.routing = c.routing,
              pc.createdAt = $ts
ON MATCH SET  pc.label = c.label, pc.description = c.description,
              pc.language = c.language, pc.routing = c.routing
"""


# ---------------------------------------------------------------------------
# Seed PROBLEM_CLASS nodes
# ---------------------------------------------------------------------------


def seed_problem_class_nodes(driver: Driver) -> None:
    """Upsert all PROBLEM_CLASS nodes from the seed file.

    Idempotent — safe to call on every run.

    Args:
        driver: Open Neo4j driver.
    """
    defs = load_problem_classes()
    rows = [
        {
            "code": d.code,
            "label": d.label,
            "description": d.description[:200],
            "language": d.language,
            "routing": d.routing,
        }
        for d in defs
    ]
    ts = datetime.now(timezone.utc).isoformat()
    with driver.session() as s:
        s.run(_PROBLEM_CLASS_UPSERT, classes=rows, ts=ts).consume()
    log.info("seed_problem_class_nodes: upserted %d PROBLEM_CLASS nodes", len(rows))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_pages(
    driver: Driver,
    *,
    max_pages: int | None = None,
    recompute: bool = False,
    batch_size: int = 50,
    client: OpenAI | None = None,
) -> EvaluatorRunReport:
    """Evaluate all fused OCR pages and write results to Neo4j.

    Pages with ``fusionStatus ∈ {ok, single}`` and ``evaluationStatus IS NULL``
    are processed by default.  Set ``recompute=True`` to re-evaluate all.

    Decision routing per plan §6 Stage 4:
    - CER < 2 %  → fast PASS without LLM call
    - CER 2–5 %  → call classifier; route on result
    - CER ≥ 5 %  → call classifier; route on result + CER thresholds

    Args:
        driver: Open Neo4j driver.
        max_pages: Stop after processing this many pages.
        recompute: If True, re-evaluate already-evaluated pages.
        batch_size: Number of pages to fetch per Neo4j query.
        client: Re-use an existing Silra client.

    Returns:
        :class:`EvaluatorRunReport` with per-decision counts.
    """
    seed_problem_class_nodes(driver)

    defs = load_problem_classes()
    llm_client = client or get_silra_client()
    report = EvaluatorRunReport()
    t_start = time.time()
    doc_context_cache: dict[str, str] = {}

    while True:
        with driver.session() as s:
            rows = s.run(
                _PAGE_QUERY, recompute=recompute, batch=batch_size
            ).data()

        if not rows:
            break

        for row in rows:
            report.pages_total += 1
            page_id = row["page_id"]
            ts = datetime.now(timezone.utc).isoformat()

            fused = row.get("text_fused") or ""
            paddle = row.get("paddle_text") or ""
            qwen = row.get("qwen_text")
            deepseek = row.get("deepseek_text")
            language = row.get("language") or "zh-classical"
            doc_id = row.get("document_id") or ""

            # Skip pages with no usable text
            if not fused and not paddle:
                report.pages_skipped += 1
                _write_eval_skip(driver, page_id, ts, "no_text")
                continue

            # Inter-engine CER (Paddle vs fused)
            cer = compute_cer(fused, paddle) if fused else 0.0
            cjk_ratio = compute_cjk_validity(fused or paddle)

            # Classify — skip LLM call for very clean pages
            if cer < _CER_LLM_CALL_MIN:
                clf = ClassificationResult(
                    problem_class="OK",
                    confidence=1.0,
                    reasoning="CER below threshold — automatic PASS",
                )
            else:
                # Fetch document context for the prompt (cached)
                if doc_id not in doc_context_cache:
                    try:
                        with driver.session() as s:
                            ctx_row = s.run(
                                _DOCUMENT_CONTEXT_QUERY, doc_id=doc_id
                            ).single()
                        doc_context_cache[doc_id] = (
                            f"{ctx_row['title'] or ''} ({ctx_row['tier'] or ''})"
                            if ctx_row
                            else ""
                        )
                    except Exception:
                        doc_context_cache[doc_id] = ""

                context = doc_context_cache.get(doc_id, "")
                try:
                    clf = classify_page(
                        paddle_text=paddle,
                        fused_text=fused,
                        qwen_text=qwen,
                        deepseek_text=deepseek,
                        language=language,
                        context=context,
                        defs=defs,
                        client=llm_client,
                    )
                    report.llm_calls += 1
                except Exception as exc:
                    log.error("evaluate_pages: classify_page failed for %s: %s", page_id, exc)
                    clf = ClassificationResult(
                        problem_class="OK",
                        confidence=0.3,
                        reasoning=f"classifier error: {exc}",
                    )
                    report.errors.append(f"{page_id}: {exc}")

            decision = _route_decision(cer, clf.problem_class)

            if decision == "pass":
                report.pages_pass += 1
            elif decision == "needs_review":
                report.pages_needs_review += 1
            else:
                report.pages_failed += 1

            result = EvaluatorResult(
                page_id=page_id,
                evaluation_decision=decision,
                problem_class=clf.problem_class,
                problem_class_confidence=clf.confidence,
                problem_class_reasoning=clf.reasoning,
                inter_engine_cer=cer,
                cjk_validity_ratio=cjk_ratio,
                evaluated_at=ts,
            )
            _write_eval_result(driver, result)

            log.debug(
                "eval page=%s cer=%.3f class=%s decision=%s",
                page_id, cer, clf.problem_class, decision,
            )

        if max_pages is not None and report.pages_total >= max_pages:
            break

    report.duration_seconds = round(time.time() - t_start, 3)
    log.info(
        "evaluate_pages: total=%d pass=%d review=%d failed=%d skipped=%d "
        "llm_calls=%d in %.1fs",
        report.pages_total,
        report.pages_pass,
        report.pages_needs_review,
        report.pages_failed,
        report.pages_skipped,
        report.llm_calls,
        report.duration_seconds,
    )
    return report


def _write_eval_result(driver: Driver, result: EvaluatorResult) -> None:
    """Write evaluation result to Neo4j and wire the CLASSIFIED_AS relationship."""
    with driver.session() as s:
        s.run(
            _EVAL_WRITE,
            page_id=result.page_id,
            decision=result.evaluation_decision,
            problem_class=result.problem_class,
            confidence=result.problem_class_confidence,
            reasoning=result.problem_class_reasoning,
            cer=result.inter_engine_cer,
            cjk_ratio=result.cjk_validity_ratio,
            ts=result.evaluated_at,
        ).consume()
        # Wire CLASSIFIED_AS relationship
        s.run(
            _CLASSIFIED_AS_REL,
            page_id=result.page_id,
            code=result.problem_class,
        ).consume()


def _write_eval_skip(driver: Driver, page_id: str, ts: str, reason: str) -> None:
    """Stamp evaluationStatus='skipped' on a page we cannot evaluate."""
    with driver.session() as s:
        s.run(
            "MATCH (p:PAGE {id: $id}) "
            "SET p.evaluationStatus='skipped', p.evaluationSkipReason=$reason, "
            "p.evaluatedAt=$ts",
            id=page_id, reason=reason, ts=ts,
        ).consume()
