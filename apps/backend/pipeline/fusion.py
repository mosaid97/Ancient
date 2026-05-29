"""Phase-3d fusion orchestrator + post-OCR language detection (plan §6 Stage 3d, 3e).

Walks every PAGE whose ``mode='ocr'`` has either or both engine results
populated, runs :func:`apps.backend.ocr.fusion.fuse_results`, and writes
the fused text plus per-segment provenance back to the PAGE node.

Stored properties (camelCase per AGENTS.md §4):

- ``textFused`` — final transcription used by Phase 4 (eval) and Phase
  5 (translation).
- ``fusionAgreementRate`` — ``[0, 1]`` agreement between the two
  engines (None when only one engine ran).
- ``fusionSegmentsJson`` — per-segment alignment for the evaluator.
- ``fusionSingleEngine`` — ``'paddleocr'`` / ``'deepseek_ocr'`` /
  ``null`` (the latter means both engines contributed).
- ``fusionCharCount`` — len(textFused).
- ``fusionStatus`` — ``'ok' | 'single' | 'failed'``.
- ``fusionAt`` — Neo4j timestamp().

After fusion, the language detector (:mod:`apps.backend.lang.detector`)
re-classifies each page using the fused text (Phase-1b's heuristics ran
on native text only and skipped OCR pages). It writes the authoritative
``language`` / ``scriptMix`` / ``kuntenMarks`` / ``langConfidence`` /
``langDetectionRule`` properties. Plan §6 Stage 3e calls this "post-OCR
authoritative language detection" — it's the version every downstream
search index will rely on.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from neo4j import Driver

from apps.backend.lang.detector import detect_language
from apps.backend.ocr.base import OCRLine, OCRPageResult
from apps.backend.ocr.fusion import FusionResult, fuse_results

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cypher.
# ---------------------------------------------------------------------------


_SELECT_FUSE_PAGES = """
MATCH (p:PAGE)
WHERE p.mode = 'ocr'
  AND (p.role IS NULL OR p.role = 'body')
  AND (p.paddleOcrStatus = 'ok'
       OR p.deepseekOcrStatus = 'ok'
       OR p.qwenVlOcrStatus = 'ok')
  AND ($recompute = true OR p.fusionStatus IS NULL OR p.fusionStatus = 'failed')
  AND ($document_id IS NULL OR p.documentId = $document_id)
RETURN p.id AS page_id,
       p.documentId AS document_id,
       p.tier AS tier,
       p.paddleOcrStatus AS paddle_status,
       p.paddleOcrText AS paddle_text,
       p.paddleOcrLinesJson AS paddle_lines_json,
       p.paddleOcrConfidence AS paddle_confidence,
       p.paddleOcrModelVersion AS paddle_model,
       p.paddleOcrError AS paddle_error,
       p.deepseekOcrStatus AS deepseek_status,
       p.deepseekOcrText AS deepseek_text,
       p.deepseekOcrLinesJson AS deepseek_lines_json,
       p.deepseekOcrConfidence AS deepseek_confidence,
       p.deepseekOcrModelVersion AS deepseek_model,
       p.deepseekOcrError AS deepseek_error,
       p.qwenVlOcrStatus AS qwen_status,
       p.qwenVlOcrText AS qwen_text,
       p.qwenVlOcrLinesJson AS qwen_lines_json,
       p.qwenVlOcrConfidence AS qwen_confidence,
       p.qwenVlOcrModelVersion AS qwen_model,
       p.qwenVlOcrError AS qwen_error
ORDER BY p.documentId, p.docPageIndex, p.id
"""


_FUSION_UPDATE = """
MATCH (p:PAGE {id: $page_id})
SET p.textFused = $text_fused,
    p.fusionAgreementRate = $agreement_rate,
    p.fusionSegmentsJson = $segments_json,
    p.fusionSingleEngine = $single_engine,
    p.fusionWinnerLlm = $winner_llm,
    p.fusionCharCount = $char_count,
    p.fusionStatus = $status,
    p.fusionError = $error,
    p.fusionAt = timestamp()
RETURN p.id AS id
"""


_LANG_UPDATE = """
MATCH (p:PAGE {id: $page_id})
SET p.language = $language,
    p.scriptMix = $script_mix_json,
    p.kuntenMarks = $kunten_marks,
    p.langConfidence = $lang_confidence,
    p.langDetectionRule = $lang_detection_rule,
    p.langDetectionAt = timestamp()
RETURN p.id AS id
"""


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


@dataclass
class FuseOutcome:
    """Per-page fusion result."""

    page_id: str
    document_id: str
    status: str  # "ok" | "single" | "failed"
    agreement_rate: float | None = None
    single_engine: str | None = None
    winner_llm: str | None = None
    paddle_chars: int = 0
    deepseek_chars: int = 0
    qwen_chars: int = 0
    fused_chars: int = 0
    language: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FuseRunReport:
    """Aggregate report for :func:`fuse_pages`."""

    pages_total: int = 0
    pages_dual_fused: int = 0
    pages_single_engine: int = 0
    pages_failed: int = 0
    avg_agreement_rate: float | None = None
    duration_seconds: float = 0.0
    by_document: dict[str, dict[str, int]] = field(default_factory=dict)
    by_language: dict[str, int] = field(default_factory=dict)
    by_winner_llm: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    sample_outcomes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _hydrate_lines(lines_json: str | None) -> list[OCRLine]:
    if not lines_json:
        return []
    try:
        payload = json.loads(lines_json)
    except json.JSONDecodeError:
        return []
    out: list[OCRLine] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        bbox = entry.get("bbox")
        out.append(
            OCRLine(
                text=entry.get("text") or "",
                confidence=float(entry.get("confidence") or 0.0),
                bbox=tuple(bbox) if bbox and len(bbox) == 4 else None,
                order=int(entry.get("order") or 0),
            )
        )
    return out


def _row_to_paddle(row: dict[str, Any]) -> OCRPageResult:
    return OCRPageResult(
        engine="paddleocr",
        model_version=row.get("paddle_model") or "PP-OCRv5",
        page_id=row["page_id"],
        text=row.get("paddle_text") or "",
        lines=_hydrate_lines(row.get("paddle_lines_json")),
        confidence=float(row.get("paddle_confidence") or 0.0),
        char_count=len((row.get("paddle_text") or "").replace("\n", "").strip()),
        error=row.get("paddle_error") if row.get("paddle_status") != "ok" else None,
    )


def _row_to_deepseek(row: dict[str, Any]) -> OCRPageResult:
    return OCRPageResult(
        engine="deepseek_ocr",
        model_version=row.get("deepseek_model") or "deepseek-ocr",
        page_id=row["page_id"],
        text=row.get("deepseek_text") or "",
        lines=_hydrate_lines(row.get("deepseek_lines_json")),
        confidence=float(row.get("deepseek_confidence") or 0.0),
        char_count=len((row.get("deepseek_text") or "").replace("\n", "").strip()),
        error=row.get("deepseek_error") if row.get("deepseek_status") != "ok" else None,
    )


def _row_to_qwen(row: dict[str, Any]) -> OCRPageResult | None:
    """Return a Qwen-VL OCR result, or None if no Qwen data is present."""
    if not row.get("qwen_status"):
        return None
    return OCRPageResult(
        engine="qwen_vl_ocr",
        model_version=row.get("qwen_model") or "qwen-vl-ocr-latest",
        page_id=row["page_id"],
        text=row.get("qwen_text") or "",
        lines=_hydrate_lines(row.get("qwen_lines_json")),
        confidence=float(row.get("qwen_confidence") or 0.0),
        char_count=len((row.get("qwen_text") or "").replace("\n", "").strip()),
        error=row.get("qwen_error") if row.get("qwen_status") != "ok" else None,
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def fuse_pages(
    driver: Driver,
    *,
    document_id: str | None = None,
    max_pages: int | None = None,
    recompute_existing: bool = False,
    run_language_detection: bool = True,
    sample_size: int = 5,
    progress_every: int = 100,
) -> FuseRunReport:
    """Fuse Paddle + DeepSeek OCR per page, then re-detect language.

    Args:
        driver: Open Neo4j driver.
        document_id: Optional filter to one document.
        max_pages: Optional cap (smoke runs).
        recompute_existing: Re-run on pages already fused.
        run_language_detection: If ``True`` (default), classify the
            fused text and write the authoritative ``language`` etc.
        sample_size: Outcomes retained in ``sample_outcomes``.
        progress_every: Log every N pages.

    Returns:
        :class:`FuseRunReport`.
    """
    report = FuseRunReport()
    started = time.monotonic()
    agreement_sum = 0.0
    agreement_count = 0

    with driver.session() as session:
        rows = list(
            session.run(
                _SELECT_FUSE_PAGES,
                recompute=recompute_existing,
                document_id=document_id,
            )
        )
    if max_pages is not None:
        rows = rows[:max_pages]
    report.pages_total = len(rows)

    sample: list[FuseOutcome] = []

    for idx, raw_row in enumerate(rows, start=1):
        row = dict(raw_row)
        page_id = row["page_id"]
        doc_id = row["document_id"]
        doc_bucket = report.by_document.setdefault(
            doc_id, {"ok": 0, "single": 0, "failed": 0}
        )

        paddle_res = _row_to_paddle(row)
        deepseek_res = _row_to_deepseek(row)
        qwen_res = _row_to_qwen(row)

        fusion = fuse_results(paddle_res, deepseek_res, qwen_res)
        if fusion.error:
            status = "failed"
        elif fusion.single_engine:
            status = "single"
        else:
            status = "ok"

        # Write fusion.
        segments_json = json.dumps(
            [seg.to_dict() for seg in fusion.segments],
            ensure_ascii=False,
        )
        try:
            with driver.session() as session:
                session.run(
                    _FUSION_UPDATE,
                    page_id=page_id,
                    text_fused=fusion.text_fused,
                    agreement_rate=fusion.agreement_rate,
                    segments_json=segments_json,
                    single_engine=fusion.single_engine,
                    winner_llm=fusion.winner_llm,
                    char_count=fusion.fused_chars,
                    status=status,
                    error=fusion.error,
                ).consume()
        except Exception as exc:  # noqa: BLE001
            outcome = FuseOutcome(
                page_id=page_id, document_id=doc_id, status="failed",
                error=f"neo4j fusion write failed: {exc}",
            )
            report.pages_failed += 1
            doc_bucket["failed"] += 1
            report.errors.append(f"{page_id}: {outcome.error}")
            if len(sample) < sample_size:
                sample.append(outcome)
            continue

        # Post-OCR language detection.
        language: str | None = None
        if run_language_detection and fusion.text_fused:
            try:
                profile = detect_language(fusion.text_fused)
                payload = profile.to_dict()
                with driver.session() as session:
                    session.run(
                        _LANG_UPDATE,
                        page_id=page_id,
                        language=payload["language"],
                        script_mix_json=json.dumps(
                            payload["scriptMix"], ensure_ascii=False
                        ),
                        kunten_marks=bool(payload["kuntenMarks"]),
                        lang_confidence=float(payload["langConfidence"]),
                        lang_detection_rule=payload["langDetectionRule"],
                    ).consume()
                language = payload["language"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("lang detect failed for %s: %s", page_id, exc)

        outcome = FuseOutcome(
            page_id=page_id, document_id=doc_id, status=status,
            agreement_rate=fusion.agreement_rate,
            single_engine=fusion.single_engine,
            winner_llm=fusion.winner_llm,
            paddle_chars=fusion.paddle_chars,
            deepseek_chars=fusion.deepseek_chars,
            qwen_chars=fusion.qwen_chars,
            fused_chars=fusion.fused_chars,
            language=language,
            duration_seconds=fusion.duration_seconds,
            error=fusion.error,
        )
        if status == "ok":
            report.pages_dual_fused += 1
            doc_bucket["ok"] += 1
            agreement_sum += fusion.agreement_rate or 0.0
            agreement_count += 1
        elif status == "single":
            report.pages_single_engine += 1
            doc_bucket["single"] += 1
        else:
            report.pages_failed += 1
            doc_bucket["failed"] += 1
            if fusion.error:
                report.errors.append(f"{page_id}: {fusion.error}")

        if language:
            report.by_language[language] = report.by_language.get(language, 0) + 1
        if fusion.winner_llm:
            report.by_winner_llm[fusion.winner_llm] = (
                report.by_winner_llm.get(fusion.winner_llm, 0) + 1
            )

        if len(sample) < sample_size:
            sample.append(outcome)

        if idx % progress_every == 0:
            so_far = time.monotonic() - started
            logger.info(
                "fusion progress %d/%d (%.1fs, %.2fs/page)",
                idx, len(rows), so_far, so_far / max(idx, 1),
            )

    if agreement_count > 0:
        report.avg_agreement_rate = round(agreement_sum / agreement_count, 4)
    report.duration_seconds = round(time.monotonic() - started, 3)
    report.sample_outcomes = [o.to_dict() for o in sample]
    return report


# ---------------------------------------------------------------------------
# Summary for notebook checks.
# ---------------------------------------------------------------------------


def fusion_summary(driver: Driver) -> dict[str, Any]:
    """Roll up fusion + language coverage."""

    cypher = """
    MATCH (p:PAGE)
    WHERE p.mode = 'ocr' AND (p.role IS NULL OR p.role = 'body')
    WITH coalesce(p.fusionStatus, '(unset)') AS status,
         coalesce(p.language, '(unset)') AS language,
         coalesce(p.tier, '(unset)') AS tier,
         p.fusionAgreementRate AS agreement
    RETURN status, language, tier, count(*) AS n, avg(agreement) AS avg_agreement
    """
    by_status: dict[str, int] = {}
    by_language: dict[str, int] = {}
    by_tier: dict[str, dict[str, int]] = {}
    agreements: list[float] = []
    with driver.session() as session:
        for row in session.run(cypher):
            s = row["status"]
            lang = row["language"]
            t = row["tier"]
            n = row["n"]
            agr = row["avg_agreement"]
            by_status[s] = by_status.get(s, 0) + n
            by_language[lang] = by_language.get(lang, 0) + n
            bucket = by_tier.setdefault(t, {})
            bucket[s] = bucket.get(s, 0) + n
            if agr is not None:
                agreements.append(float(agr))
    overall_agreement = (
        round(sum(agreements) / len(agreements), 4) if agreements else None
    )
    return {
        "by_status": by_status,
        "by_language": by_language,
        "by_tier_status": by_tier,
        "avg_agreement_rate": overall_agreement,
    }
