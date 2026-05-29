"""Character-level align-and-vote fusion (plan §6 Stage 3d).

3-engine variant: PaddleOCR (anchor) + Qwen-VL-OCR (primary LLM) + DeepSeek-OCR (secondary).

Algorithm
---------
1. **Quality gate** each LLM result using three corpus-tuned signals.
   Silra returns **hardcoded confidence constants** (DeepSeek=0.85, Qwen=0.82)
   that carry zero quality information and must NOT be used for gating or
   tie-breaking.  The gate replaces the retired ``llm_confidence_threshold``
   parameter with three explicit signals:

   - ``char_count <= char_ratio_limit × paddle_char_count``
     (blocks DeepSeek's 32.67 % length-outlier hallucinations)
   - ``cjk_ratio >= cjk_ratio_min``
     (blocks low-CJK / Latin-dominated outputs)
   - ``noise_index <= noise_index_max``
     (blocks markdown-header / pipe-table contamination)

   Paddle is **always** trusted when it succeeds; it has calibrated per-line
   confidence scores from PP-OCRv5 and a 0.28 % outlier rate.

2. **Engine pool & best-LLM selection** (priority order, NOT confidence):

   - Among LLMs that pass the gate, **Qwen wins by default**.
     DeepSeek wins only when Qwen is excluded.
     Rationale: audit confirmed Qwen has comparable noise to Paddle (0.25 %
     length-outliers) while DeepSeek has 32.67 % with 380 pages of hallucinated
     markdown scaffolding (see AGENTS.md §11 entry 2026-05-20).

3. **2-way SequenceMatcher** (Paddle as ``a``, best-LLM as ``b``):

   - ``equal`` → both agree; confidence 1.0.
   - ``replace`` → **LLM wins by default** (rare-glyph recovery is the LLM's
     job; the quality gate has already validated the LLM at page level).
     Confidence = ``insert_delete_confidence`` (reflects the disagreement).
   - ``insert`` → LLM adds chars Paddle missed; kept at ``insert_delete_confidence``.
   - ``delete`` → Paddle's extra chars kept at ``insert_delete_confidence``
     (preserve Paddle's layout detections).

4. **Single-engine fallback**: if all LLMs fail the gate, Paddle text is used
   verbatim with ``single_engine='paddleocr'``.

5. **Both engines failed** → empty text + error.

``FusionResult`` carries ``winner_llm`` (which LLM was chosen, or None),
``qwen_chars``, ``deepseek_chars``, and ``paddle_chars`` alongside the fused
text and per-character confidence list.
"""

from __future__ import annotations

import difflib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from apps.backend.ocr.base import OCRLine, OCRPageResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality-gate helpers (same metric definitions as scripts/audit_ocr_engines.py)
# ---------------------------------------------------------------------------

_CJK_RANGES: list[tuple[int, int]] = [
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0x2A700, 0x2EBEF),  # CJK Extensions C-F
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
]

_HTML_TAG_RE: re.Pattern[str] = re.compile(r"<[^>]+>")
_TABLE_PIPE_RE: re.Pattern[str] = re.compile(r"\|[^|\n]*\|")
_SEQ_NUM_RE: re.Pattern[str] = re.compile(r"\|\s*\d+\s*\|\s*\d+\s*\|")


def _is_cjk(c: str) -> bool:
    cp = ord(c)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _cjk_ratio(text: str) -> float:
    """Fraction of all characters that are CJK ideographs."""
    if not text:
        return 0.0
    cjk = sum(1 for c in text if _is_cjk(c))
    return cjk / max(len(text), 1)


def _noise_index(text: str) -> float:
    """Structural-noise index (0 = clean, 1 = fully noisy).

    Weighted combination of non-CJK fraction + HTML-tag density +
    pipe-table density + sequential-number-table density.  Corpus-tuned
    thresholds: median Paddle = 0.098, median Qwen = 0.090, p90 DeepSeek = 0.50.
    """
    if not text:
        return 0.0
    cjk_r = _cjk_ratio(text)
    html = len(_HTML_TAG_RE.findall(text))
    pipes = len(_TABLE_PIPE_RE.findall(text))
    seq = len(_SEQ_NUM_RE.findall(text))
    return (
        0.4 * (1.0 - cjk_r)
        + 0.2 * min(html / 5.0, 1.0)
        + 0.2 * min(pipes / 10.0, 1.0)
        + 0.2 * min(seq / 3.0, 1.0)
    )


def _llm_quality_gate(
    llm_result: OCRPageResult,
    paddle_result: OCRPageResult,
    *,
    char_ratio_limit: float,
    cjk_ratio_min: float,
    noise_index_max: float,
) -> tuple[bool, str | None]:
    """Return ``(passes, reason_if_failed)`` for a single LLM result.

    Paddle's char-count is used as the reference for the length-outlier gate.
    If Paddle produced no output, the char-count gate is skipped (only CJK and
    noise are checked) so a lone-LLM page can still be accepted.
    """
    text = llm_result.text or ""
    paddle_chars = max(paddle_result.char_count or 0, 1)

    if paddle_result.char_count and paddle_result.char_count > 0:
        char_ratio = len(text) / paddle_chars
        if char_ratio > char_ratio_limit:
            return False, (
                f"char_ratio={char_ratio:.1f} > limit={char_ratio_limit:.1f}"
            )

    if len(text) > 100:
        cr = _cjk_ratio(text)
        if cr < cjk_ratio_min:
            return False, f"cjk_ratio={cr:.3f} < min={cjk_ratio_min:.3f}"

        ni = _noise_index(text)
        if ni > noise_index_max:
            return False, f"noise_index={ni:.3f} > max={noise_index_max:.3f}"

    return True, None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FusionSegment:
    """One contiguous slice of the fused output."""

    text: str
    source: str   # "agree" | "paddleocr" | "qwen_vl_ocr" | "deepseek_ocr" | "noop"
    op: str       # difflib opcode: "equal" / "replace" / "insert" / "delete"
    a_range: tuple[int, int]
    b_range: tuple[int, int]
    char_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "op": self.op,
            "aRange": list(self.a_range),
            "bRange": list(self.b_range),
            "charConfidence": round(float(self.char_confidence), 4),
        }


@dataclass
class FusionResult:
    """Output of :func:`fuse_results` (3-engine variant)."""

    page_id: str
    text_fused: str
    char_confidences: list[float] = field(default_factory=list)
    segments: list[FusionSegment] = field(default_factory=list)
    agreement_rate: float | None = None
    single_engine: str | None = None
    winner_llm: str | None = None   # 'qwen_vl_ocr' | 'deepseek_ocr' | None
    paddle_chars: int = 0
    deepseek_chars: int = 0
    qwen_chars: int = 0
    fused_chars: int = 0
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pageId": self.page_id,
            "textFused": self.text_fused,
            "agreementRate": (
                None if self.agreement_rate is None
                else round(float(self.agreement_rate), 4)
            ),
            "singleEngine": self.single_engine,
            "winnerLlm": self.winner_llm,
            "paddleChars": self.paddle_chars,
            "deepseekChars": self.deepseek_chars,
            "qwenChars": self.qwen_chars,
            "fusedChars": self.fused_chars,
            "durationSeconds": round(float(self.duration_seconds), 3),
            "segments": [seg.to_dict() for seg in self.segments],
            "error": self.error,
        }

    def to_ocr_page_result(self, *, model_version: str = "fused-v1") -> OCRPageResult:
        """Adapt a FusionResult to the canonical OCRPageResult shape."""

        lines: list[OCRLine] = []
        for idx, raw in enumerate(self.text_fused.split("\n")):
            chunk = raw.strip()
            if chunk:
                lines.append(OCRLine(text=chunk, confidence=1.0, order=idx))
        conf = (
            (sum(self.char_confidences) / len(self.char_confidences))
            if self.char_confidences else 0.0
        )
        return OCRPageResult(
            engine="paddleocr",  # schema-wise fused is recorded under its own property;
                                  # this adapter provides a uniform shape for callers.
            model_version=model_version,
            page_id=self.page_id,
            text=self.text_fused,
            lines=lines,
            confidence=round(conf, 4),
            char_count=len(self.text_fused),
            duration_seconds=self.duration_seconds,
            metadata={
                "agreementRate": self.agreement_rate,
                "singleEngine": self.single_engine,
            },
        )


# ---------------------------------------------------------------------------
# Inner 2-way alignment
# ---------------------------------------------------------------------------


def _normalize_for_alignment(text: str) -> str:
    """Return text as-is; normalisation hook reserved for future use."""
    return text


def _fuse_two(
    a_result: OCRPageResult,   # Paddle — alignment anchor
    b_result: OCRPageResult,   # best-LLM — character-correction source
    *,
    insert_delete_confidence: float,
) -> tuple[str, list[float], list[FusionSegment], float]:
    """Inner 2-way SequenceMatcher alignment.

    Resolution rules (role-asymmetric per plan §6 Stage 3d):

    - ``equal``  → both engines agree; confidence 1.0.
    - ``replace`` → **LLM (b) wins**; the quality gate already validated b at
      page level, so its character choices are trusted for rare-glyph correction.
      Confidence = ``insert_delete_confidence`` (reflects the disagreement).
    - ``insert``  → LLM adds characters Paddle missed; kept at
      ``insert_delete_confidence``.
    - ``delete``  → Paddle's extra characters kept at ``insert_delete_confidence``
      (preserves Paddle's layout detections that the LLM may have skipped).

    Returns:
        ``(fused_text, char_confidences, segments, agreement_rate)``
    """
    a_conf = a_result.confidence or 0.0
    b_conf = b_result.confidence or 0.0
    a = _normalize_for_alignment(a_result.text)
    b = _normalize_for_alignment(b_result.text)
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    fused_parts: list[str] = []
    confidences: list[float] = []
    segments: list[FusionSegment] = []
    agree_chars = 0

    a_name = a_result.engine
    b_name = b_result.engine

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        a_span = a[i1:i2]
        b_span = b[j1:j2]

        if tag == "equal":
            fused_parts.append(a_span)
            confidences.extend([1.0] * len(a_span))
            segments.append(FusionSegment(
                text=a_span, source="agree", op="equal",
                a_range=(i1, i2), b_range=(j1, j2),
                char_confidence=max(a_conf, b_conf),
            ))
            agree_chars += len(a_span)

        elif tag == "replace":
            # LLM (b) wins: character correction is its purpose.  The page-level
            # quality gate validated b before this call, so we trust its glyph
            # choices over Paddle in disagreement regions.
            fused_parts.append(b_span)
            confidences.extend([insert_delete_confidence] * len(b_span))
            segments.append(FusionSegment(
                text=b_span, source=b_name, op="replace",
                a_range=(i1, i2), b_range=(j1, j2),
                char_confidence=insert_delete_confidence,
            ))

        elif tag == "insert":
            # LLM adds characters Paddle did not detect.
            fused_parts.append(b_span)
            confidences.extend([insert_delete_confidence] * len(b_span))
            segments.append(FusionSegment(
                text=b_span, source=b_name, op="insert",
                a_range=(i1, i2), b_range=(j1, j2),
                char_confidence=insert_delete_confidence,
            ))

        elif tag == "delete":
            # Paddle has characters the LLM skipped; keep them (Paddle's layout
            # detections — e.g., marginal numbers, punctuation runs — are
            # structurally reliable).
            fused_parts.append(a_span)
            confidences.extend([insert_delete_confidence] * len(a_span))
            segments.append(FusionSegment(
                text=a_span, source=a_name, op="delete",
                a_range=(i1, i2), b_range=(j1, j2),
                char_confidence=insert_delete_confidence,
            ))

        else:  # pragma: no cover
            raise RuntimeError(f"unknown difflib opcode: {tag}")

    fused = "".join(fused_parts)
    rate = (agree_chars / len(fused)) if fused else 0.0
    return fused, confidences, segments, round(rate, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fuse_results(
    paddle_result: OCRPageResult,
    deepseek_result: OCRPageResult | None = None,
    qwen_result: OCRPageResult | None = None,
    *,
    char_ratio_limit: float = 3.0,
    cjk_ratio_min: float = 0.40,
    noise_index_max: float = 0.30,
    insert_delete_confidence: float = 0.7,
    # Deprecated — Silra confidence constants are hardcoded and carry no quality
    # information.  Kept for backward-compatibility; value is ignored.
    llm_confidence_threshold: float = 0.60,  # noqa: ARG001
) -> FusionResult:
    """3-engine quality-signal-gated fusion: Paddle + Qwen-VL-OCR + DeepSeek-OCR.

    LLM engines that fail the quality gate (char-count ratio, CJK ratio, or
    structural-noise index) are excluded so Paddle acts as the safe fallback
    (plan §6 Stage 3d).  ``llm_confidence_threshold`` is **ignored** — the
    Silra API returns hardcoded constants (DeepSeek=0.85, Qwen=0.82) with zero
    variance that are useless as quality signals.

    Engine priority when both LLMs pass the gate: **Qwen wins by default**.
    DeepSeek's 32.67 % length-outlier rate disqualifies it as a tie-breaker;
    Qwen's outlier rate (0.25 %) is on par with Paddle's (0.28 %).

    Replace-opcode resolution: **LLM wins** (rare-glyph correction is its job).
    Agreement-rate still measures the equal-opcode fraction so Phase 4 / 5 can
    use it as the uncertainty signal for active-learning prioritisation.

    Args:
        paddle_result: PaddleOCR output (always trusted when it succeeds).
        deepseek_result: DeepSeek-OCR result, or ``None`` if not yet run.
        qwen_result: Qwen-VL-OCR result, or ``None`` if not yet run.
        char_ratio_limit: LLM char-count must be ≤ this multiple of Paddle's.
        cjk_ratio_min: LLM CJK-ratio must be ≥ this value (applied when
            ``len(text) > 100``).
        noise_index_max: LLM structural-noise index must be ≤ this value.
        insert_delete_confidence: Per-char confidence assigned to replace /
            insert / delete segments (reflects the disagreement penalty).
        llm_confidence_threshold: **Deprecated, ignored.**  Kept to avoid
            breaking callers that pass it as a keyword argument.

    Returns:
        :class:`FusionResult`.
    """
    started = time.monotonic()
    page_id = paddle_result.page_id

    paddle_chars = paddle_result.char_count
    deepseek_chars = deepseek_result.char_count if deepseek_result else 0
    qwen_chars = qwen_result.char_count if qwen_result else 0

    # ---------- engine availability ----------
    p_ok = paddle_result.succeeded

    # Quality gate for each LLM (replaces the retired confidence-threshold gate).
    ds_ok = False
    ds_gate_reason: str | None = None
    if deepseek_result is not None and deepseek_result.succeeded:
        ds_ok, ds_gate_reason = _llm_quality_gate(
            deepseek_result, paddle_result,
            char_ratio_limit=char_ratio_limit,
            cjk_ratio_min=cjk_ratio_min,
            noise_index_max=noise_index_max,
        )
        if not ds_ok:
            logger.debug(
                "page %s: DeepSeek excluded by quality gate: %s",
                page_id, ds_gate_reason,
            )

    qw_ok = False
    qw_gate_reason: str | None = None
    if qwen_result is not None and qwen_result.succeeded:
        qw_ok, qw_gate_reason = _llm_quality_gate(
            qwen_result, paddle_result,
            char_ratio_limit=char_ratio_limit,
            cjk_ratio_min=cjk_ratio_min,
            noise_index_max=noise_index_max,
        )
        if not qw_ok:
            logger.debug(
                "page %s: Qwen excluded by quality gate: %s",
                page_id, qw_gate_reason,
            )

    # ---------- best-LLM selection (priority order, NOT confidence) ----------
    # Qwen wins when both pass; DeepSeek is the fallback secondary.
    best_llm: OCRPageResult | None = None
    winner_llm: str | None = None
    if qw_ok:
        assert qwen_result is not None
        best_llm, winner_llm = qwen_result, "qwen_vl_ocr"
    elif ds_ok:
        assert deepseek_result is not None
        best_llm, winner_llm = deepseek_result, "deepseek_ocr"

    # ---------- empty pool ----------
    if not p_ok and best_llm is None:
        elapsed = time.monotonic() - started
        err_parts = []
        if not p_ok:
            err_parts.append(f"paddleocr={paddle_result.error!r}")
        if deepseek_result and not deepseek_result.succeeded:
            err_parts.append(f"deepseek_ocr={deepseek_result.error!r}")
        elif deepseek_result and not ds_ok:
            err_parts.append(f"deepseek_ocr=quality_gate({ds_gate_reason})")
        if qwen_result and not qwen_result.succeeded:
            err_parts.append(f"qwen_vl_ocr={qwen_result.error!r}")
        elif qwen_result and not qw_ok:
            err_parts.append(f"qwen_vl_ocr=quality_gate({qw_gate_reason})")
        return FusionResult(
            page_id=page_id,
            text_fused="",
            duration_seconds=round(time.monotonic() - started, 3),
            paddle_chars=paddle_chars,
            deepseek_chars=deepseek_chars,
            qwen_chars=qwen_chars,
            fused_chars=0,
            error="all engines failed: " + "; ".join(err_parts),
        )

    # ---------- single-engine fallback ----------
    if (p_ok and best_llm is None) or (not p_ok and best_llm is not None):
        winner = paddle_result if (p_ok and best_llm is None) else best_llm
        single_engine = "paddleocr" if winner is paddle_result else winner_llm
        return FusionResult(
            page_id=page_id,
            text_fused=winner.text,
            char_confidences=[winner.confidence] * len(winner.text),
            segments=[FusionSegment(
                text=winner.text,
                source=single_engine or "paddleocr",
                op="equal",
                a_range=(0, len(winner.text) if winner is paddle_result else 0),
                b_range=(0, len(winner.text) if winner is not paddle_result else 0),
                char_confidence=winner.confidence,
            )],
            agreement_rate=None,
            single_engine=single_engine,
            winner_llm=winner_llm if winner is not paddle_result else None,
            paddle_chars=paddle_chars,
            deepseek_chars=deepseek_chars,
            qwen_chars=qwen_chars,
            fused_chars=len(winner.text),
            duration_seconds=round(time.monotonic() - started, 3),
        )

    # ---------- 2-way alignment: Paddle (anchor) vs best-LLM (corrector) ----------
    assert best_llm is not None
    fused, confidences, segments, agreement_rate = _fuse_two(
        paddle_result, best_llm,
        insert_delete_confidence=insert_delete_confidence,
    )
    return FusionResult(
        page_id=page_id,
        text_fused=fused,
        char_confidences=confidences,
        segments=segments,
        agreement_rate=agreement_rate,
        single_engine=None,
        winner_llm=winner_llm,
        paddle_chars=paddle_chars,
        deepseek_chars=deepseek_chars,
        qwen_chars=qwen_chars,
        fused_chars=len(fused),
        duration_seconds=round(time.monotonic() - started, 3),
    )
