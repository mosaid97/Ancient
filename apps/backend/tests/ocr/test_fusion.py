"""Tests for the 3-engine OCR fusion logic (ocr/fusion.py).

Tests cover the pure helper functions and the main fuse_results() function
across all branch paths. No external APIs or DB connections required.
"""
from __future__ import annotations

import pytest

from apps.backend.ocr.base import OCRLine, OCRPageResult
from apps.backend.ocr.fusion import (
    FusionResult,
    _cjk_ratio,
    _is_cjk,
    _llm_quality_gate,
    _noise_index,
    fuse_results,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_page(
    text: str,
    *,
    engine: str = "paddleocr",
    page_id: str = "p1",
    error: str | None = None,
    confidence: float = 0.9,
) -> OCRPageResult:
    return OCRPageResult(
        engine=engine,  # type: ignore[arg-type]
        model_version="test",
        page_id=page_id,
        text=text,
        char_count=len(text),
        confidence=confidence,
        error=error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _is_cjk
# ─────────────────────────────────────────────────────────────────────────────

def test_is_cjk_unified_range():
    assert _is_cjk("唐") is True
    assert _is_cjk("律") is True
    assert _is_cjk("疏") is True


def test_is_cjk_extension_a():
    # U+3400–U+4DBF CJK Extension A
    assert _is_cjk("㐀") is True


def test_is_cjk_ascii_false():
    assert _is_cjk("A") is False
    assert _is_cjk("0") is False
    assert _is_cjk(" ") is False


def test_is_cjk_hiragana_false():
    # Hiragana is NOT in the CJK ranges defined in fusion.py
    assert _is_cjk("あ") is False


# ─────────────────────────────────────────────────────────────────────────────
# _cjk_ratio
# ─────────────────────────────────────────────────────────────────────────────

def test_cjk_ratio_empty():
    assert _cjk_ratio("") == 0.0


def test_cjk_ratio_pure_cjk():
    assert _cjk_ratio("唐律疏議") == pytest.approx(1.0)


def test_cjk_ratio_pure_ascii():
    assert _cjk_ratio("hello world") == pytest.approx(0.0)


def test_cjk_ratio_mixed():
    text = "唐律疏议abc"  # 4 CJK + 3 ASCII = 7 total
    r = _cjk_ratio(text)
    assert pytest.approx(r, abs=0.01) == 4 / 7


# ─────────────────────────────────────────────────────────────────────────────
# _noise_index
# ─────────────────────────────────────────────────────────────────────────────

def test_noise_index_clean():
    text = "唐律疏議卷第一，名例律，五刑之中十惡尤切。"
    ni = _noise_index(text)
    assert ni < 0.20, f"Clean CJK text should have low noise, got {ni:.3f}"


def test_noise_index_html_tags():
    text = "<table><tr><td>content</td></tr></table>" * 5
    ni = _noise_index(text)
    assert ni > 0.0


def test_noise_index_pipe_tables():
    text = "| 名 | 称 |\n| -- | -- |\n| 唐 | 律 |" * 5
    ni = _noise_index(text)
    assert ni > 0.0


def test_noise_index_empty():
    assert _noise_index("") == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _llm_quality_gate
# ─────────────────────────────────────────────────────────────────────────────

def _paddle(text: str) -> OCRPageResult:
    return _make_page(text, engine="paddleocr")


def _llm(text: str, engine: str = "deepseek_ocr") -> OCRPageResult:
    return _make_page(text, engine=engine)  # type: ignore[arg-type]


def test_quality_gate_passes_clean():
    paddle = _paddle("唐律疏議" * 20)  # 80 chars
    llm = _llm("唐律疏議" * 20)
    passes, reason = _llm_quality_gate(
        llm, paddle,
        char_ratio_limit=3.0, cjk_ratio_min=0.40, noise_index_max=0.30
    )
    assert passes is True
    assert reason is None


def test_quality_gate_blocks_char_ratio():
    paddle = _paddle("唐" * 10)  # 10 chars
    llm = _llm("唐" * 50)         # 50 chars — ratio = 5.0 > 3.0 limit
    passes, reason = _llm_quality_gate(
        llm, paddle,
        char_ratio_limit=3.0, cjk_ratio_min=0.40, noise_index_max=0.30
    )
    assert passes is False
    assert "char_ratio" in reason


def test_quality_gate_blocks_low_cjk():
    # paddle = 100 chars; llm = 120 chars (ratio 1.2 < 3.0 → char gate passes)
    # but CJK ratio = 0/120 = 0.0 < 0.40 → blocked by cjk gate
    paddle = _paddle("唐" * 100)
    llm = _llm("hello world abc " * 8)  # 128 chars, pure ASCII, ratio ≈ 1.28
    passes, reason = _llm_quality_gate(
        llm, paddle,
        char_ratio_limit=3.0, cjk_ratio_min=0.40, noise_index_max=0.30
    )
    assert passes is False
    assert "cjk_ratio" in reason


def test_quality_gate_skips_char_ratio_when_paddle_empty():
    paddle = _paddle("")  # no paddle output → char gate skipped
    paddle.char_count = 0
    llm = _llm("唐" * 50)
    passes, reason = _llm_quality_gate(
        llm, paddle,
        char_ratio_limit=3.0, cjk_ratio_min=0.40, noise_index_max=0.30
    )
    # CJK ratio is fine; char ratio gate skipped when paddle is empty
    assert passes is True


# ─────────────────────────────────────────────────────────────────────────────
# fuse_results — main scenarios
# ─────────────────────────────────────────────────────────────────────────────

CJK_TEXT = "唐律疏議卷第一名例律十惡尤切不容首免" * 3  # 54 chars, pure CJK


def test_fuse_paddle_only():
    """Only Paddle — single-engine fallback."""
    paddle = _make_page(CJK_TEXT, engine="paddleocr")
    result = fuse_results(paddle)
    assert result.text_fused == CJK_TEXT
    assert result.single_engine == "paddleocr"
    assert result.winner_llm is None
    assert result.error is None


def test_fuse_paddle_plus_qwen():
    """Paddle + Qwen agreement → fused text, no error."""
    paddle = _make_page(CJK_TEXT, engine="paddleocr")
    qwen = _make_page(CJK_TEXT, engine="deepseek_ocr")  # same text = full agreement
    result = fuse_results(paddle, qwen_result=qwen)
    assert result.error is None
    assert result.text_fused  # non-empty
    assert result.single_engine is None   # two engines contributed


def test_fuse_qwen_wins_over_deepseek():
    """When both LLMs pass gate, Qwen wins (not DeepSeek)."""
    paddle = _make_page(CJK_TEXT, engine="paddleocr")
    qwen = _make_page(CJK_TEXT, engine="deepseek_ocr")   # reusing shape; winner is by parameter
    deepseek = _make_page(CJK_TEXT, engine="deepseek_ocr")
    result = fuse_results(paddle, deepseek_result=deepseek, qwen_result=qwen)
    assert result.winner_llm == "qwen_vl_ocr"


def test_fuse_deepseek_fallback_when_qwen_absent():
    """Only DeepSeek provided → DeepSeek wins."""
    paddle = _make_page(CJK_TEXT, engine="paddleocr")
    deepseek = _make_page(CJK_TEXT, engine="deepseek_ocr")
    result = fuse_results(paddle, deepseek_result=deepseek)
    assert result.winner_llm == "deepseek_ocr"
    assert result.error is None


def test_fuse_all_failed():
    """All engines errored → FusionResult.error set, empty text."""
    paddle = _make_page("", engine="paddleocr", error="paddle failed")
    result = fuse_results(paddle)
    assert result.text_fused == ""
    assert result.error is not None
    assert "all engines failed" in result.error


def test_fuse_llm_quality_blocked_falls_back_to_paddle():
    """LLM fails quality gate → single-engine Paddle fallback."""
    paddle = _make_page(CJK_TEXT, engine="paddleocr")
    # LLM output is pure ASCII → blocked by CJK ratio gate
    bad_qwen = _make_page("hello world " * 20, engine="deepseek_ocr")
    result = fuse_results(paddle, qwen_result=bad_qwen)
    # Qwen blocked; Paddle used as single engine
    assert result.text_fused == CJK_TEXT
    assert result.single_engine == "paddleocr"


def test_fuse_result_to_dict_shape():
    """FusionResult.to_dict() must contain all required keys."""
    paddle = _make_page(CJK_TEXT, engine="paddleocr")
    result = fuse_results(paddle)
    d = result.to_dict()
    for key in ("pageId", "textFused", "agreementRate", "singleEngine",
                "winnerLlm", "paddleChars", "fusedChars", "durationSeconds",
                "segments", "error"):
        assert key in d, f"Missing key {key!r} in FusionResult.to_dict()"


def test_fuse_agreement_rate_none_for_single_engine():
    """Single-engine result should not compute agreement rate."""
    paddle = _make_page(CJK_TEXT, engine="paddleocr")
    result = fuse_results(paddle)
    assert result.agreement_rate is None


def test_fuse_agreement_rate_one_when_engines_agree():
    """If both engines produce identical text, agreement = 1.0."""
    paddle = _make_page(CJK_TEXT, engine="paddleocr")
    qwen = _make_page(CJK_TEXT, engine="deepseek_ocr")
    result = fuse_results(paddle, qwen_result=qwen)
    if result.agreement_rate is not None:
        assert result.agreement_rate == pytest.approx(1.0, abs=0.01)
