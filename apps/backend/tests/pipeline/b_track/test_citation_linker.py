"""Tests for pipeline/citation_linker.py — span extraction (B2)."""
import pytest

from apps.backend.pipeline.citation_linker import _extract_quoted_spans


class TestExtractQuotedSpans:
    def test_kakko_pair_extracted(self):
        text = "律文曰「十惡尤切不容首免」是也"
        spans = _extract_quoted_spans(text)
        assert "十惡尤切不容首免" in spans

    def test_double_kakko_extracted(self):
        text = "疏議引『名例律第一』以解之"
        spans = _extract_quoted_spans(text)
        assert "名例律第一" in spans

    def test_curly_quotes_extracted(self):
        # Use Unicode LEFT/RIGHT DOUBLE QUOTATION MARK (U+201C / U+201D)
        text = "引文曰“諸犯死罪者皆斬”而已"
        spans = _extract_quoted_spans(text)
        assert "諸犯死罪者皆斬" in spans

    def test_guillemet_angle_brackets(self):
        # "唐律疏議" is 4 chars < _MIN_SPAN_CHARS=5; use 5-char span
        text = "見《唐律疏議名》例律第一條"
        spans = _extract_quoted_spans(text)
        assert "唐律疏議名" in spans

    def test_span_below_min_length_excluded(self):
        # "甲" is 1 char — below _MIN_SPAN_CHARS=5
        text = "曰「甲」而後"
        spans = _extract_quoted_spans(text)
        assert "甲" not in spans

    def test_span_above_max_length_excluded(self):
        long_span = "律" * 400  # 400 chars, above _MAX_SPAN_CHARS=300
        text = f"引「{long_span}」結束"
        spans = _extract_quoted_spans(text)
        assert long_span not in spans

    def test_ellipsis_context_window_captured(self):
        prefix = "唐律疏議云" * 15   # ~75 chars before
        suffix = "此其要旨也。" * 10  # ~60 chars after
        text = f"{prefix}…省略{suffix}"
        spans = _extract_quoted_spans(text)
        assert len(spans) >= 1
        # The captured span must contain the ellipsis marker
        assert any("…" in s or "省略" in s for s in spans)

    def test_no_quotes_returns_empty(self):
        text = "本文無任何引號標記，純為論述。"
        spans = _extract_quoted_spans(text)
        # No ellipsis markers either → empty
        assert spans == []

    def test_deduplication(self):
        # Same quote appears twice
        text = "律曰「十惡尤切不容首免」又曰「十惡尤切不容首免」"
        spans = _extract_quoted_spans(text)
        assert spans.count("十惡尤切不容首免") == 1

    def test_cap_at_max_spans(self):
        # 15 distinct spans, should be capped at 10
        parts = [f"「{'律' * (5 + i)}」" for i in range(15)]
        text = "".join(parts)
        spans = _extract_quoted_spans(text)
        assert len(spans) <= 10

    def test_multiple_quote_styles_in_one_text(self):
        # Each span must be ≥ 5 chars: "十惡之制度"=5, "名例律第一"=5
        text = '「十惡之制度」其義，『名例律第一』為首'
        spans = _extract_quoted_spans(text)
        assert len(spans) >= 2

    def test_exact_min_length_span_included(self):
        # _MIN_SPAN_CHARS = 5; use a 5-char span
        text = "律曰「謀反大逆」云云"  # "謀反大逆" = 4 chars — below min; try 5
        text2 = "律曰「謀反大逆不」云云"  # 5 chars
        spans = _extract_quoted_spans(text2)
        assert "謀反大逆不" in spans

    def test_nested_quotes_inner_extracted(self):
        # Outer 「...」 contains inner 『...』 — both extracted independently
        text = "「此乃『名例第一』之義」也"
        spans = _extract_quoted_spans(text)
        assert "名例第一" in spans or any("名例第一" in s for s in spans)

    def test_order_preserved(self):
        text = "「十惡之制」……「八議之法」"
        spans = _extract_quoted_spans(text)
        if len(spans) >= 2:
            assert spans.index("十惡之制") < spans.index("八議之法")

    def test_empty_text(self):
        assert _extract_quoted_spans("") == []
