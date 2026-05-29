"""Tests for pure helper functions in pipeline/chunk.py.

No Neo4j connection required — tests cover the deterministic text-splitting
and page-text-selection logic that must be correct for downstream embedding.
"""
from __future__ import annotations

import pytest

from apps.backend.pipeline.chunk import (
    _make_chunk_id,
    _markdown_sections,
    _sliding_window,
    resolve_page_text,
)


# ─────────────────────────────────────────────────────────────────────────────
# _sliding_window
# ─────────────────────────────────────────────────────────────────────────────

def test_sliding_window_empty():
    assert _sliding_window("", chunk_size=500, overlap=50) == []


def test_sliding_window_fits_in_one():
    text = "唐律疏議"
    result = _sliding_window(text, chunk_size=500, overlap=50)
    assert result == [text]


def test_sliding_window_exact_fit():
    text = "x" * 500
    result = _sliding_window(text, chunk_size=500, overlap=50)
    assert result == [text]


def test_sliding_window_two_chunks():
    text = "a" * 600
    result = _sliding_window(text, chunk_size=500, overlap=50)
    assert len(result) == 2
    assert result[0] == "a" * 500
    assert result[1] == "a" * 150  # 600 - (500 - 50) = 150


def test_sliding_window_overlap_content():
    # Overlap means the tail of chunk N == head of chunk N+1
    text = "ABCDE" * 200  # 1000 chars
    result = _sliding_window(text, chunk_size=500, overlap=100)
    # Last 100 chars of chunk 0 == first 100 chars of chunk 1
    assert result[0][-100:] == result[1][:100]


def test_sliding_window_step_math():
    text = "x" * 1000
    result = _sliding_window(text, chunk_size=100, overlap=20)
    # step = 80; expected chunks = ceil((1000 - 100) / 80) + 1 ≈ 13
    for chunk in result:
        assert len(chunk) <= 100


def test_sliding_window_no_duplication_past_end():
    text = "x" * 550
    result = _sliding_window(text, chunk_size=500, overlap=50)
    assert len(result) == 2
    # Second chunk must not exceed text length
    assert len(result[1]) <= 100


# ─────────────────────────────────────────────────────────────────────────────
# _markdown_sections
# ─────────────────────────────────────────────────────────────────────────────

def test_markdown_no_headings_falls_back():
    text = "x" * 1000
    md = _markdown_sections(text, chunk_size=500, overlap=50)
    sw = _sliding_window(text, chunk_size=500, overlap=50)
    assert md == sw


def test_markdown_single_heading():
    text = "# 名例律\n十惡尤切，不容首免。"
    result = _markdown_sections(text, chunk_size=500, overlap=50)
    assert len(result) == 1
    assert "名例律" in result[0]
    assert "十惡" in result[0]


def test_markdown_two_headings_merged_when_small():
    text = "# 第一章\n唐律。\n# 第二章\n疏議。"
    result = _markdown_sections(text, chunk_size=500, overlap=50)
    # Both sections are tiny — should merge into one chunk
    assert len(result) == 1
    assert "第一章" in result[0]
    assert "第二章" in result[0]


def test_markdown_large_section_split():
    # Section content too large → overflow into sliding_window
    section_text = "x" * 600
    text = f"# 大章\n{section_text}"
    result = _markdown_sections(text, chunk_size=500, overlap=50)
    # Must split the large section
    assert len(result) >= 2


def test_markdown_preamble_before_first_heading():
    text = "前言内容\n# 第一章\n正文内容"
    result = _markdown_sections(text, chunk_size=500, overlap=50)
    combined = "".join(result)
    assert "前言" in combined
    assert "正文" in combined


def test_markdown_filters_short_chunks():
    # Very short sections below _MIN_CHUNK_CHARS (10) must be filtered
    text = "# A\n短\n# B\n这是一段较长的正文内容，超过了最小字符数要求"
    result = _markdown_sections(text, chunk_size=500, overlap=50)
    for chunk in result:
        assert len(chunk) >= 10


# ─────────────────────────────────────────────────────────────────────────────
# resolve_page_text
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_prefers_structured_markdown():
    row = {
        "structuredMarkdown": "# 头\n内容",
        "layoutStatus": "ok",
        "textFused": "fused text",
        "fusionStatus": "ok",
        "text": "native text",
    }
    text, strategy = resolve_page_text(row)
    assert text == "# 头\n内容"
    assert strategy == "markdown_section"


def test_resolve_markdown_skipped_when_layout_not_ok():
    row = {
        "structuredMarkdown": "# 头\n内容",
        "layoutStatus": "pending",
        "textFused": "fused text",
        "fusionStatus": "ok",
        "text": "native text",
    }
    text, strategy = resolve_page_text(row)
    assert text == "fused text"
    assert strategy == "sliding_window"


def test_resolve_fused_preferred_over_native():
    row = {
        "structuredMarkdown": None,
        "layoutStatus": None,
        "textFused": "fused text",
        "fusionStatus": "ok",
        "text": "native text",
    }
    text, strategy = resolve_page_text(row)
    assert text == "fused text"
    assert strategy == "sliding_window"


def test_resolve_fused_single_status():
    row = {
        "textFused": "only one engine",
        "fusionStatus": "single",
        "text": "native",
    }
    text, strategy = resolve_page_text(row)
    assert text == "only one engine"


def test_resolve_fused_skipped_when_status_not_ok():
    row = {
        "textFused": "fused text",
        "fusionStatus": "pending",
        "text": "native text",
    }
    text, strategy = resolve_page_text(row)
    assert text == "native text"
    assert strategy == "sliding_window"


def test_resolve_native_fallback():
    row = {
        "structuredMarkdown": None,
        "textFused": None,
        "text": "native text",
    }
    text, strategy = resolve_page_text(row)
    assert text == "native text"
    assert strategy == "sliding_window"


def test_resolve_all_null_returns_none():
    row = {
        "structuredMarkdown": None,
        "textFused": None,
        "text": None,
    }
    text, strategy = resolve_page_text(row)
    assert text is None
    assert strategy == "sliding_window"


# ─────────────────────────────────────────────────────────────────────────────
# _make_chunk_id
# ─────────────────────────────────────────────────────────────────────────────

def test_make_chunk_id_format():
    cid = _make_chunk_id("page_001", 0)
    assert cid == "page_001::chunk_0000"


def test_make_chunk_id_zero_padded():
    assert _make_chunk_id("p", 42) == "p::chunk_0042"
    assert _make_chunk_id("p", 9999) == "p::chunk_9999"


def test_make_chunk_id_unique_per_index():
    ids = {_make_chunk_id("page_x", i) for i in range(100)}
    assert len(ids) == 100
