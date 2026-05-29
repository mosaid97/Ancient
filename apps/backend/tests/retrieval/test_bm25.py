"""Tests for retrieval/bm25.py — tokenizer and BM25 index."""
import pytest

from apps.backend.retrieval.bm25 import _ngrams, _tokenize


class TestNgrams:
    def test_trigram_count(self):
        text = "唐律疏議"   # 4 chars → 2 trigrams
        result = _ngrams(text, sizes=(3,))
        assert result == ["唐律疏", "律疏議"]

    def test_fourgram_count(self):
        text = "唐律疏議名"  # 5 chars → 2 four-grams
        result = _ngrams(text, sizes=(4,))
        assert result == ["唐律疏議", "律疏議名"]

    def test_both_sizes_combined(self):
        text = "唐律疏議"   # 4 chars: 2 trigrams + 1 four-gram = 3 tokens
        result = _ngrams(text)
        assert len(result) == 3

    def test_short_text_no_tokens(self):
        # 2-char text → no 3-gram, no 4-gram
        assert _ngrams("唐律") == []

    def test_empty_string(self):
        assert _ngrams("") == []

    def test_exact_three_chars(self):
        result = _ngrams("唐律疏", sizes=(3,))
        assert result == ["唐律疏"]

    def test_overlapping_windows(self):
        result = _ngrams("abcde", sizes=(3,))
        assert result == ["abc", "bcd", "cde"]


class TestTokenize:
    def test_strips_whitespace(self):
        # Whitespace collapsed, then n-grams applied
        result = _tokenize("唐律 疏議")
        # After stripping: "唐律疏議" (4 chars) → 2 tri + 1 quad = 3 tokens
        assert len(result) == 3

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_whitespace_only(self):
        assert _tokenize("   \t\n") == []

    def test_classical_chinese_passage(self):
        text = "諸犯死罪者"  # 5 chars → 3 tri + 2 quad = 5 tokens
        result = _tokenize(text)
        assert len(result) == 5

    def test_contains_specific_trigrams(self):
        result = _tokenize("唐律疏議")
        assert "唐律疏" in result
        assert "律疏議" in result

    def test_contains_specific_fourgrams(self):
        result = _tokenize("唐律疏議名")
        assert "唐律疏議" in result

    def test_multiline_joined(self):
        result = _tokenize("唐律\n疏議")
        assert "唐律疏" in result

    def test_mixed_ascii_cjk_stripped(self):
        # ASCII spaces stripped, CJK chars remain
        result = _tokenize("唐 律 疏 議")
        assert "唐律疏" in result
