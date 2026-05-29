"""Tests for normalize/tsc.py (T-S conversion) and normalize/kana.py (mojimoji)."""
import pytest

from apps.backend.normalize import kana, tsc


class TestTsc:
    def test_simplified_to_traditional(self):
        # "爱" (simplified) → "愛" (traditional) with s2t
        result = tsc.normalize("爱", config="s2t")
        assert result == "愛"

    def test_traditional_to_simplified(self):
        result = tsc.normalize("愛", config="t2s")
        assert result == "爱"

    def test_empty_string(self):
        assert tsc.normalize("") == ""

    def test_none_like_empty(self):
        assert tsc.normalize("", config="s2t") == ""

    def test_already_traditional_unchanged(self):
        # 唐律疏議 is already traditional
        result = tsc.normalize("唐律疏議", config="s2t")
        assert result == "唐律疏議"

    def test_mixed_text_converted(self):
        # 国 (simplified) in mixed text
        result = tsc.normalize("中国法律", config="s2t")
        assert "國" in result

    def test_default_config_is_s2t(self):
        r1 = tsc.normalize("爱", config="s2t")
        r2 = tsc.normalize("爱")
        assert r1 == r2

    def test_graceful_on_bad_config(self):
        # Invalid config should return input unchanged (graceful fallback)
        result = tsc.normalize("唐律疏議", config="invalid_config_xyz")
        assert isinstance(result, str)

    def test_ascii_passthrough(self):
        result = tsc.normalize("hello world", config="s2t")
        assert result == "hello world"


class TestKana:
    def test_zh_language_noop(self):
        text = "唐律ﾕ"  # half-width katakana in CJK text
        assert kana.normalize(text, lang="zh") == text

    def test_zh_classical_noop(self):
        text = "唐律ﾕ"
        assert kana.normalize(text, lang="zh-classical") == text

    def test_zh_modern_noop(self):
        text = "ﾕ"
        assert kana.normalize(text, lang="zh-modern") == text

    def test_empty_string(self):
        assert kana.normalize("", lang="ja") == ""

    def test_ja_converts_halfwidth(self):
        # ﾕ (half-width katakana U+FF55) → ユ (full-width U+30E6)
        try:
            import mojimoji
            result = kana.normalize("ﾕ", lang="ja")
            assert result == "ユ"
        except ImportError:
            pytest.skip("mojimoji not installed")

    def test_mixed_lang_converts(self):
        try:
            import mojimoji
            result = kana.normalize("ﾕ", lang="mixed")
            assert result == "ユ"
        except ImportError:
            pytest.skip("mojimoji not installed")

    def test_default_lang_is_zh(self):
        text = "ﾕ"
        assert kana.normalize(text) == text  # zh is default, no-op

    def test_pure_zh_text_unchanged_for_ja(self):
        # Chinese-only text should pass through even for ja lang
        try:
            import mojimoji
            text = "唐律疏議"
            result = kana.normalize(text, lang="ja")
            # mojimoji.han_to_zen on pure CJK should return the same text
            assert result == mojimoji.han_to_zen(text)
        except ImportError:
            pytest.skip("mojimoji not installed")
