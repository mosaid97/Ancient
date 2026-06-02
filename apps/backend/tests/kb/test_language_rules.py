"""Tests for the language-rules registry (kb/language_rules.py, Track E)."""
from __future__ import annotations

from apps.backend.kb.language_rules import (
    REGISTRY,
    UPLOAD_LANGUAGE_TO_CODE,
    get_language_rule,
    normalize_for_language,
)


# ─────────────────────────────────────────────────────────────────────────────
# Registry shape
# ─────────────────────────────────────────────────────────────────────────────

def test_registry_has_four_languages():
    assert set(REGISTRY) == {"zh", "ja", "en", "ar"}


def test_only_chinese_is_fully_wired():
    assert REGISTRY["zh"].fully_wired is True
    assert REGISTRY["ja"].fully_wired is False
    assert REGISTRY["en"].fully_wired is False
    assert REGISTRY["ar"].fully_wired is False


def test_arabic_is_rtl_others_are_not():
    assert REGISTRY["ar"].rtl is True
    assert REGISTRY["zh"].rtl is False
    assert REGISTRY["en"].rtl is False
    assert REGISTRY["ja"].rtl is False


def test_upload_label_mapping():
    assert UPLOAD_LANGUAGE_TO_CODE == {
        "chinese": "zh", "japanese": "ja", "english": "en", "arabic": "ar",
    }
    for code in UPLOAD_LANGUAGE_TO_CODE.values():
        assert code in REGISTRY


# ─────────────────────────────────────────────────────────────────────────────
# get_language_rule
# ─────────────────────────────────────────────────────────────────────────────

def test_get_language_rule_known_code():
    assert get_language_rule("ar").code == "ar"


def test_get_language_rule_unknown_falls_back_to_chinese():
    rule = get_language_rule("xx-unknown")
    assert rule.code == "zh"


# ─────────────────────────────────────────────────────────────────────────────
# normalize_for_language
# ─────────────────────────────────────────────────────────────────────────────

def test_identity_normalize_collapses_whitespace():
    out = normalize_for_language("hello   world\t\nfoo", "en")
    assert out == "hello world foo"


def test_identity_normalize_empty():
    assert normalize_for_language("", "en") == ""


def test_arabic_uses_identity_stub():
    # Stub keeps text intact apart from whitespace folding.
    out = normalize_for_language("العربية   نص", "ar")
    assert out == "العربية نص"


def test_chinese_normalize_runs_canonical_pipeline():
    # Simplified input should be converted toward traditional (s2t default).
    out = normalize_for_language("汉字", "zh")
    assert out  # non-empty
    # The canonical pipeline produces the traditional form 漢字.
    assert out == "漢字"
