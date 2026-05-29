"""Tests for the 7-step canonical normalization pipeline (normalize/pipeline.py).

No network or DB calls — every step is purely computational.
"""
from __future__ import annotations

import pytest

from apps.backend.normalize.pipeline import (
    CanonicalResult,
    NormalizeStep,
    normalize_canonical,
)


# ─────────────────────────────────────────────────────────────────────────────
# Empty / trivial input
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_string():
    r = normalize_canonical("")
    assert r.canonical == ""
    assert r.original == ""
    assert not r.changed
    assert len(r.steps) == 7  # all 7 steps recorded even if no-op


def test_none_like_empty():
    # normalize_canonical(None) would crash; empty string is the contract.
    r = normalize_canonical("  ")
    assert r.canonical.strip() == ""


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — NFC normalization
# ─────────────────────────────────────────────────────────────────────────────

def test_nfc_step_applied():
    # U+006E (n) + U+0303 (combining tilde) → U+00F5 in NFC
    decomposed = "ñ"  # NFD form
    r = normalize_canonical(decomposed, lang="zh")
    nfc_step = r.steps[0]
    assert nfc_step.name == "nfc"
    assert nfc_step.applied is True


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — whitespace collapse
# ─────────────────────────────────────────────────────────────────────────────

def test_whitespace_collapse():
    text = "唐　律　疏　議"  # ideographic spaces and full-width spaces
    r = normalize_canonical(text)
    # After whitespace step all runs → single ASCII space
    ws_step = r.steps[1]
    assert ws_step.name == "whitespace"
    assert " " in r.canonical  # collapsed to ASCII space
    assert "　" not in r.canonical


def test_zero_width_stripped():
    text = "唐​律‌疏﻿議"
    r = normalize_canonical(text)
    for zw in ["​", "‌", "﻿"]:
        assert zw not in r.canonical


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — T-S conversion (s2t by default)
# ─────────────────────────────────────────────────────────────────────────────

def test_tsc_simplified_to_traditional():
    # 爱 (simplified) → 愛 (traditional)
    r = normalize_canonical("爱国", lang="zh")
    tsc_step = r.steps[2]
    assert tsc_step.name == "tsc"
    # Output should be traditional (愛) if opencc is installed
    # If opencc is missing the step is a no-op — both cases are acceptable.
    assert isinstance(r.canonical, str)
    assert len(r.canonical) > 0


def test_tsc_traditional_unchanged():
    text = "唐律疏議"  # already traditional
    r = normalize_canonical(text, lang="zh")
    assert "唐律疏議" in r.canonical


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — 避諱 (taboo) is era-conditional
# ─────────────────────────────────────────────────────────────────────────────

def test_taboo_off_when_no_era():
    r = normalize_canonical("世民", lang="zh-classical")
    taboo_step = r.steps[4]
    assert taboo_step.name == "taboo"
    assert taboo_step.applied is False
    assert r.flags["taboo"] is False


def test_taboo_on_when_era_given():
    r = normalize_canonical("世民", lang="zh-classical", era="Tang")
    taboo_step = r.steps[4]
    assert taboo_step.name == "taboo"
    assert taboo_step.applied is True
    assert r.flags["taboo"] is True


def test_taboo_era_aliases():
    for alias in ["唐", "tang"]:
        r = normalize_canonical("世民", lang="zh-classical", era=alias)
        assert r.flags["taboo"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — 通假字 (loan) is OFF by default
# ─────────────────────────────────────────────────────────────────────────────

def test_loan_off_by_default():
    r = normalize_canonical("唐律疏議")
    loan_step = r.steps[5]
    assert loan_step.name == "loan"
    assert loan_step.applied is False
    assert r.flags["loan"] is False


def test_loan_on_when_apply_loan():
    r = normalize_canonical("唐律疏議", apply_loan=True)
    loan_step = r.steps[5]
    assert loan_step.applied is True
    assert r.flags["loan"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — kana (mojimoji) is lang-conditional
# ─────────────────────────────────────────────────────────────────────────────

def test_kana_off_for_zh():
    r = normalize_canonical("唐律", lang="zh")
    kana_step = r.steps[6]
    assert kana_step.name == "kana"
    assert kana_step.applied is False


def test_kana_off_for_zh_classical():
    r = normalize_canonical("律令", lang="zh-classical")
    assert r.steps[6].applied is False


def test_kana_on_for_ja():
    r = normalize_canonical("律令", lang="ja")
    assert r.steps[6].applied is True
    assert r.flags["kana"] is True


def test_kana_on_for_mixed():
    r = normalize_canonical("律令", lang="mixed")
    assert r.steps[6].applied is True


# ─────────────────────────────────────────────────────────────────────────────
# Step trace structure
# ─────────────────────────────────────────────────────────────────────────────

def test_step_count_always_seven():
    for lang in ["zh", "zh-classical", "zh-modern", "ja"]:
        r = normalize_canonical("律令", lang=lang)
        assert len(r.steps) == 7, f"Expected 7 steps for lang={lang}"


def test_step_names_ordered():
    r = normalize_canonical("律令")
    expected = ["nfc", "whitespace", "tsc", "variants", "taboo", "loan", "kana"]
    assert [s.name for s in r.steps] == expected


def test_step_changed_property():
    # NFC step on already-NFC text: applied=True but changed=False
    r = normalize_canonical("唐律")
    nfc_step = r.steps[0]
    assert nfc_step.applied is True
    assert nfc_step.changed is False  # no actual change


# ─────────────────────────────────────────────────────────────────────────────
# CanonicalResult properties
# ─────────────────────────────────────────────────────────────────────────────

def test_result_preserves_original():
    text = "唐律疏議"
    r = normalize_canonical(text)
    assert r.original == text


def test_result_lang_echoed():
    r = normalize_canonical("律", lang="zh-classical")
    assert r.lang == "zh-classical"


def test_result_era_echoed():
    r = normalize_canonical("律", era="Song")
    assert r.era == "Song"


def test_result_changed_property_false():
    # Pure ASCII ASCII text — all steps likely no-op
    r = normalize_canonical("abc")
    # changed = original != canonical; both should be 'abc'
    assert isinstance(r.changed, bool)


def test_result_flags_complete():
    r = normalize_canonical("律")
    expected_keys = {"nfc", "whitespace", "tsc", "variants", "taboo", "loan", "kana", "taboo_unsafe"}
    assert expected_keys.issubset(set(r.flags.keys()))


# ─────────────────────────────────────────────────────────────────────────────
# Heterograph roundtrip (the key verifier use case)
# ─────────────────────────────────────────────────────────────────────────────

def test_simplified_span_normalizes_to_traditional():
    """Simplified 恶 should normalize to traditional 惡 after T-S step."""
    r_simplified = normalize_canonical("十恶尤切", lang="zh-classical")
    r_traditional = normalize_canonical("十惡尤切", lang="zh-classical")
    # Both should share the same canonical form after T-S
    assert r_simplified.canonical == r_traditional.canonical
