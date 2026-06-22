"""Tests for the deterministic zero-hallucination verifier (agents/verifier.py).

Covers plan A3 requirements:
  1. Faithful span → outcome='ok'
  2. 異體字-mutated span that only matches AFTER normalization → outcome='ok'
  3. Fabricated (absent) span → outcome='insufficient_evidence' + VERIFIER_FAILURE node

All Neo4j calls are mocked; no real DB connection required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from apps.backend.agents.verifier import (
    MIN_SPAN_CHARS,
    VerifierFailureMode,
    VerifierResult,
    _evidence_strength,
    extract_quotable_span,
    verify_cite,
)


def test_extract_quotable_span_picks_longest_han_run():
    assert extract_quotable_span("唐代律法") == "唐代律法"
    # Mixed natural-language query — pick the longest Han run.
    assert extract_quotable_span("What does 十惡尤切 mean in 唐律?") == "十惡尤切"


def test_extract_quotable_span_returns_none_for_short_runs():
    # Below MIN_SPAN_CHARS: no quotable span.
    assert extract_quotable_span("唐律") is None
    # No Han chars at all.
    assert extract_quotable_span("what is the law?") is None
    # Empty input.
    assert extract_quotable_span("") is None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_driver(row: dict | None) -> MagicMock:
    """Return a mocked Neo4j driver that returns a single row or empty."""
    session_mock = MagicMock()
    run_mock = MagicMock()
    run_mock.data.return_value = [row] if row is not None else []
    run_mock.consume.return_value = None
    session_mock.run.return_value = run_mock
    session_mock.__enter__ = lambda s: s
    session_mock.__exit__ = MagicMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = session_mock
    return driver


# ─────────────────────────────────────────────────────────────────────────────
# Evidence-strength badge tests
# ─────────────────────────────────────────────────────────────────────────────

def test_badge_primary_pure_source():
    assert _evidence_strength("primary", "pure-source") == "primary_source"


def test_badge_primary_shuyi():
    assert _evidence_strength("primary", "疏議") == "primary_疏議"


def test_badge_primary_jiaokan():
    assert _evidence_strength("primary", "校點") == "editorial_commentary"
    assert _evidence_strength("primary", "校訂") == "editorial_commentary"
    assert _evidence_strength("primary", "箋解") == "editorial_commentary"


def test_badge_secondary():
    assert _evidence_strength("secondary", None) == "scholarly_interpretation"
    assert _evidence_strength("secondary", "pure-source") == "scholarly_interpretation"


def test_badge_primary_unknown_layer():
    # Unlisted layer → treat as primary_source (most conservative)
    assert _evidence_strength("primary", None) == "primary_source"


# ─────────────────────────────────────────────────────────────────────────────
# Empty / missing span / chunk
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_span_no_db_call():
    """Empty span must fail immediately without hitting Neo4j."""
    driver = MagicMock()
    result = verify_cite(driver, "c1", "")
    driver.session.assert_not_called()
    assert result.outcome == "insufficient_evidence"
    assert result.failure_mode == VerifierFailureMode.NO_TEXT


def test_whitespace_only_span_no_db_call():
    driver = MagicMock()
    result = verify_cite(driver, "c1", "   ")
    driver.session.assert_not_called()
    assert result.outcome == "insufficient_evidence"


def test_chunk_not_found():
    """Missing chunk → insufficient_evidence + VERIFIER_FAILURE node written."""
    driver = _make_driver(None)
    # Use a ≥ MIN_SPAN_CHARS span so we exercise the chunk-not-found path
    # rather than the short-span guard.
    result = verify_cite(driver, "missing_chunk", "唐律疏議")
    assert result.outcome == "insufficient_evidence"
    assert result.failure_mode == VerifierFailureMode.CHUNK_NOT_FOUND
    # VERIFIER_FAILURE write should be attempted (second session.run call)
    assert driver.session.call_count >= 2


def test_chunk_with_no_text():
    """Chunk exists but has no text → insufficient_evidence."""
    row = {
        "chunk_id": "c1",
        "text_canonical": None,
        "text_vernacular": None,
        "text_raw": None,
        "tier": "primary",
        "language": "zh-classical",
        "editorial_layer_type": "pure-source",
        "detected_era": None,
    }
    driver = _make_driver(row)
    result = verify_cite(driver, "c1", "唐律疏議")
    assert result.outcome == "insufficient_evidence"
    assert result.failure_mode == VerifierFailureMode.NO_TEXT


def test_short_span_rejected():
    """Spans shorter than MIN_SPAN_CHARS (4) are rejected as too short
    — trivial matches like 之/曰 do not earn the verified badge."""
    driver = _make_driver(None)
    result = verify_cite(driver, "any_chunk", "之曰")
    # Must not reach the chunk-fetch session call (guard runs first).
    driver.session.assert_called()  # only the failure-node write
    assert result.outcome == "insufficient_evidence"
    assert result.failure_mode == VerifierFailureMode.SPAN_TOO_SHORT


# ─────────────────────────────────────────────────────────────────────────────
# Core verification logic
# ─────────────────────────────────────────────────────────────────────────────

def test_faithful_span_matches_canonical():
    """Case 1: span exists verbatim in textCanonical → outcome='ok'."""
    row = {
        "chunk_id": "c1",
        "text_canonical": "唐律疏議卷第一，名例律，五刑之中十惡尤切",
        "text_vernacular": None,
        "text_raw": "唐律疏議卷第一",
        "tier": "primary",
        "language": "zh-classical",
        "editorial_layer_type": "pure-source",
        "detected_era": None,
    }
    driver = _make_driver(row)
    result = verify_cite(driver, "c1", "十惡尤切")
    assert result.ok
    assert result.outcome == "ok"
    assert result.matched_in == "canonical"
    assert result.evidence_strength == "primary_source"
    assert result.tier == "primary"


def test_vernacular_match_is_translation_only_not_verified():
    """Span only present in textVernacular (LLM translation) must NOT
    earn the verified badge — outcome='translation_match', ok=False."""
    row = {
        "chunk_id": "c2",
        "text_canonical": "十惡尤切",
        "text_vernacular": "十种最严重的罪行中，尤为严重",
        "text_raw": "十惡尤切",
        "tier": "primary",
        "language": "zh-classical",
        "editorial_layer_type": "pure-source",
        "detected_era": None,
    }
    driver = _make_driver(row)
    result = verify_cite(driver, "c2", "最严重的罪行")
    assert result.outcome == "translation_match"
    assert result.translation_match is True
    assert result.ok is False
    assert result.matched_in == "vernacular"
    assert result.evidence_strength == "translation_match"


def test_span_only_in_raw_fallback():
    """When textCanonical is NULL, falls back to text_raw for matching."""
    row = {
        "chunk_id": "c3",
        "text_canonical": None,
        "text_vernacular": None,
        "text_raw": "唐律疏議記載十惡之罪",
        "tier": "primary",
        "language": "zh-classical",
        "editorial_layer_type": None,
        "detected_era": None,
    }
    driver = _make_driver(row)
    result = verify_cite(driver, "c3", "十惡之罪")
    assert result.ok
    assert result.matched_in == "canonical"  # raw fed into canonical path


def test_heterograph_span_matches_after_normalization():
    """Case 2: 異體字 variant span matches only after normalization.

    Example: simplified 恶 vs traditional 惡 — normalization (T-S step)
    converts simplified to traditional before the substring check.
    """
    row = {
        "chunk_id": "c4",
        # canonical form uses traditional character 惡
        "text_canonical": "十惡尤切，不容首免",
        "text_vernacular": None,
        "text_raw": "十惡尤切",
        "tier": "primary",
        "language": "zh-classical",
        "editorial_layer_type": "pure-source",
        "detected_era": None,
    }
    driver = _make_driver(row)
    # Span uses simplified 恶 — normalization converts 恶→惡 via T-S step
    result = verify_cite(driver, "c4", "十恶尤切")
    assert result.ok, (
        f"Expected ok after normalization, got {result.outcome!r} "
        f"(normalized_span={result.normalized_span!r})"
    )
    assert result.matched_in == "canonical"


def test_fabricated_span_fails():
    """Case 3: span not in chunk text → insufficient_evidence + VERIFIER_FAILURE node."""
    row = {
        "chunk_id": "c5",
        "text_canonical": "唐律疏議卷第一，名例律",
        "text_vernacular": "唐朝法律体系卷一，基本法则",
        "text_raw": "唐律疏議卷第一",
        "tier": "primary",
        "language": "zh-classical",
        "editorial_layer_type": "pure-source",
        "detected_era": None,
    }
    driver = _make_driver(row)
    result = verify_cite(driver, "c5", "皇帝御批诛九族")  # completely fabricated
    assert result.outcome == "insufficient_evidence"
    assert result.failure_mode == VerifierFailureMode.SPAN_NOT_FOUND
    # VERIFIER_FAILURE node should be written (multiple session calls)
    assert driver.session.call_count >= 2


def test_fabricated_span_writes_failure_node():
    """VERIFIER_FAILURE Cypher must contain the correct properties."""
    row = {
        "chunk_id": "c6",
        "text_canonical": "名例律第一",
        "text_vernacular": None,
        "text_raw": "名例律第一",
        "tier": "secondary",
        "language": "zh-modern",
        "editorial_layer_type": None,
        "detected_era": None,
    }
    written_queries: list[str] = []
    written_params: list[dict] = []

    # Track every session.run call
    calls_log: list[tuple[str, dict]] = []

    session_mock = MagicMock()

    def _run(query, **params):
        calls_log.append((query, params))
        mock_result = MagicMock()
        # First call (CHUNK fetch) returns our row
        if "MATCH (c:CHUNK" in query:
            mock_result.data.return_value = [row]
        else:
            mock_result.data.return_value = []
        mock_result.consume.return_value = None
        return mock_result

    session_mock.run.side_effect = _run
    session_mock.__enter__ = lambda s: s
    session_mock.__exit__ = MagicMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = session_mock

    result = verify_cite(driver, "c6", "完全捏造的内容")
    assert result.outcome == "insufficient_evidence"

    # Find the VERIFIER_FAILURE write call
    failure_calls = [(q, p) for q, p in calls_log if "VERIFIER_FAILURE" in q]
    assert failure_calls, "Expected a VERIFIER_FAILURE MERGE query to be executed"
    _, params = failure_calls[0]
    assert params.get("chunk_id") == "c6"
    assert params.get("failure_mode") == "span_not_found"
    assert params.get("tier") == "secondary"


def test_secondary_tier_badge():
    """Secondary tier always gets scholarly_interpretation badge."""
    row = {
        "chunk_id": "c7",
        "text_canonical": "察举制度是两汉选官的主要方式",
        "text_vernacular": None,
        "text_raw": "察举制度是两汉选官的主要方式",
        "tier": "secondary",
        "language": "zh-modern",
        "editorial_layer_type": None,
        "detected_era": None,
    }
    driver = _make_driver(row)
    result = verify_cite(driver, "c7", "察举制度")
    assert result.ok
    assert result.evidence_strength == "scholarly_interpretation"


def test_loose_mode_enables_loan_chars():
    """loose=True must pass apply_loan=True to normalize_canonical."""
    row = {
        "chunk_id": "c8",
        "text_canonical": "知己知彼",
        "text_vernacular": None,
        "text_raw": "知己知彼",
        "tier": "primary",
        "language": "zh-classical",
        "editorial_layer_type": "pure-source",
        "detected_era": None,
    }
    driver = _make_driver(row)
    with patch("apps.backend.agents.verifier.normalize_canonical") as mock_norm:
        mock_norm.side_effect = lambda text, **kw: type(
            "R", (), {"canonical": text}
        )()
        verify_cite(driver, "c8", "知己知彼", loose=True)
        # Every normalize_canonical call must have apply_loan=True
        for c in mock_norm.call_args_list:
            assert c.kwargs.get("apply_loan") is True, (
                f"Expected apply_loan=True, got {c.kwargs}"
            )
