"""Deterministic citation verifier — Move 5 (plan §2.5, §7, A3).

What this module proves
-----------------------
Given a chunk and a candidate cited *span*, the verifier checks that the span
is a substring of the chunk's text **after** the 7-step philological
normalization pipeline (plan §2.7):

    NFC → whitespace → T-S → 異體字 → 避諱 (era-conditional)
        → 通假字 (optional, off by default) → mojimoji (ja)

The guarantee is therefore "the cited span is present in the source modulo
philological normalization," not raw byte-for-byte equality. Loose=True
additionally folds 通假字 (phonetic loans), which broadens what counts as a
match. The verifier is deterministic — no LLM involvement.

Outcome model
-------------
A successful match against ``textCanonical`` (or its ``text`` raw fallback)
is a **source match** and earns ``outcome='ok'``. A successful match against
``textVernacular`` is a **translation match** — vernacular text is
LLM-generated and is not a primary source — so it earns
``outcome='translation_match'``; ``VerifierResult.ok`` is False in that case.
Callers that wish to surface translation matches separately can read
``.matched_in`` and ``.translation_match``.

Short spans (under :data:`MIN_SPAN_CHARS` after normalization) are rejected
with ``failure_mode='span_too_short'`` to prevent trivially-containing
spans like single characters (之, 曰, 王) from earning a "verified" badge.

Public API
----------
verify_cite(driver, chunk_id, span, *, loose=False, era=None,
            min_span_chars=MIN_SPAN_CHARS) -> VerifierResult
VerifierResult
VerifierFailureMode
extract_quotable_span(text, *, min_chars=MIN_SPAN_CHARS) -> str | None
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from neo4j import Driver

from apps.backend.normalize.pipeline import normalize_canonical

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

#: Minimum span length (Unicode code points) for the verifier to accept a
#: candidate citation. Single characters like 之/曰/王 occur in virtually
#: every classical chunk, so a substring match on them proves nothing.
MIN_SPAN_CHARS = 4

_EDITORIAL_COMMENTARY_TYPES = {"校點", "校訂", "箋解"}

# CJK Unified Ideographs blocks — matches Han characters used in classical
# Chinese / Japanese kanbun. Used by extract_quotable_span to find verbatim
# spans in a (potentially natural-language) query.
_HAN_RUN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")

_CHUNK_FETCH = """
MATCH (c:CHUNK {id: $chunk_id})
RETURN
  c.id                   AS chunk_id,
  c.textCanonical        AS text_canonical,
  c.textVernacular       AS text_vernacular,
  c.text                 AS text_raw,
  c.tier                 AS tier,
  c.language             AS language,
  c.editorialLayerType   AS editorial_layer_type,
  c.detectedEra          AS detected_era
"""

_WRITE_FAILURE = """
MERGE (f:VERIFIER_FAILURE {id: $id})
ON CREATE SET
  f.chunkId           = $chunk_id,
  f.span              = $span,
  f.failureMode       = $failure_mode,
  f.tier              = $tier,
  f.language          = $language,
  f.ts                = $ts
"""


# ── Enums / data classes ─────────────────────────────────────────────────────

class VerifierFailureMode(str, Enum):
    CHUNK_NOT_FOUND = "chunk_not_found"
    NO_TEXT = "no_text"
    SPAN_NOT_FOUND = "span_not_found"
    SPAN_TOO_SHORT = "span_too_short"


@dataclass
class VerifierResult:
    """Result of a single cite verification."""

    chunk_id: str
    span: str
    outcome: str           # 'ok' | 'translation_match' | 'insufficient_evidence'
    failure_mode: str | None = None
    evidence_strength: str | None = None
    tier: str | None = None
    language: str | None = None
    matched_in: str | None = None   # 'canonical' | 'vernacular' | 'raw'
    normalized_span: str | None = None

    @property
    def ok(self) -> bool:
        """True only for source matches (canonical or raw fallback).

        Translation-only matches (matched_in='vernacular') return False;
        the vernacular text is LLM-generated and does not satisfy the
        "present in a numbered source page" guarantee.
        """
        return self.outcome == "ok"

    @property
    def translation_match(self) -> bool:
        return self.outcome == "translation_match"


# ── Internal helpers ─────────────────────────────────────────────────────────

def _evidence_strength(tier: str | None, layer: str | None) -> str:
    """Assign the evidence-strength badge per plan §7 step 8."""
    if tier == "primary":
        if layer == "pure-source":
            return "primary_source"
        if layer == "疏議":
            return "primary_疏議"
        if layer in _EDITORIAL_COMMENTARY_TYPES:
            return "editorial_commentary"
        return "primary_source"   # unknown layer → treat as source
    return "scholarly_interpretation"


def _normalize(text: str, *, lang: str, era: str | None, apply_loan: bool) -> str:
    """Run the 7-step pipeline and return the canonical string."""
    result = normalize_canonical(
        text,
        lang=lang,
        era=era,
        apply_loan=apply_loan,
    )
    return result.canonical


def _write_failure(
    driver: Driver,
    *,
    chunk_id: str,
    span: str,
    failure_mode: str,
    tier: str | None,
    language: str | None,
) -> None:
    try:
        with driver.session() as s:
            s.run(
                _WRITE_FAILURE,
                id=str(uuid.uuid4()),
                chunk_id=chunk_id,
                span=span,
                failure_mode=failure_mode,
                tier=tier or "",
                language=language or "",
                ts=datetime.now(timezone.utc).isoformat(),
            ).consume()
    except Exception as exc:
        log.error("Failed to write VERIFIER_FAILURE node: %s", exc)


# ── Public helpers ───────────────────────────────────────────────────────────

def extract_quotable_span(text: str, *, min_chars: int = MIN_SPAN_CHARS) -> str | None:
    """Return the longest contiguous Han-character run in ``text`` of length
    ``>= min_chars``, or None.

    Used to pull a verbatim classical-Chinese substring out of a
    natural-language query. Returns None when the input has no qualifying
    Han run (typical for queries written in modern punctuation-rich prose),
    signalling to the caller that no verbatim span is available to verify
    against.
    """
    if not text:
        return None
    runs = _HAN_RUN_RE.findall(text)
    if not runs:
        return None
    longest = max(runs, key=len)
    return longest if len(longest) >= min_chars else None


# ── Public API ───────────────────────────────────────────────────────────────

def verify_cite(
    driver: Driver,
    chunk_id: str,
    span: str,
    *,
    loose: bool = False,
    era: str | None = None,
    min_span_chars: int = MIN_SPAN_CHARS,
) -> VerifierResult:
    """Deterministically verify that ``span`` occurs in the referenced CHUNK.

    Args:
        driver: Open Neo4j driver.
        chunk_id: The CHUNK node ``id`` to verify against.
        span: The cited text span to look for.
        loose: If True, enable the 通假字 step (phonetic-loan substitution).
            Default False per plan §2.7 — off by default in the verifier.
        era: Override the era for 避諱 selection.  When None, the chunk's
            ``detectedEra`` property is used.
        min_span_chars: Reject spans shorter than this after stripping (in
            Unicode code points). Defaults to :data:`MIN_SPAN_CHARS`.

    Returns:
        :class:`VerifierResult`. ``outcome`` is:
          - ``'ok'`` — span found in the canonical / raw source text.
          - ``'translation_match'`` — span only found in the vernacular
            (LLM-translated) text; ``ok`` property is False.
          - ``'insufficient_evidence'`` — chunk missing, text empty, span
            absent, or span shorter than ``min_span_chars``.

        A ``(:VERIFIER_FAILURE)`` node is written when the chunk exists but
        the span cannot be located (or is too short).
    """
    stripped = span.strip() if span else ""
    if not stripped:
        return VerifierResult(
            chunk_id=chunk_id,
            span=span,
            outcome="insufficient_evidence",
            failure_mode=VerifierFailureMode.NO_TEXT,
        )

    if len(stripped) < min_span_chars:
        _write_failure(
            driver,
            chunk_id=chunk_id,
            span=span,
            failure_mode=VerifierFailureMode.SPAN_TOO_SHORT,
            tier=None,
            language=None,
        )
        return VerifierResult(
            chunk_id=chunk_id,
            span=span,
            outcome="insufficient_evidence",
            failure_mode=VerifierFailureMode.SPAN_TOO_SHORT,
        )

    # ── 1. Fetch chunk ────────────────────────────────────────────────────────
    with driver.session() as s:
        rows = s.run(_CHUNK_FETCH, chunk_id=chunk_id).data()

    if not rows:
        log.warning("verify_cite: chunk %s not found", chunk_id)
        _write_failure(driver, chunk_id=chunk_id, span=span,
                       failure_mode=VerifierFailureMode.CHUNK_NOT_FOUND,
                       tier=None, language=None)
        return VerifierResult(
            chunk_id=chunk_id,
            span=span,
            outcome="insufficient_evidence",
            failure_mode=VerifierFailureMode.CHUNK_NOT_FOUND,
        )

    row = rows[0]
    tier = row.get("tier")
    language = row.get("language") or "zh"
    edit_layer = row.get("editorial_layer_type")
    chunk_era = era or row.get("detected_era")

    # textCanonical is preferred; fall back to raw text when translation not yet run
    text_canonical = row.get("text_canonical") or row.get("text_raw") or ""
    text_vernacular = row.get("text_vernacular") or ""

    if not text_canonical:
        _write_failure(driver, chunk_id=chunk_id, span=span,
                       failure_mode=VerifierFailureMode.NO_TEXT,
                       tier=tier, language=language)
        return VerifierResult(
            chunk_id=chunk_id,
            span=span,
            outcome="insufficient_evidence",
            failure_mode=VerifierFailureMode.NO_TEXT,
            tier=tier,
            language=language,
        )

    # ── 2. Normalize span and chunk texts ─────────────────────────────────────
    norm_span = _normalize(stripped, lang=language, era=chunk_era, apply_loan=loose)
    norm_canonical = _normalize(text_canonical, lang=language, era=chunk_era, apply_loan=loose)
    norm_vernacular = (
        _normalize(text_vernacular, lang=language, era=chunk_era, apply_loan=loose)
        if text_vernacular
        else ""
    )

    # Re-check post-normalization length — normalization can drop chars
    # (NFC fold, whitespace collapse) and we want the guard to apply to
    # what actually gets matched.
    if len(norm_span) < min_span_chars:
        _write_failure(
            driver,
            chunk_id=chunk_id,
            span=span,
            failure_mode=VerifierFailureMode.SPAN_TOO_SHORT,
            tier=tier,
            language=language,
        )
        return VerifierResult(
            chunk_id=chunk_id,
            span=span,
            outcome="insufficient_evidence",
            failure_mode=VerifierFailureMode.SPAN_TOO_SHORT,
            tier=tier,
            language=language,
            normalized_span=norm_span,
        )

    # ── 3. Substring check: canonical wins over vernacular ────────────────────
    if norm_span in norm_canonical:
        return VerifierResult(
            chunk_id=chunk_id,
            span=span,
            outcome="ok",
            evidence_strength=_evidence_strength(tier, edit_layer),
            tier=tier,
            language=language,
            matched_in="canonical",
            normalized_span=norm_span,
        )

    if norm_vernacular and norm_span in norm_vernacular:
        # Translation-only match: surface separately, not as 'ok'.
        # The vernacular is an LLM translation, not a primary source.
        return VerifierResult(
            chunk_id=chunk_id,
            span=span,
            outcome="translation_match",
            evidence_strength="translation_match",
            tier=tier,
            language=language,
            matched_in="vernacular",
            normalized_span=norm_span,
        )

    # ── 4. Failure ────────────────────────────────────────────────────────────
    log.info(
        "verify_cite FAIL chunk=%s span=%r (normalized=%r) not in canonical[:%d]",
        chunk_id, span[:40], norm_span[:40], len(norm_canonical),
    )
    _write_failure(
        driver,
        chunk_id=chunk_id,
        span=span,
        failure_mode=VerifierFailureMode.SPAN_NOT_FOUND,
        tier=tier,
        language=language,
    )
    return VerifierResult(
        chunk_id=chunk_id,
        span=span,
        outcome="insufficient_evidence",
        failure_mode=VerifierFailureMode.SPAN_NOT_FOUND,
        tier=tier,
        language=language,
        normalized_span=norm_span,
    )
