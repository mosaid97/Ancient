"""Deterministic zero-hallucination citation verifier — Move 5 (plan §2.5, §7, A3).

Every cite tuple (chunk_id, span) returned by the search pipeline passes
through this gate before reaching the user.  The check is:

    normalize(span) ⊆ normalize(chunk.textCanonical)
              OR
    normalize(span) ⊆ normalize(chunk.textVernacular)

using the full 7-step philological normalization pipeline (plan §2.7):
  NFC → whitespace → T-S → 異體字 → 避諱 (era-conditional) → 通假字 (opt) → mojimoji (ja)

Any failure → write a (:VERIFIER_FAILURE) node and return
``VerifierResult(outcome='insufficient_evidence', ...)``.

Evidence-strength badges (plan §7, step 8):
  'primary_source'        tier=primary, editorialLayerType='pure-source'
  'primary_疏議'          tier=primary, editorialLayerType='疏議'
  'editorial_commentary'  tier=primary, editorialLayerType in {校點,校訂,箋解}
  'scholarly_interpretation' tier=secondary

Public API
----------
verify_cite(driver, chunk_id, span, *, loose=False, era=None) -> VerifierResult
VerifierResult
VerifierFailureMode
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from neo4j import Driver

from apps.backend.normalize.pipeline import normalize_canonical

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_EDITORIAL_COMMENTARY_TYPES = {"校點", "校訂", "箋解"}

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


@dataclass
class VerifierResult:
    """Result of a single cite verification."""

    chunk_id: str
    span: str
    outcome: str           # 'ok' | 'insufficient_evidence'
    failure_mode: str | None = None
    evidence_strength: str | None = None
    tier: str | None = None
    language: str | None = None
    matched_in: str | None = None   # 'canonical' | 'vernacular' | 'raw'
    normalized_span: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome == "ok"


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


# ── Public API ───────────────────────────────────────────────────────────────

def verify_cite(
    driver: Driver,
    chunk_id: str,
    span: str,
    *,
    loose: bool = False,
    era: str | None = None,
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

    Returns:
        :class:`VerifierResult` with ``outcome='ok'`` on success or
        ``outcome='insufficient_evidence'`` on any failure.  A
        ``(:VERIFIER_FAILURE)`` node is written on failure.
    """
    if not span or not span.strip():
        return VerifierResult(
            chunk_id=chunk_id,
            span=span,
            outcome="insufficient_evidence",
            failure_mode=VerifierFailureMode.NO_TEXT,
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
    norm_span = _normalize(span.strip(), lang=language, era=chunk_era, apply_loan=loose)
    norm_canonical = _normalize(text_canonical, lang=language, era=chunk_era, apply_loan=loose)
    norm_vernacular = (
        _normalize(text_vernacular, lang=language, era=chunk_era, apply_loan=loose)
        if text_vernacular
        else ""
    )

    # ── 3. Exact-substring check ──────────────────────────────────────────────
    matched_in: str | None = None
    if norm_span in norm_canonical:
        matched_in = "canonical"
    elif norm_vernacular and norm_span in norm_vernacular:
        matched_in = "vernacular"

    if matched_in:
        return VerifierResult(
            chunk_id=chunk_id,
            span=span,
            outcome="ok",
            evidence_strength=_evidence_strength(tier, edit_layer),
            tier=tier,
            language=language,
            matched_in=matched_in,
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
