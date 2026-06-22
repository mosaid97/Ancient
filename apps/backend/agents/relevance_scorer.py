"""Agent-layer alias for the LLM relevance + evidence scorer.

This is **not** the deterministic citation gate — that lives in
:mod:`apps.backend.agents.verifier`. The scorer assigns each candidate
chunk a relevance + evidence score (0–10) via an LLM and is used to
re-rank candidates *before* the deterministic gate runs.

The implementation lives in :mod:`apps.backend.pipeline.evidence_scorer`;
this module just re-exports under the agent-layer name so callers can
import from either layer.
"""
from __future__ import annotations

from apps.backend.pipeline.evidence_scorer import (  # noqa: F401
    EvidenceItem,
    EvidenceRunReport,
    score_evidence,
)

__all__ = ["EvidenceItem", "EvidenceRunReport", "score_evidence"]
