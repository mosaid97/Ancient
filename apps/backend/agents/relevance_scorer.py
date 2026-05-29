"""LLM-based relevance + evidence scorer (formerly called 'verifier').

This is NOT the zero-hallucination citation gate — that lives in
:mod:`apps.backend.agents.verifier`.  This module assigns a 0–10 relevance
and 0–10 evidence score to each search result via deepseek-chat, useful for
re-ranking before the deterministic gate runs.

The module re-exports the full public API from pipeline.verifier so that
existing callers (scripts, notebooks) can migrate to the new name without a
hard cut-over.

Preferred import going forward::

    from apps.backend.agents.relevance_scorer import verify_chunks, VerificationItem

Old import (still works, but deprecated)::

    from apps.backend.pipeline.verifier import verify_chunks, VerificationItem
"""
from __future__ import annotations

# Re-export everything from the original module so callers can migrate at
# their own pace.
from apps.backend.pipeline.verifier import (  # noqa: F401
    VerificationItem,
    VerifierRunReport,
    verify_chunks,
)

__all__ = ["VerificationItem", "VerifierRunReport", "verify_chunks"]
