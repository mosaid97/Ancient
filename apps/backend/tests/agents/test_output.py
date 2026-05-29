"""Tests for agents/output.py — RibbonResult, SearchResponse (dataclasses)."""
import pytest

from apps.backend.agents.output import RibbonResult, SearchResponse
from apps.backend.agents.verifier import VerifierResult


def _verifier(ok: bool, outcome: str = "insufficient_evidence", strength: str | None = "strong") -> VerifierResult:
    return VerifierResult(
        chunk_id="test_chunk",
        span="test_span",
        outcome="ok" if ok else outcome,
        evidence_strength=strength if ok else None,
    )


class TestRibbonResult:
    def _make(self, **kw) -> RibbonResult:
        defaults = dict(
            chunk_id="c1",
            text="唐律疏議",
            tier="primary",
            page_id="p1",
            document_id="d1",
            chunk_index=0,
            retrieval_score=0.8,
            rerank_score=0.9,
            verifier=_verifier(True),
        )
        defaults.update(kw)
        return RibbonResult(**defaults)

    def test_verified_true_when_verifier_ok(self):
        r = self._make(verifier=_verifier(True))
        assert r.verified is True

    def test_verified_false_when_verifier_not_ok(self):
        r = self._make(verifier=_verifier(False, outcome="insufficient_evidence"))
        assert r.verified is False

    def test_evidence_strength_from_verifier(self):
        r = self._make(verifier=_verifier(True, strength="strong"))
        assert r.evidence_strength == "strong"

    def test_evidence_strength_none_when_not_ok(self):
        r = self._make(verifier=_verifier(False, outcome="insufficient_evidence"))
        assert r.evidence_strength is None

    def test_to_dict_keys(self):
        r = self._make()
        d = r.to_dict()
        for key in ("chunkId", "text", "tier", "pageId", "documentId", "chunkIndex",
                    "retrievalScore", "rerankScore", "verified", "evidenceStrength", "verifierOutcome"):
            assert key in d

    def test_to_dict_scores_rounded(self):
        r = self._make(retrieval_score=0.123456789, rerank_score=0.987654321)
        d = r.to_dict()
        assert d["retrievalScore"] == pytest.approx(0.1235, abs=1e-4)
        assert d["rerankScore"] == pytest.approx(0.9877, abs=1e-4)

    def test_to_dict_chunk_id(self):
        r = self._make(chunk_id="chunk_abc")
        assert r.to_dict()["chunkId"] == "chunk_abc"


class TestSearchResponse:
    def _primary(self) -> RibbonResult:
        return RibbonResult(
            chunk_id="p1", text="primary text", tier="primary",
            page_id="pg1", document_id="d1", chunk_index=0,
            retrieval_score=0.9, rerank_score=0.95,
            verifier=_verifier(True),
        )

    def _secondary(self) -> RibbonResult:
        return RibbonResult(
            chunk_id="s1", text="secondary text", tier="secondary",
            page_id="pg2", document_id="d2", chunk_index=0,
            retrieval_score=0.7, rerank_score=0.75,
            verifier=_verifier(False, outcome="insufficient_evidence"),
        )

    def test_all_results_concatenation(self):
        resp = SearchResponse(query="q", intent="factual")
        resp.primary_ribbon.append(self._primary())
        resp.secondary_ribbon.append(self._secondary())
        assert len(resp.all_results) == 2
        assert resp.all_results[0].tier == "primary"
        assert resp.all_results[1].tier == "secondary"

    def test_to_dict_keys(self):
        resp = SearchResponse(query="q", intent="factual", duration_ms=42.5)
        d = resp.to_dict()
        for key in ("query", "intent", "primaryRibbon", "secondaryRibbon",
                    "durationMs", "totalResults"):
            assert key in d

    def test_to_dict_total_results(self):
        resp = SearchResponse(query="q", intent="factual")
        resp.primary_ribbon.append(self._primary())
        resp.secondary_ribbon.append(self._secondary())
        assert resp.to_dict()["totalResults"] == 2

    def test_empty_ribbons(self):
        resp = SearchResponse(query="q", intent="unknown")
        d = resp.to_dict()
        assert d["primaryRibbon"] == []
        assert d["secondaryRibbon"] == []
        assert d["totalResults"] == 0

    def test_duration_ms_rounded(self):
        resp = SearchResponse(query="q", intent="factual", duration_ms=123.456789)
        assert resp.to_dict()["durationMs"] == pytest.approx(123.5, abs=0.1)

    def test_query_and_intent_preserved(self):
        resp = SearchResponse(query="唐律", intent="interpretive")
        d = resp.to_dict()
        assert d["query"] == "唐律"
        assert d["intent"] == "interpretive"
