"""Tests for agents/intent.py — tier_boost (pure Python) + classify_intent mock."""
from unittest.mock import MagicMock, patch

import pytest

from apps.backend.agents.intent import classify_intent, tier_boost


class TestTierBoost:
    def test_factual_boosts_primary(self):
        hits = [("a", 0.5), ("b", 0.4)]
        tiers = {"a": "primary", "b": "secondary"}
        result = tier_boost(hits, tiers, "factual")
        scores = dict(result)
        assert scores["a"] == pytest.approx(0.5 + 0.15)
        assert scores["b"] == pytest.approx(0.4)

    def test_interpretive_boosts_secondary(self):
        hits = [("a", 0.5), ("b", 0.4)]
        tiers = {"a": "primary", "b": "secondary"}
        result = tier_boost(hits, tiers, "interpretive")
        scores = dict(result)
        assert scores["a"] == pytest.approx(0.5)
        assert scores["b"] == pytest.approx(0.4 + 0.15)

    def test_synthesis_no_boost(self):
        hits = [("a", 0.5), ("b", 0.4)]
        tiers = {"a": "primary", "b": "secondary"}
        result = tier_boost(hits, tiers, "synthesis")
        assert result == hits

    def test_unknown_no_boost(self):
        hits = [("a", 0.5)]
        tiers = {"a": "primary"}
        assert tier_boost(hits, tiers, "unknown") == hits

    def test_custom_boost_value(self):
        hits = [("x", 0.3)]
        tiers = {"x": "primary"}
        result = tier_boost(hits, tiers, "factual", boost=0.20)
        assert dict(result)["x"] == pytest.approx(0.5)

    def test_missing_tier_no_boost(self):
        hits = [("x", 0.5)]
        tiers = {}  # chunk not in tier map
        result = tier_boost(hits, tiers, "factual")
        assert dict(result)["x"] == pytest.approx(0.5)

    def test_empty_hits(self):
        assert tier_boost([], {}, "factual") == []

    def test_order_preserved(self):
        hits = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        tiers = {"a": "secondary", "b": "primary", "c": "secondary"}
        result = tier_boost(hits, tiers, "factual")
        ids = [cid for cid, _ in result]
        assert ids == ["a", "b", "c"]

    def test_none_tier_treated_as_no_boost(self):
        hits = [("x", 0.5)]
        tiers = {"x": None}
        result = tier_boost(hits, tiers, "factual")
        assert dict(result)["x"] == pytest.approx(0.5)


class TestClassifyIntent:
    def _mock_client(self, raw_text: str):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = raw_text
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        return mock_client

    def test_factual_returned(self):
        with patch("apps.backend.agents.intent.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = self._mock_client("factual")
            assert classify_intent("唐律中謀反罪如何定義") == "factual"

    def test_interpretive_returned(self):
        with patch("apps.backend.agents.intent.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = self._mock_client("interpretive")
            assert classify_intent("學者如何解讀唐代連坐制度") == "interpretive"

    def test_synthesis_returned(self):
        with patch("apps.backend.agents.intent.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = self._mock_client("synthesis")
            assert classify_intent("唐代整體法律體系") == "synthesis"

    def test_unknown_on_garbage_response(self):
        with patch("apps.backend.agents.intent.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = self._mock_client("I cannot classify this")
            assert classify_intent("some query") == "unknown"

    def test_unknown_on_api_error(self):
        with patch("apps.backend.agents.intent.OpenAI") as MockOpenAI:
            client = MagicMock()
            client.chat.completions.create.side_effect = RuntimeError("API down")
            MockOpenAI.return_value = client
            assert classify_intent("any query") == "unknown"

    def test_case_insensitive_matching(self):
        with patch("apps.backend.agents.intent.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = self._mock_client("FACTUAL")
            assert classify_intent("q") == "factual"
