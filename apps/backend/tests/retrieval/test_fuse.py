"""Tests for retrieval/fuse.py — RRF fusion."""
import pytest

from apps.backend.retrieval.fuse import normalize_scores, rrf_fuse


class TestRrfFuse:
    def test_empty_input(self):
        assert rrf_fuse([]) == []

    def test_single_list_passthrough(self):
        hits = [("a", 1.0), ("b", 0.5), ("c", 0.1)]
        result = rrf_fuse([hits])
        ids = [r[0] for r in result]
        assert ids == ["a", "b", "c"]

    def test_rrf_score_formula(self):
        # rank 1 in one list → 1/(60+1)
        result = rrf_fuse([[("a", 1.0)]])
        assert result[0][1] == pytest.approx(1.0 / 61)

    def test_item_in_both_lists_beats_item_in_one(self):
        list1 = [("a", 1.0), ("b", 0.5)]
        list2 = [("a", 0.9), ("c", 0.4)]
        result = rrf_fuse([list1, list2])
        scores = dict(result)
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]

    def test_top_k_limits_output(self):
        hits = [(str(i), float(i)) for i in range(200)]
        result = rrf_fuse([hits], top_k=50)
        assert len(result) == 50

    def test_ranked_descending(self):
        list1 = [("a", 1.0), ("b", 0.5)]
        list2 = [("b", 0.9), ("c", 0.1)]
        result = rrf_fuse([list1, list2])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_score_independent_of_original_score_values(self):
        # Same rank positions, different score values → same RRF result
        list_a = [("x", 100.0), ("y", 99.0)]
        list_b = [("x", 0.01), ("y", 0.001)]
        r1 = rrf_fuse([list_a])
        r2 = rrf_fuse([list_b])
        assert [cid for cid, _ in r1] == [cid for cid, _ in r2]
        assert [s for _, s in r1] == pytest.approx([s for _, s in r2])

    def test_three_legs_accumulate(self):
        l1 = [("a", 1.0)]
        l2 = [("a", 1.0)]
        l3 = [("a", 1.0)]
        result = rrf_fuse([l1, l2, l3])
        assert result[0][1] == pytest.approx(3.0 / 61)

    def test_tie_broken_stably(self):
        # Two items both at rank 1 in one list each → equal RRF scores
        list1 = [("a", 1.0)]
        list2 = [("b", 1.0)]
        result = rrf_fuse([list1, list2])
        scores = {cid: s for cid, s in result}
        assert scores["a"] == pytest.approx(scores["b"])

    def test_empty_sublist_handled(self):
        result = rrf_fuse([[], [("a", 1.0)]])
        assert result[0][0] == "a"

    def test_default_top_k_100(self):
        hits = [(str(i), float(i)) for i in range(200)]
        result = rrf_fuse([hits])
        assert len(result) == 100


class TestNormalizeScores:
    def test_empty(self):
        assert normalize_scores([]) == []

    def test_all_same_gives_1(self):
        hits = [("a", 5.0), ("b", 5.0)]
        result = normalize_scores(hits)
        assert all(s == pytest.approx(1.0) for _, s in result)

    def test_range_0_to_1(self):
        hits = [("a", 3.0), ("b", 1.0), ("c", 2.0)]
        result = normalize_scores(hits)
        scores = [s for _, s in result]
        assert min(scores) == pytest.approx(0.0)
        assert max(scores) == pytest.approx(1.0)

    def test_order_preserved(self):
        hits = [("a", 3.0), ("b", 1.0), ("c", 2.0)]
        result = normalize_scores(hits)
        ids = [cid for cid, _ in result]
        assert ids == ["a", "b", "c"]

    def test_single_item(self):
        result = normalize_scores([("a", 0.7)])
        assert result[0][1] == pytest.approx(1.0)
