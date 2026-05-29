"""Tests for eval/bootstrap_ci.py — bootstrap CI + ranking metrics."""
import math

import numpy as np
import pytest

from eval.bootstrap_ci import (
    bootstrap_ci,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# --- bootstrap_ci ---

class TestBootstrapCi:
    def test_empty_returns_nan(self):
        r = bootstrap_ci([])
        assert r["n"] == 0
        assert math.isnan(r["mean"])
        assert math.isnan(r["ci_low"])
        assert math.isnan(r["ci_high"])

    def test_single_value(self):
        r = bootstrap_ci([0.75])
        assert r["n"] == 1
        assert r["mean"] == pytest.approx(0.75)
        assert r["ci_low"] == pytest.approx(0.75)
        assert r["ci_high"] == pytest.approx(0.75)

    def test_mean_close_to_true_mean(self):
        vals = [0.1, 0.2, 0.3, 0.4, 0.5]
        r = bootstrap_ci(vals)
        assert r["mean"] == pytest.approx(0.3, abs=1e-9)
        assert r["n"] == 5

    def test_ci_contains_mean(self):
        vals = list(np.linspace(0, 1, 50))
        r = bootstrap_ci(vals, n_resamples=1000)
        assert r["ci_low"] <= r["mean"] <= r["ci_high"]

    def test_ci_width_shrinks_with_more_data(self):
        rng = np.random.default_rng(0)
        small = list(rng.random(10))
        large = list(rng.random(200))
        r_small = bootstrap_ci(small, n_resamples=1000, seed=1)
        r_large = bootstrap_ci(large, n_resamples=1000, seed=1)
        width_small = r_small["ci_high"] - r_small["ci_low"]
        width_large = r_large["ci_high"] - r_large["ci_low"]
        assert width_small > width_large

    def test_deterministic_with_same_seed(self):
        vals = [0.1, 0.5, 0.9, 0.3, 0.7]
        r1 = bootstrap_ci(vals, n_resamples=500, seed=99)
        r2 = bootstrap_ci(vals, n_resamples=500, seed=99)
        assert r1 == r2

    def test_different_seed_same_mean(self):
        vals = [0.2, 0.4, 0.6]
        r1 = bootstrap_ci(vals, seed=1)
        r2 = bootstrap_ci(vals, seed=2)
        assert r1["mean"] == r2["mean"]

    def test_custom_stat_median(self):
        vals = [1.0, 2.0, 3.0, 4.0, 100.0]
        r = bootstrap_ci(vals, stat=np.median, n_resamples=500)
        assert r["mean"] == pytest.approx(3.0)

    def test_all_same_values_zero_width_ci(self):
        vals = [0.5] * 20
        r = bootstrap_ci(vals, n_resamples=200)
        assert r["ci_low"] == pytest.approx(0.5)
        assert r["ci_high"] == pytest.approx(0.5)


# --- ndcg_at_k ---

class TestNdcgAtK:
    def test_perfect_ranking(self):
        relevant = {"a", "b", "c"}
        ranked = ["a", "b", "c", "d", "e"]
        score = ndcg_at_k(relevant, ranked, k=3)
        assert score == pytest.approx(1.0)

    def test_no_relevant_returns_zero(self):
        assert ndcg_at_k(set(), ["a", "b"], k=10) == 0.0

    def test_no_ranked_returns_zero(self):
        assert ndcg_at_k({"a"}, [], k=10) == 0.0

    def test_first_hit_better_than_last_hit(self):
        relevant = {"a"}
        ranked_first = ["a", "b", "c"]
        ranked_last = ["b", "c", "a"]
        assert ndcg_at_k(relevant, ranked_first, k=3) > ndcg_at_k(relevant, ranked_last, k=3)

    def test_cutoff_k_limits_scoring(self):
        relevant = {"d"}
        ranked = ["a", "b", "c", "d"]
        # d is rank 4, so NDCG@3 should be 0 (not counted)
        assert ndcg_at_k(relevant, ranked, k=3) == pytest.approx(0.0)

    def test_partial_recall(self):
        relevant = {"a", "b", "c"}
        ranked = ["a", "x", "y", "z"]
        score = ndcg_at_k(relevant, ranked, k=4)
        assert 0.0 < score < 1.0


# --- recall_at_k ---

class TestRecallAtK:
    def test_all_found(self):
        assert recall_at_k({"a", "b"}, ["a", "b", "c"], k=2) == pytest.approx(1.0)

    def test_none_found(self):
        assert recall_at_k({"a", "b"}, ["c", "d"], k=2) == pytest.approx(0.0)

    def test_empty_relevant(self):
        assert recall_at_k(set(), ["a"], k=5) == 0.0

    def test_partial(self):
        assert recall_at_k({"a", "b", "c"}, ["a", "x", "y"], k=3) == pytest.approx(1 / 3)


# --- precision_at_k ---

class TestPrecisionAtK:
    def test_perfect(self):
        assert precision_at_k({"a", "b"}, ["a", "b"], k=2) == pytest.approx(1.0)

    def test_zero(self):
        assert precision_at_k({"a"}, ["b", "c"], k=2) == pytest.approx(0.0)

    def test_k_zero(self):
        assert precision_at_k({"a"}, ["a"], k=0) == pytest.approx(0.0)

    def test_partial(self):
        assert precision_at_k({"a"}, ["a", "b", "c"], k=3) == pytest.approx(1 / 3)


# --- mrr ---

class TestMrr:
    def test_first_hit(self):
        assert mrr({"a"}, ["a", "b", "c"]) == pytest.approx(1.0)

    def test_second_hit(self):
        assert mrr({"b"}, ["a", "b", "c"]) == pytest.approx(0.5)

    def test_no_hit(self):
        assert mrr({"d"}, ["a", "b", "c"]) == pytest.approx(0.0)

    def test_empty_relevant(self):
        assert mrr(set(), ["a", "b"]) == pytest.approx(0.0)
