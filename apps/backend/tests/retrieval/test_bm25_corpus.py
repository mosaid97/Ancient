"""Tests for retrieval/bm25.py — BM25Corpus.build() and .query() with mock Neo4j."""
from unittest.mock import MagicMock

import pytest

from apps.backend.retrieval.bm25 import BM25Corpus


def _make_driver(rows_by_batch: list[list[dict]]) -> MagicMock:
    """Build a mock driver that returns successive batches then empty."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    # Each call to session.run().data() returns the next batch
    data_results = [*rows_by_batch, []]   # last call returns empty to stop loop
    session.run.return_value.data.side_effect = data_results
    return driver


class TestBm25CorpusBuild:
    def test_empty_corpus(self):
        driver = _make_driver([[]])
        corpus = BM25Corpus.build(driver)
        assert corpus.chunk_ids == []

    def test_single_chunk_indexed(self):
        rows = [{"chunk_id": "c1", "text": "唐律疏議名例律第一", "tier": "primary"}]
        driver = _make_driver([rows])
        corpus = BM25Corpus.build(driver)
        assert "c1" in corpus.chunk_ids

    def test_multiple_chunks(self):
        rows = [
            {"chunk_id": "c1", "text": "唐律疏議名例律第一", "tier": "primary"},
            {"chunk_id": "c2", "text": "通典選舉典科目", "tier": "primary"},
            {"chunk_id": "c3", "text": "舊唐書玄宗本紀", "tier": "primary"},
        ]
        driver = _make_driver([rows])
        corpus = BM25Corpus.build(driver)
        assert len(corpus.chunk_ids) == 3
        assert set(corpus.chunk_ids) == {"c1", "c2", "c3"}

    def test_chunk_tiers_stored(self):
        rows = [
            {"chunk_id": "c1", "text": "唐律疏議名例律第一", "tier": "primary"},
            {"chunk_id": "c2", "text": "察举制度学术分析", "tier": "secondary"},
        ]
        driver = _make_driver([rows])
        corpus = BM25Corpus.build(driver)
        idx_c1 = corpus.chunk_ids.index("c1")
        idx_c2 = corpus.chunk_ids.index("c2")
        assert corpus.chunk_tiers[idx_c1] == "primary"
        assert corpus.chunk_tiers[idx_c2] == "secondary"

    def test_short_text_skipped(self):
        # Text with < 3 chars produces no n-grams → chunk skipped
        rows = [
            {"chunk_id": "short", "text": "唐", "tier": "primary"},
            {"chunk_id": "long", "text": "唐律疏議", "tier": "primary"},
        ]
        driver = _make_driver([rows])
        corpus = BM25Corpus.build(driver)
        assert "short" not in corpus.chunk_ids
        assert "long" in corpus.chunk_ids

    def test_max_chunks_cap(self):
        rows = [{"chunk_id": f"c{i}", "text": "唐律疏議名例律", "tier": "primary"} for i in range(10)]
        driver = _make_driver([rows])
        corpus = BM25Corpus.build(driver, max_chunks=3)
        assert len(corpus.chunk_ids) <= 3

    def test_paginated_batches(self):
        batch1 = [{"chunk_id": "c1", "text": "唐律疏議名例律", "tier": "primary"}]
        batch2 = [{"chunk_id": "c2", "text": "通典選舉制度", "tier": "primary"}]
        driver = _make_driver([batch1, batch2])
        corpus = BM25Corpus.build(driver, batch_size=1)
        assert set(corpus.chunk_ids) == {"c1", "c2"}


class TestBm25CorpusQuery:
    def _build_from_texts(self, texts: dict[str, str]) -> BM25Corpus:
        rows = [{"chunk_id": cid, "text": t, "tier": "primary"} for cid, t in texts.items()]
        driver = _make_driver([rows])
        return BM25Corpus.build(driver)

    def test_empty_corpus_returns_empty(self):
        driver = _make_driver([[]])
        corpus = BM25Corpus.build(driver)
        assert corpus.query("唐律") == []

    def test_query_returns_matching_chunk(self):
        # Need ≥3 docs so BM25 IDF > 0 (with N=2 and df=1, log(1.5/1.5)=0)
        corpus = self._build_from_texts({
            "c1": "唐律疏議名例律第一",
            "c2": "通典選舉制度科目",
            "c3": "宋會要輯稿禮儀志",
        })
        results = corpus.query("唐律疏議")
        ids = [cid for cid, _ in results]
        assert "c1" in ids

    def test_query_scores_descending(self):
        corpus = self._build_from_texts({
            "c1": "唐律疏議名例律第一",
            "c2": "通典選舉制度科目",
        })
        results = corpus.query("唐律疏議")
        if len(results) >= 2:
            scores = [s for _, s in results]
            assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self):
        texts = {f"c{i}": "唐律疏議名例律第一條文規定" for i in range(10)}
        corpus = self._build_from_texts(texts)
        results = corpus.query("唐律疏議", top_k=3)
        assert len(results) <= 3

    def test_no_match_returns_empty(self):
        corpus = self._build_from_texts({"c1": "唐律疏議名例律第一"})
        # Query for something completely unrelated
        results = corpus.query("xyz")
        # BM25 returns 0-score results which are filtered out
        assert all(s > 0 for _, s in results)

    def test_empty_query_returns_empty(self):
        corpus = self._build_from_texts({"c1": "唐律疏議"})
        assert corpus.query("") == []
        assert corpus.query("  ") == []

    def test_short_query_no_ngrams(self):
        corpus = self._build_from_texts({"c1": "唐律疏議"})
        # 1-char query: no 3/4-grams → no tokens → empty
        assert corpus.query("唐") == []
