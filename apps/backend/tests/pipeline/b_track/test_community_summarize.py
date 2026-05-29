"""Tests for pipeline/community_summarize.py — community ID + Leiden (B3)."""
import hashlib

import pytest

from apps.backend.pipeline.community_summarize import (
    _community_id,
    load_bipartite_graph,
    run_leiden,
)


class TestCommunityId:
    def test_deterministic_same_input(self):
        ids = ["chunk_001", "chunk_002", "chunk_003"]
        assert _community_id(ids) == _community_id(ids)

    def test_order_independent(self):
        ids_a = ["chunk_001", "chunk_002", "chunk_003"]
        ids_b = ["chunk_003", "chunk_001", "chunk_002"]
        assert _community_id(ids_a) == _community_id(ids_b)

    def test_different_sets_different_id(self):
        assert _community_id(["a", "b"]) != _community_id(["a", "c"])

    def test_length_21(self):
        # Format: "comm_" (5) + 16 hex chars = 21 total
        cid = _community_id(["x", "y", "z"])
        assert len(cid) == 21
        assert cid.startswith("comm_")
        int(cid[5:], 16)  # trailing 16 chars must be valid hex

    def test_single_chunk(self):
        cid = _community_id(["chunk_solo"])
        assert len(cid) == 21
        assert cid.startswith("comm_")

    def test_empty_list(self):
        cid = _community_id([])
        assert len(cid) == 21
        assert cid.startswith("comm_")

    def test_sha1_prefix_matches(self):
        # Separator is "|", not ","
        ids = sorted(["chunk_001", "chunk_002"])
        key = "|".join(ids)
        expected = "comm_" + hashlib.sha1(key.encode()).hexdigest()[:16]
        assert _community_id(["chunk_001", "chunk_002"]) == expected


class TestLoadBipartiteGraph:
    def test_returns_four_tuple(self):
        """load_bipartite_graph with empty Neo4j result returns empty structure."""
        from unittest.mock import MagicMock
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        session.run.return_value.data.return_value = []

        chunk_ids, edges, n_chunks, n_keywords = load_bipartite_graph(driver)
        assert chunk_ids == []
        assert edges == []
        assert n_chunks == 0
        assert n_keywords == 0

    def test_single_edge_parsed(self):
        from unittest.mock import MagicMock
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        session.run.return_value.data.return_value = [
            {"chunk_id": "c1", "keyword_name": "均田"},
        ]

        chunk_ids, edges, n_chunks, n_keywords = load_bipartite_graph(driver)
        assert "c1" in chunk_ids
        assert n_chunks == 1
        assert n_keywords == 1

    def test_multiple_keywords_per_chunk(self):
        from unittest.mock import MagicMock
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        session.run.return_value.data.return_value = [
            {"chunk_id": "c1", "keyword_name": "均田"},
            {"chunk_id": "c1", "keyword_name": "府兵"},
            {"chunk_id": "c2", "keyword_name": "均田"},
        ]

        chunk_ids, edges, n_chunks, n_keywords = load_bipartite_graph(driver)
        assert n_chunks == 2
        assert n_keywords == 2
        assert len(edges) == 3


class TestRunLeiden:
    def test_empty_graph_returns_empty(self):
        communities, modularity = run_leiden([], [], 0, 0)
        assert communities == []

    def test_returns_two_tuple(self):
        chunk_ids = [f"c{i}" for i in range(5)]
        result = run_leiden(chunk_ids, [], 5, 0, min_size=1)
        assert isinstance(result, tuple) and len(result) == 2

    def test_isolated_chunks_below_min_size_filtered(self):
        # 4 isolated chunks with no edges → no community has size ≥ 5 (default)
        chunk_ids = [f"c{i}" for i in range(4)]
        communities, _ = run_leiden(chunk_ids, [], 4, 0)
        assert communities == []

    def test_connected_chunks_grouped(self):
        # 8 chunks all sharing keyword at vertex index 8; min_size=1 so all caught
        chunk_ids = [f"c{i}" for i in range(8)]
        edges = [(i, 8) for i in range(8)]   # keyword node is vertex 8
        communities, _ = run_leiden(chunk_ids, edges, 8, 1, min_size=1)
        assert len(communities) >= 1
        total_chunks = sum(len(c) for c in communities)
        assert 1 <= total_chunks <= len(chunk_ids)

    def test_communities_contain_valid_chunk_ids(self):
        chunk_ids = [f"c{i}" for i in range(8)]
        edges = [(i, 8) for i in range(8)]
        communities, _ = run_leiden(chunk_ids, edges, 8, 1, min_size=1)
        all_returned = {cid for comm in communities for cid in comm}
        assert all_returned.issubset(set(chunk_ids))

    def test_min_size_filters_small_communities(self):
        # 3 isolated chunks (no edges) → all singletons; filtered with min_size=5
        chunk_ids = ["a", "b", "c"]
        edges = [(0, 3), (1, 4), (2, 5)]   # each chunk to own keyword
        communities, _ = run_leiden(chunk_ids, edges, 3, 3, min_size=5)
        assert communities == []

    def test_modularity_is_float(self):
        chunk_ids = [f"c{i}" for i in range(6)]
        edges = [(i, 6) for i in range(6)]
        _, modularity = run_leiden(chunk_ids, edges, 6, 1, min_size=1)
        assert isinstance(modularity, float)
