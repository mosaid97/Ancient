"""C1: BM25 retrieval over char-3/4-gram tokenized textCanonical (plan §0.6 C1).

BM25 is computed in-memory over a corpus snapshot pulled from Neo4j. For
production scale this snapshot should be refreshed periodically; for the
research artifact it is rebuilt on each search session or once at startup.

Tokenization: overlapping char-3-gram + char-4-gram bag for classical Chinese
(no whitespace segmentation, following rank-bm25 conventions for CJK).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from neo4j import Driver
from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

_NGRAM_SIZES = (3, 4)
_BM25_TOP_K_MULTIPLIER = 3   # fetch 3× top_k candidates, then trim

# ── Cypher ────────────────────────────────────────────────────────────────────

_CORPUS_QUERY = """
MATCH (c:CHUNK)
WHERE coalesce(c.textCanonical, c.text) IS NOT NULL
  AND ($tier IS NULL OR c.tier = $tier)
RETURN c.id AS chunk_id,
       coalesce(c.textCanonical, c.text) AS text,
       c.tier AS tier
ORDER BY c.id
SKIP $skip LIMIT $batch
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ngrams(text: str, sizes: tuple[int, ...] = _NGRAM_SIZES) -> list[str]:
    """Return a bag of overlapping char-n-grams for all n in sizes."""
    tokens: list[str] = []
    for n in sizes:
        tokens.extend(text[i : i + n] for i in range(len(text) - n + 1))
    return tokens


def _tokenize(text: str) -> list[str]:
    """Tokenize classical-Chinese text for BM25."""
    # Strip whitespace and punctuation before n-gramming
    cleaned = re.sub(r"\s+", "", text or "")
    return _ngrams(cleaned)


# ── BM25 Index ────────────────────────────────────────────────────────────────

@dataclass
class BM25Corpus:
    """In-memory BM25 index built from Neo4j CHUNK nodes."""

    chunk_ids: list[str]
    chunk_tiers: list[str | None]
    index: BM25Okapi

    @classmethod
    def build(
        cls,
        driver: Driver,
        *,
        tier: str | None = None,
        batch_size: int = 2_000,
        max_chunks: int | None = None,
    ) -> "BM25Corpus":
        """Pull corpus from Neo4j and build BM25Okapi index.

        Args:
            driver: Open Neo4j driver.
            tier: If set, only index chunks of this tier.
            batch_size: Neo4j fetch page size.
            max_chunks: Optional cap (for smoke tests).
        """
        chunk_ids: list[str] = []
        chunk_tiers: list[str | None] = []
        tokenized_corpus: list[list[str]] = []
        skip = 0

        while True:
            with driver.session() as s:
                rows = s.run(
                    _CORPUS_QUERY,
                    tier=tier,
                    skip=skip,
                    batch=batch_size,
                ).data()
            if not rows:
                break
            for row in rows:
                tokens = _tokenize(row["text"])
                if tokens:
                    chunk_ids.append(row["chunk_id"])
                    chunk_tiers.append(row.get("tier"))
                    tokenized_corpus.append(tokens)
            skip += len(rows)
            if max_chunks and len(chunk_ids) >= max_chunks:
                break

        log.info("BM25Corpus built: %d chunks indexed", len(chunk_ids))
        return cls(
            chunk_ids=chunk_ids,
            chunk_tiers=chunk_tiers,
            index=BM25Okapi(tokenized_corpus),
        )

    def query(self, query_text: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        """Return (chunk_id, bm25_score) pairs for the top-K results.

        Returns an empty list if the corpus is empty.
        """
        if not self.chunk_ids:
            return []
        tokens = _tokenize(query_text)
        if not tokens:
            return []
        scores = self.index.get_scores(tokens)
        # argsort descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            (self.chunk_ids[i], float(scores[i]))
            for i, _ in ranked[:top_k * _BM25_TOP_K_MULTIPLIER]
            if scores[i] > 0
        ][:top_k]
