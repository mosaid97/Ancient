"""C1: Dense (vector) retrieval over chunk_embedding_classical (plan §0.6 C1).

Thin adapter over the existing pipeline/search.py vector_search() so the
hybrid fuser has a uniform interface for all retrieval legs.
"""
from __future__ import annotations

import logging
import os

from neo4j import Driver
from openai import OpenAI

log = logging.getLogger(__name__)

_DEFAULT_TOP_K = 50
_EMBED_BATCH = 10

# ── Cypher ────────────────────────────────────────────────────────────────────

_DENSE_QUERY = """
CALL db.index.vector.queryNodes('chunk_embedding_classical', $top_k, $embedding)
YIELD node AS c, score
WHERE ($tier IS NULL OR c.tier = $tier)
  AND score >= $threshold
RETURN c.id AS chunk_id, c.tier AS tier, score
ORDER BY score DESC
"""

_VERN_DENSE_QUERY = """
CALL db.index.vector.queryNodes('chunk_embedding_vernacular', $top_k, $embedding)
YIELD node AS c, score
WHERE ($tier IS NULL OR c.tier = $tier)
  AND score >= $threshold
  AND c.textVernacular IS NOT NULL
RETURN c.id AS chunk_id, c.tier AS tier, score
ORDER BY score DESC
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _embed_query(text: str, *, model: str) -> list[float] | None:
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.silra.cn/v1/"),
    )
    for attempt in range(3):
        try:
            resp = client.embeddings.create(input=[text], model=model)
            return resp.data[0].embedding
        except Exception as exc:
            import time
            if attempt == 2:
                log.warning("Query embed failed: %s", exc)
                return None
            time.sleep(2 ** attempt)
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def dense_search(
    driver: Driver,
    query_text: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
    tier: str | None = None,
    threshold: float = 0.0,
    use_vernacular: bool = False,
    model: str | None = None,
) -> list[tuple[str, float]]:
    """Run dense ANN vector search and return (chunk_id, score) pairs.

    Args:
        driver: Open Neo4j driver.
        query_text: Natural-language query string.
        top_k: Number of candidates to retrieve.
        tier: Optional tier filter ('primary' | 'secondary').
        threshold: Minimum cosine score.
        use_vernacular: If True, query the vernacular embedding index instead.
        model: Override embedding model.
    """
    embed_model = model or os.getenv("EMBED_LLM_MODEL", "text-embedding-v4")
    vec = _embed_query(query_text, model=embed_model)
    if vec is None:
        return []

    cypher = _VERN_DENSE_QUERY if use_vernacular else _DENSE_QUERY
    with driver.session() as s:
        rows = s.run(cypher, top_k=top_k, embedding=vec, tier=tier, threshold=threshold).data()

    return [(r["chunk_id"], float(r["score"])) for r in rows]
