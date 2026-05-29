"""C1: Community-summary retrieval for synthesis/broad intents (plan §0.6 C1).

For queries classified as 'synthesis' or 'broad', the community-summary
ANN index is queried first to identify relevant communities, then all
chunks in those communities are added as candidates.
"""
from __future__ import annotations

import logging
import os

from neo4j import Driver
from openai import OpenAI

log = logging.getLogger(__name__)

_COMMUNITY_TOP_K = 5     # community summaries to retrieve per query
_DEFAULT_THRESHOLD = 0.6  # lower than chunk threshold — summaries are looser

_COMMUNITY_VECTOR_QUERY = """
CALL db.index.vector.queryNodes('community_summary_embedding_index', $top_k, $embedding)
YIELD node AS comm, score
WHERE score >= $threshold
RETURN comm.id AS community_id, score
ORDER BY score DESC
"""

_COMMUNITY_CHUNKS_QUERY = """
MATCH (c:CHUNK)-[:IN_COMMUNITY]->(comm:COMMUNITY {id: $community_id})
WHERE c.embedding IS NOT NULL
RETURN c.id AS chunk_id, c.tier AS tier
LIMIT 50
"""


def community_search(
    driver: Driver,
    query_text: str,
    *,
    top_k: int = _COMMUNITY_TOP_K,
    threshold: float = _DEFAULT_THRESHOLD,
    model: str | None = None,
) -> list[tuple[str, float]]:
    """Find chunks via community-summary routing.

    Returns (chunk_id, community_score) pairs for all chunks in the
    top-K matching communities. Score is inherited from the community match.
    """
    embed_model = model or os.getenv("EMBED_LLM_MODEL", "text-embedding-v4")
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.silra.cn/v1/"),
    )

    # Embed query
    try:
        resp = client.embeddings.create(input=[query_text], model=embed_model)
        vec = resp.data[0].embedding
    except Exception as exc:
        log.warning("Community search embed failed: %s", exc)
        return []

    # Find matching communities
    with driver.session() as s:
        comm_rows = s.run(
            _COMMUNITY_VECTOR_QUERY,
            top_k=top_k,
            embedding=vec,
            threshold=threshold,
        ).data()

    if not comm_rows:
        return []

    # Collect chunk IDs from each community
    results: list[tuple[str, float]] = []
    seen: set[str] = set()
    for row in comm_rows:
        comm_score = float(row["score"])
        with driver.session() as s:
            chunk_rows = s.run(
                _COMMUNITY_CHUNKS_QUERY,
                community_id=row["community_id"],
            ).data()
        for cr in chunk_rows:
            cid = cr["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                results.append((cid, comm_score))

    return results
