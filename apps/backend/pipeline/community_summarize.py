"""B3: GraphRAG-style Leiden community detection + LLM summaries (plan §0.6 Track B, B3).

Pipeline:
  1. load_bipartite_graph(): pull (:CHUNK)-[:MENTION]->(:KEYWORD) edges from Neo4j
     into an igraph bipartite graph.
  2. run_leiden(): Leiden community detection (igraph.Graph.community_leiden) over
     the CHUNK projection (project away KEYWORD nodes).
  3. For each community with size ≥ min_size:
     a. collect representative chunk texts
     b. call deepseek-chat to generate a ~200-token modern-Chinese community summary
     c. embed the summary via text-embedding-v4
     d. MERGE (:COMMUNITY {id, summary, embedding, size, modularity}) nodes
     e. MERGE (:CHUNK)-[:IN_COMMUNITY]->(:COMMUNITY) edges
  4. Add community_summary_embedding_index (already in schema.py).

ADR (AGENTS.md §11):
  - Leiden resolution = 1.0, min_community_size = 5 chunks
  - Summary model: deepseek-chat, ~200 tokens
  - Expected community count: tens–hundreds (sane range for this corpus)
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver
from openai import OpenAI

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_LEIDEN_RESOLUTION = 1.0
_MIN_COMMUNITY_SIZE = 5
_MAX_SUMMARY_CHUNKS = 10    # representative chunks fed to LLM per community
_SUMMARY_MAX_TOKENS = 300
_EMBED_BATCH = 10
_EMBED_DIMS = 1024

_SUMMARY_PROMPT = """\
你是一位专研中国古代史的学者。以下是来自同一主题社群的若干文献片段，请用现代汉语（约200字）概括这些片段的核心主题、时代背景和学术意义。概括应简明扼要，以便研究者快速判断该社群是否与其查询相关。

文献片段：
{chunk_texts}

请输出一段200字左右的现代汉语概括："""


# ── Cypher ────────────────────────────────────────────────────────────────────

_BIPARTITE_QUERY = """
MATCH (c:CHUNK)-[:MENTION]->(k:KEYWORD)
WHERE c.embedding IS NOT NULL
RETURN c.id AS chunk_id, k.name AS keyword_name
LIMIT $limit
"""

_CHUNK_TEXTS_QUERY = """
UNWIND $chunk_ids AS cid
MATCH (c:CHUNK {id: cid})
RETURN c.id AS chunk_id, coalesce(c.textCanonical, c.text) AS text
"""

_COMMUNITY_MERGE = """
UNWIND $communities AS comm
MERGE (c:COMMUNITY {id: comm.id})
ON CREATE SET
  c.summary    = comm.summary,
  c.embedding  = comm.embedding,
  c.size       = comm.size,
  c.modularity = comm.modularity,
  c.ts         = comm.ts
ON MATCH SET
  c.summary    = comm.summary,
  c.embedding  = comm.embedding,
  c.size       = comm.size,
  c.ts         = comm.ts
"""

_IN_COMMUNITY_MERGE = """
UNWIND $memberships AS m
MATCH (chunk:CHUNK {id: m.chunk_id}), (comm:COMMUNITY {id: m.community_id})
MERGE (chunk)-[:IN_COMMUNITY]->(comm)
"""


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class CommunityResult:
    community_id: str
    chunk_ids: list[str]
    summary: str
    embedding: list[float] | None
    modularity: float
    size: int


@dataclass
class CommunityReport:
    chunks_loaded: int = 0
    edges_loaded: int = 0
    communities_detected: int = 0
    communities_written: int = 0
    memberships_written: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_loaded": self.chunks_loaded,
            "edges_loaded": self.edges_loaded,
            "communities_detected": self.communities_detected,
            "communities_written": self.communities_written,
            "memberships_written": self.memberships_written,
            "duration_seconds": round(self.duration_seconds, 2),
            "errors": self.errors[:20],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_llm_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.silra.cn/v1/"),
    )


def _generate_summary(chunk_texts: list[str], *, model: str) -> str:
    """Call deepseek-chat to summarize representative chunks."""
    client = _get_llm_client()
    combined = "\n\n".join(f"[{i+1}] {t[:300]}" for i, t in enumerate(chunk_texts))
    prompt = _SUMMARY_PROMPT.format(chunk_texts=combined)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_SUMMARY_MAX_TOKENS,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt == 2:
                log.warning("Summary generation failed: %s", exc)
                return ""
            time.sleep(2 ** attempt)
    return ""


def _embed_texts(texts: list[str], *, model: str) -> list[list[float] | None]:
    """Embed a list of texts; return None for failed items."""
    client = _get_llm_client()
    results: list[list[float] | None] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i : i + _EMBED_BATCH]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(input=batch, model=model)
                results.extend([item.embedding for item in resp.data])
                break
            except Exception as exc:
                if attempt == 2:
                    log.warning("Embed batch failed: %s", exc)
                    results.extend([None] * len(batch))
                else:
                    time.sleep(2 ** attempt)
    return results


def _community_id(chunk_ids: list[str]) -> str:
    """Deterministic community ID based on sorted member chunk IDs."""
    key = "|".join(sorted(chunk_ids))
    return "comm_" + hashlib.sha1(key.encode()).hexdigest()[:16]


# ── Core algorithm ────────────────────────────────────────────────────────────

def load_bipartite_graph(
    driver: Driver,
    *,
    limit: int = 500_000,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Load CHUNK–KEYWORD edges into an edge list for igraph.

    Returns:
        chunk_ids: ordered list of unique CHUNK IDs (vertex labels)
        edges: list of (chunk_vertex_idx, keyword_vertex_idx) — NOTE: this
            returns only the chunk indices for the CHUNK projection.
    """
    with driver.session() as s:
        rows = s.run(_BIPARTITE_QUERY, limit=limit).data()

    chunk_set: dict[str, int] = {}
    keyword_set: dict[str, int] = {}

    # Pass 1: assign stable indices to all unique IDs
    for row in rows:
        cid = row["chunk_id"]
        kid = row["keyword_name"]
        if cid not in chunk_set:
            chunk_set[cid] = len(chunk_set)
        if kid not in keyword_set:
            keyword_set[kid] = len(keyword_set)

    n_chunks = len(chunk_set)
    n_keywords = len(keyword_set)

    # Pass 2: build edge list with stable keyword vertex offset
    edge_list: list[tuple[int, int]] = [
        (chunk_set[row["chunk_id"]], n_chunks + keyword_set[row["keyword_name"]])
        for row in rows
    ]

    chunk_ids = [None] * n_chunks
    for cid, idx in chunk_set.items():
        chunk_ids[idx] = cid

    return chunk_ids, edge_list, n_chunks, n_keywords


def run_leiden(
    chunk_ids: list[str],
    edge_list: list[tuple[int, int]],
    n_chunks: int,
    n_keywords: int,
    *,
    resolution: float = _LEIDEN_RESOLUTION,
    min_size: int = _MIN_COMMUNITY_SIZE,
) -> list[list[str]]:
    """Run Leiden community detection on the CHUNK projection.

    Projects the bipartite CHUNK-KEYWORD graph to a CHUNK-CHUNK co-mention
    graph, then runs Leiden. Returns list of community member lists (chunk IDs).
    """
    import igraph as ig  # lazy import — optional dep

    n_total = n_chunks + n_keywords
    g = ig.Graph(n=n_total, edges=edge_list, directed=False)

    # Project to CHUNK-only unipartite graph via shared keyword neighbours
    chunk_verts = list(range(n_chunks))
    # which=1 → True-type vertices = chunks (types[i]=True for i < n_chunks)
    proj = g.bipartite_projection(
        types=[i < n_chunks for i in range(n_total)],
        which=1,
    )

    # Leiden community detection
    partition = proj.community_leiden(
        objective_function="modularity",
        resolution=resolution,
        n_iterations=10,
    )

    communities: list[list[str]] = []
    for members in partition:
        if len(members) < min_size:
            continue
        communities.append([chunk_ids[i] for i in members])

    log.info(
        "Leiden: %d total communities (≥%d members), modularity=%.4f",
        len(communities), min_size, partition.modularity,
    )
    return communities, partition.modularity


# ── Public API ────────────────────────────────────────────────────────────────

def build_community_summaries(
    driver: Driver,
    *,
    resolution: float = _LEIDEN_RESOLUTION,
    min_size: int = _MIN_COMMUNITY_SIZE,
    max_summary_chunks: int = _MAX_SUMMARY_CHUNKS,
    limit: int = 500_000,
    llm_model: str | None = None,
    embed_model: str | None = None,
) -> CommunityReport:
    """Full B3 pipeline: Leiden detection + LLM summaries + Neo4j write.

    Args:
        driver: Open Neo4j driver.
        resolution: Leiden resolution parameter (default 1.0).
        min_size: Minimum community size to keep (default 5 chunks).
        max_summary_chunks: Max representative chunks fed to the LLM.
        limit: Max CHUNK-KEYWORD edges to load (for smoke tests reduce to ~10_000).
        llm_model: Model for summary generation (default: deepseek-chat).
        embed_model: Model for summary embedding (default: text-embedding-v4).
    """
    chat_model = llm_model or os.getenv("LLM_MODEL", "deepseek-chat")
    emb_model = embed_model or os.getenv("EMBED_LLM_MODEL", "text-embedding-v4")

    report = CommunityReport()
    t_start = time.time()

    # 1. Load graph
    log.info("Loading CHUNK-KEYWORD bipartite graph (limit=%d)…", limit)
    chunk_ids, edge_list, n_chunks, n_keywords = load_bipartite_graph(driver, limit=limit)
    report.chunks_loaded = n_chunks
    report.edges_loaded = len(edge_list)
    log.info("Loaded %d chunks, %d keywords, %d edges", n_chunks, n_keywords, len(edge_list))

    if not edge_list:
        report.errors.append("No CHUNK-KEYWORD edges found — run keyword extraction first")
        report.duration_seconds = time.time() - t_start
        return report

    # 2. Leiden
    log.info("Running Leiden (resolution=%.2f, min_size=%d)…", resolution, min_size)
    communities, overall_modularity = run_leiden(
        chunk_ids, edge_list, n_chunks, n_keywords,
        resolution=resolution, min_size=min_size,
    )
    report.communities_detected = len(communities)
    log.info("Detected %d communities above min_size=%d", len(communities), min_size)

    if not communities:
        report.errors.append("Leiden produced 0 communities — check CHUNK-KEYWORD density")
        report.duration_seconds = time.time() - t_start
        return report

    # 3. Fetch representative chunk texts
    all_community_results: list[CommunityResult] = []
    summaries_to_embed: list[str] = []
    comm_order: list[int] = []  # index into all_community_results for embedding

    for i, member_ids in enumerate(communities):
        sample_ids = member_ids[:max_summary_chunks]
        with driver.session() as s:
            text_rows = s.run(_CHUNK_TEXTS_QUERY, chunk_ids=sample_ids).data()
        texts = [r["text"][:400] for r in text_rows if r["text"]]

        summary = _generate_summary(texts, model=chat_model) if texts else ""
        comm_id = _community_id(member_ids)

        result = CommunityResult(
            community_id=comm_id,
            chunk_ids=member_ids,
            summary=summary,
            embedding=None,
            modularity=overall_modularity,
            size=len(member_ids),
        )
        all_community_results.append(result)

        if summary:
            summaries_to_embed.append(summary)
            comm_order.append(i)

        if (i + 1) % 10 == 0:
            log.info("Summarized %d / %d communities", i + 1, len(communities))

    # 4. Embed all summaries in bulk
    if summaries_to_embed:
        log.info("Embedding %d community summaries…", len(summaries_to_embed))
        embeddings = _embed_texts(summaries_to_embed, model=emb_model)
        for idx, emb in zip(comm_order, embeddings):
            all_community_results[idx].embedding = emb

    # 5. Write COMMUNITY nodes + IN_COMMUNITY edges
    ts_now = datetime.now(timezone.utc).isoformat()
    community_dicts = [
        {
            "id": r.community_id,
            "summary": r.summary,
            "embedding": r.embedding,
            "size": r.size,
            "modularity": round(r.modularity, 6),
            "ts": ts_now,
        }
        for r in all_community_results
    ]

    with driver.session() as s:
        s.run(_COMMUNITY_MERGE, communities=community_dicts).consume()
    report.communities_written = len(community_dicts)

    membership_dicts = [
        {"chunk_id": cid, "community_id": r.community_id}
        for r in all_community_results
        for cid in r.chunk_ids
    ]
    # Write in batches to avoid huge transactions
    for start in range(0, len(membership_dicts), 5_000):
        batch = membership_dicts[start : start + 5_000]
        with driver.session() as s:
            s.run(_IN_COMMUNITY_MERGE, memberships=batch).consume()
    report.memberships_written = len(membership_dicts)

    log.info(
        "Written %d communities, %d IN_COMMUNITY edges",
        report.communities_written, report.memberships_written,
    )
    report.duration_seconds = time.time() - t_start
    return report
