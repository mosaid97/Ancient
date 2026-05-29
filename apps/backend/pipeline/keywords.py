"""Keyword extraction pipeline — Phase 6 KG construction (plan §6 Stage 6).

For every CHUNK node with sufficient CJK text, sends the chunk to
``deepseek-chat`` and extracts 3–8 named entities / key concepts.
Each keyword is MERGEd as a ``(:KEYWORD)`` node and linked to the chunk
via ``(:CHUNK)-[:MENTION {weight, mentionedAt}]->(:KEYWORD)``.

KEYWORD node properties (camelCase per AGENTS.md §4):
  name, type, frequency, keywordAt

MENTION relationship properties:
  weight, mentionedAt

CHUNK properties written:
  mentionStatus ('ok' | 'failed' | 'skipped'), mentionAt, mentionKeywordCount

Keyword types
-------------
PERSON, PLACE, DYNASTY, OFFICIAL_TITLE, EVENT, TEXT_TITLE,
CONCEPT, LEGAL_TERM, ERA_NAME, OTHER

Pipeline gating
---------------
Chunks with ``mentionStatus IS NULL`` are processed by default.
Set ``recompute=True`` to reprocess already-extracted chunks.
Chunks with ``charCount < min_chars`` are stamped ``'skipped'``.
Pages with language ``unknown`` or ``None`` are skipped silently.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver

from apps.backend.llm.silra import chat_completion, get_silra_client

log = logging.getLogger(__name__)

# Minimum chunk characters to attempt extraction (too-short chunks yield junk).
_MIN_CHARS = 50

# Accepted keyword types — anything the LLM returns outside this set is
# coerced to "OTHER" rather than rejected, so we never silently drop keywords.
_VALID_TYPES: frozenset[str] = frozenset({
    "PERSON",
    "PLACE",
    "DYNASTY",
    "OFFICIAL_TITLE",
    "EVENT",
    "TEXT_TITLE",
    "CONCEPT",
    "LEGAL_TERM",
    "ERA_NAME",
    "OTHER",
})

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "你是古代中國知識圖譜的建構助手，專門處理唐代歷史文獻。\n"
    "你的任務是從文本片段中提取關鍵詞，包含人名、地名、朝代、官職、"
    "事件、典籍名、法律術語、年號及歷史概念。\n"
    "只返回 JSON 陣列，不要包含任何解釋或 markdown 代碼塊。"
)

_USER_TEMPLATE = (
    "請從以下文本中提取 3–8 個關鍵詞。\n"
    "對每個關鍵詞返回：\n"
    '- "name": 繁體中文標準形式（如原文為簡體請轉換）\n'
    '- "type": 以下之一：PERSON | PLACE | DYNASTY | OFFICIAL_TITLE | '
    "EVENT | TEXT_TITLE | CONCEPT | LEGAL_TERM | ERA_NAME | OTHER\n"
    '- "weight": 0.0–1.0，該詞在此段文本中的重要程度\n\n'
    "只返回 JSON 陣列，例如：\n"
    '[{{"name":"唐太宗","type":"PERSON","weight":0.95}}]\n\n'
    "文本：\n{text}"
)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _parse_keywords(raw: str) -> list[dict[str, Any]]:
    """Parse the LLM JSON response into a list of keyword dicts.

    Tolerates:
    - Markdown fences (```json … ```)
    - Trailing commas
    - Leading/trailing whitespace
    """
    text = raw.strip()
    # Strip markdown code fences if present.
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Find the first JSON array.
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        # Response may be truncated before the closing ].
        # Extract all complete {...} objects and wrap them.
        objects = re.findall(r'\{[^{}]+\}', text, re.DOTALL)
        if not objects:
            raise ValueError(f"No JSON array found in LLM response: {raw[:200]!r}")
        array_str = "[" + ",".join(objects) + "]"
    else:
        array_str = m.group(0)
    try:
        items = json.loads(array_str)
    except json.JSONDecodeError:
        # Attempt to fix trailing commas and then salvage complete objects.
        fixed = re.sub(r",\s*([}\]])", r"\1", array_str)
        try:
            items = json.loads(fixed)
        except json.JSONDecodeError:
            objects = re.findall(r'\{[^{}]+\}', array_str, re.DOTALL)
            if not objects:
                raise
            items = json.loads("[" + ",".join(objects) + "]")

    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        ktype = str(item.get("type", "OTHER")).strip().upper()
        if ktype not in _VALID_TYPES:
            ktype = "OTHER"
        try:
            weight = float(item.get("weight", 0.7))
            weight = max(0.0, min(1.0, weight))
        except (TypeError, ValueError):
            weight = 0.7
        result.append({"name": name, "type": ktype, "weight": weight})
    return result


def extract_keywords_for_chunk(
    text: str,
    *,
    client=None,
    model: str | None = None,
    max_keywords: int = 8,
) -> list[dict[str, Any]]:
    """Call the LLM to extract keywords from a single chunk of text.

    Args:
        text: The chunk text (should be CJK, ≥ _MIN_CHARS chars).
        client: Silra OpenAI client (built if None).
        model: Chat model override (defaults to CHAT_LLM_MODEL env var).
        max_keywords: Maximum keywords to keep (truncates LLM output).

    Returns:
        List of dicts with keys: name, type, weight.

    Raises:
        ValueError: If the LLM response cannot be parsed as JSON.
        Exception: Propagates Silra API errors after retries.
    """
    c = client or get_silra_client()
    user_msg = _USER_TEMPLATE.format(text=text[:800])
    resp = chat_completion(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        model=model,
        client=c,
        temperature=0.0,
        max_tokens=1024,
    )
    raw = (resp.choices[0].message.content or "").strip()
    keywords = _parse_keywords(raw)
    return keywords[:max_keywords]


# ---------------------------------------------------------------------------
# Neo4j Cypher
# ---------------------------------------------------------------------------

_CHUNK_QUERY = """
MATCH (c:CHUNK)
WHERE (
    ($recompute AND c.mentionStatus = 'ok')
    OR ($recompute_failed AND c.mentionStatus = 'failed')
    OR ($recompute_skipped AND c.mentionStatus = 'skipped')
    OR (NOT $recompute AND NOT $recompute_failed AND NOT $recompute_skipped
        AND c.mentionStatus IS NULL)
  )
  AND c.charCount >= $min_chars
  AND c.text IS NOT NULL
RETURN c.id AS chunk_id, c.text AS text, c.language AS language
ORDER BY c.id
LIMIT $batch
"""

_KEYWORD_UPSERT = """
UNWIND $keywords AS kw
MERGE (k:KEYWORD {name: kw.name})
ON CREATE SET
  k.type        = kw.type,
  k.frequency   = 1,
  k.keywordAt   = kw.ts
ON MATCH SET
  k.frequency   = coalesce(k.frequency, 0) + 1,
  k.type        = CASE WHEN k.type IS NULL THEN kw.type ELSE k.type END
WITH k, kw
MATCH (c:CHUNK {id: kw.chunk_id})
MERGE (c)-[m:MENTION]->(k)
ON CREATE SET
  m.weight      = kw.weight,
  m.mentionedAt = kw.ts
ON MATCH SET
  m.weight      = kw.weight,
  m.mentionedAt = kw.ts
"""

_STAMP_CHUNK_OK = """
MATCH (c:CHUNK {id: $chunk_id})
SET c.mentionStatus       = 'ok',
    c.mentionAt           = $ts,
    c.mentionKeywordCount = $count
"""

_STAMP_CHUNK_FAIL = """
MATCH (c:CHUNK {id: $chunk_id})
SET c.mentionStatus = 'failed',
    c.mentionError  = $error,
    c.mentionAt     = $ts
"""

_STAMP_CHUNK_SKIP = """
MATCH (c:CHUNK {id: $chunk_id})
SET c.mentionStatus = 'skipped',
    c.mentionAt     = $ts
"""


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


class KeywordRunReport:
    """Thread-safe summary of a keyword extraction run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.chunks_total: int = 0
        self.chunks_ok: int = 0
        self.chunks_failed: int = 0
        self.chunks_skipped: int = 0
        self.keywords_extracted: int = 0
        self.keywords_unique: int = 0
        self.duration_seconds: float = 0.0
        self.errors: list[str] = []

    def add_ok(self, kw_count: int) -> None:
        with self._lock:
            self.chunks_ok += 1
            self.keywords_extracted += kw_count

    def add_failed(self, error: str) -> None:
        with self._lock:
            self.chunks_failed += 1
            if len(self.errors) < 30:
                self.errors.append(error)

    def add_skipped(self) -> None:
        with self._lock:
            self.chunks_skipped += 1

    def add_total(self, n: int) -> None:
        with self._lock:
            self.chunks_total += n

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_total": self.chunks_total,
            "chunks_ok": self.chunks_ok,
            "chunks_failed": self.chunks_failed,
            "chunks_skipped": self.chunks_skipped,
            "keywords_extracted": self.keywords_extracted,
            "keywords_unique": self.keywords_unique,
            "duration_seconds": round(self.duration_seconds, 2),
            "errors": self.errors[:30],
        }


# ---------------------------------------------------------------------------
# Per-chunk worker (called from thread pool)
# ---------------------------------------------------------------------------


def _process_chunk(
    row: dict[str, Any],
    *,
    driver: Driver,
    client: Any,
    model: str | None,
    skip_languages: frozenset[str],
    report: KeywordRunReport,
) -> None:
    """Process one chunk: LLM call → Neo4j MERGE.  Runs in a worker thread.

    All Neo4j writes for a single chunk are batched into ONE session to avoid
    triggering Neo4j's AuthenticationRateLimit under high thread concurrency.
    """
    chunk_id = row["chunk_id"]
    text = row.get("text") or ""
    language = row.get("language") or "unknown"
    ts_now = datetime.now(timezone.utc).isoformat()

    # Fast path: no LLM call needed for skipped languages.
    if language in skip_languages:
        with driver.session() as s:
            s.run(_STAMP_CHUNK_SKIP, chunk_id=chunk_id, ts=ts_now).consume()
        report.add_skipped()
        return

    # LLM call (outside Neo4j session — can be long).
    try:
        keywords = extract_keywords_for_chunk(text, client=client, model=model)
    except Exception as exc:
        log.warning("keyword extraction failed chunk=%s: %s", chunk_id, exc)
        with driver.session() as s:
            s.run(_STAMP_CHUNK_FAIL, chunk_id=chunk_id,
                  error=str(exc)[:500], ts=ts_now).consume()
        report.add_failed(f"{chunk_id}: {exc}")
        return

    if not keywords:
        with driver.session() as s:
            s.run(_STAMP_CHUNK_SKIP, chunk_id=chunk_id, ts=ts_now).consume()
        report.add_skipped()
        return

    kw_rows = [
        {"name": kw["name"], "type": kw["type"], "weight": kw["weight"],
         "chunk_id": chunk_id, "ts": ts_now}
        for kw in keywords
    ]

    # Single session for all writes: KEYWORD MERGE + MENTION + chunk stamp.
    try:
        with driver.session() as s:
            s.run(_KEYWORD_UPSERT, keywords=kw_rows).consume()
            s.run(_STAMP_CHUNK_OK, chunk_id=chunk_id,
                  ts=ts_now, count=len(keywords)).consume()
        report.add_ok(len(keywords))
    except Exception as exc:
        log.error("neo4j write failed chunk=%s: %s", chunk_id, exc)
        try:
            with driver.session() as s:
                s.run(_STAMP_CHUNK_FAIL, chunk_id=chunk_id,
                      error=str(exc)[:500], ts=ts_now).consume()
        except Exception:
            pass
        report.add_failed(f"neo4j {chunk_id}: {exc}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_keyword_extraction(
    driver: Driver,
    *,
    model: str | None = None,
    batch_size: int = 200,
    max_chunks: int | None = None,
    min_chars: int = _MIN_CHARS,
    recompute: bool = False,
    skip_languages: frozenset[str] = frozenset({"unknown"}),
    max_workers: int = 1,
    recompute_failed: bool = False,
    recompute_skipped: bool = False,
) -> KeywordRunReport:
    """Extract keywords from all eligible CHUNK nodes and wire KEYWORD graph.

    Each chunk triggers one LLM call to ``deepseek-chat``.  With
    ``max_workers > 1`` the calls run in a thread pool (the Neo4j driver and
    OpenAI client are both thread-safe).

    Pagination note: the query always uses ``SKIP 0`` because the
    ``mentionStatus IS NULL`` filter shrinks as chunks are stamped; a fixed
    SKIP would silently skip over unprocessed chunks.

    Args:
        driver: Open Neo4j driver.
        model: Chat model override. Defaults to CHAT_LLM_MODEL env var.
        batch_size: Neo4j fetch size per loop iteration.
        max_chunks: Stop after processing this many chunks. None = all.
        min_chars: Minimum chunk character count to attempt extraction.
        recompute: If True, reprocess chunks with mentionStatus = 'ok'.
        recompute_failed: If True, retry only chunks with mentionStatus = 'failed'.
        recompute_skipped: If True, retry chunks with mentionStatus = 'skipped'.
            Automatically sets skip_languages=frozenset() so language-unknown
            chunks are attempted rather than re-skipped.
        skip_languages: Page-language values to silently skip (no LLM call).
        max_workers: Thread-pool size. 1 = sequential (default).
            8–12 recommended for Silra; tune down if you hit rate limits.

    Returns:
        :class:`KeywordRunReport` with run statistics.
    """
    # Each worker gets its own LLM client so connections aren't shared.
    # When retrying skipped chunks, don't re-skip them by language.
    effective_skip_languages = frozenset() if recompute_skipped else skip_languages

    clients = [get_silra_client() for _ in range(max(1, max_workers))]
    report = KeywordRunReport()
    t_start = time.time()
    batch_num = 0

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        while True:
            with driver.session() as s:
                rows = s.run(
                    _CHUNK_QUERY,
                    recompute=recompute,
                    recompute_failed=recompute_failed,
                    recompute_skipped=recompute_skipped,
                    min_chars=min_chars,
                    batch=batch_size,
                ).data()

            if not rows:
                break

            report.add_total(len(rows))

            futures = {
                executor.submit(
                    _process_chunk,
                    row,
                    driver=driver,
                    client=clients[i % len(clients)],
                    model=model,
                    skip_languages=effective_skip_languages,
                    report=report,
                ): row["chunk_id"]
                for i, row in enumerate(rows)
            }

            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    chunk_id = futures[fut]
                    log.error("unhandled worker error chunk=%s: %s", chunk_id, exc)

            batch_num += 1
            log.info(
                "KG batch=%d rows=%d ok=%d failed=%d skipped=%d total=%d",
                batch_num,
                len(rows),
                report.chunks_ok,
                report.chunks_failed,
                report.chunks_skipped,
                report.chunks_total,
            )

            if max_chunks is not None and report.chunks_total >= max_chunks:
                break

    report.duration_seconds = time.time() - t_start

    try:
        with driver.session() as s:
            r = s.run("MATCH (k:KEYWORD) RETURN count(k) AS n").single()
            report.keywords_unique = r["n"] if r else 0
    except Exception:
        pass

    return report


def keyword_summary(driver: Driver) -> dict[str, Any]:
    """Return corpus-wide keyword extraction statistics.

    Returns:
        Dict with keys: chunks_total, chunks_ok, chunks_failed, chunks_skipped,
        keywords_total, top_keywords, type_distribution.
    """
    with driver.session() as s:
        chunk_stats = s.run("""
            MATCH (c:CHUNK)
            RETURN
              count(c) AS total,
              count(CASE WHEN c.mentionStatus = 'ok' THEN 1 END) AS ok,
              count(CASE WHEN c.mentionStatus = 'failed' THEN 1 END) AS failed,
              count(CASE WHEN c.mentionStatus = 'skipped' THEN 1 END) AS skipped
        """).single()

        kw_total = s.run("MATCH (k:KEYWORD) RETURN count(k) AS n").single()

        top_kws = s.run("""
            MATCH (k:KEYWORD)
            RETURN k.name AS name, k.type AS type, k.frequency AS freq
            ORDER BY k.frequency DESC
            LIMIT 20
        """).data()

        type_dist = s.run("""
            MATCH (k:KEYWORD)
            RETURN k.type AS type, count(k) AS n
            ORDER BY n DESC
        """).data()

    return {
        "chunks_total": chunk_stats["total"] if chunk_stats else 0,
        "chunks_ok": chunk_stats["ok"] if chunk_stats else 0,
        "chunks_failed": chunk_stats["failed"] if chunk_stats else 0,
        "chunks_skipped": chunk_stats["skipped"] if chunk_stats else 0,
        "keywords_total": kw_total["n"] if kw_total else 0,
        "top_keywords": top_kws,
        "type_distribution": type_dist,
    }
