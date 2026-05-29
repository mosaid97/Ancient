"""Translation Agent pipeline orchestrator — Phase 6 (plan §6 Stage 6).

Tier-aware batch runner that processes CHUNK nodes:
- ``tier == 'primary'``: full word → paragraph → review pipeline
  → writes ``CHUNK.textCanonical``, ``CHUNK.textVernacular``,
    ``CHUNK.textVernacularJa`` (if ja/kanbun), ``CHUNK.editorialLayerType``
- ``tier == 'secondary'``: keyword + concept extraction only (skips word.py
  + paragraph.py + review.py, ~70% LLM cost saving per plan)
  → writes ``CHUNK.textCanonical`` (normalized only), ``CHUNK.semanticConcepts``

Idempotency gate: ``CHUNK.translationStatus`` ∈ {None, 'ok', 'failed', 'skipped'}
Re-runs only NULL/failed chunks by default.

Public API
----------
TranslationReport          — run report dataclass
TranslatePageResult        — per-page result
translate_chunks(driver, *, ...) -> TranslationReport
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver
from openai import OpenAI

from apps.backend.agents.translation.paragraph import translate_paragraph
from apps.backend.agents.translation.review import review_translation
from apps.backend.agents.translation.word import WordAnalysisResult, analyze_words
from apps.backend.llm.silra import ANCIENT_CHINA_SYSTEM_PROMPT, get_silra_client
from apps.backend.normalize.pipeline import normalize_canonical

log = logging.getLogger(__name__)

_DEFAULT_BATCH = 50


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TranslatePageResult:
    """Result for a single chunk translation."""

    chunk_id: str
    page_id: str
    tier: str
    language: str
    status: str          # 'ok' | 'failed' | 'skipped'
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class TranslationReport:
    """Aggregate report for a translate_chunks run."""

    total: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    started_at: str = ""
    ended_at: str = ""
    results: list[TranslatePageResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "ok": self.ok,
            "failed": self.failed,
            "skipped": self.skipped,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

_FETCH_CHUNKS = """
MATCH (ch:CHUNK)-[:HAS]-(p:PAGE)
WHERE ($recompute OR ch.translationStatus IS NULL OR ch.translationStatus = 'failed')
  AND ($tier_filter IS NULL OR p.tier = $tier_filter)
  AND p.fusionStatus = 'ok'
OPTIONAL MATCH (sec:SECTION)-[:INCLUDE]->(p)
OPTIONAL MATCH (doc:DOCUMENT)-[:CONSIST_OF]->(:CHAPTER)-[:INCLUDE]->(sec)
RETURN
    ch.id          AS chunk_id,
    ch.text        AS text,
    p.id           AS page_id,
    p.tier         AS tier,
    p.language     AS language,
    p.detectedEra  AS era,
    doc.id         AS doc_id,
    ch.editorialLayerType AS editorial_layer_type
ORDER BY p.tier DESC, p.id, ch.id
LIMIT $limit
"""

_WRITE_PRIMARY = """
MATCH (ch:CHUNK {id: $chunk_id})
SET
    ch.textCanonical        = $text_canonical,
    ch.textVernacular       = $text_vernacular,
    ch.textVernacularJa     = $text_vernacular_ja,
    ch.editorialLayerType   = $editorial_layer_type,
    ch.translationStatus    = 'ok',
    ch.translationAt        = $ts
"""

_WRITE_SECONDARY = """
MATCH (ch:CHUNK {id: $chunk_id})
SET
    ch.textCanonical     = $text_canonical,
    ch.semanticConcepts  = $semantic_concepts,
    ch.translationStatus = 'ok',
    ch.translationAt     = $ts
"""

_WRITE_FAILED = """
MATCH (ch:CHUNK {id: $chunk_id})
SET ch.translationStatus = 'failed',
    ch.translationError  = $error,
    ch.translationAt     = $ts
"""


# ---------------------------------------------------------------------------
# Secondary tier: concept extraction
# ---------------------------------------------------------------------------

_CONCEPT_SYSTEM = (
    ANCIENT_CHINA_SYSTEM_PROMPT
    + "\n\n從以下古籍文本中提取3-8個核心語義概念（人名、地名、事件、制度、法律術語、思想概念等）。"
    "以 JSON 列表回答，格式：[\"concept1\", \"concept2\", ...]"
    "只輸出 JSON，不含其他文字。"
)


def _extract_concepts(
    text: str,
    client: OpenAI,
    model: str,
) -> list[str]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CONCEPT_SYSTEM},
                {"role": "user", "content": text[:1500]},
            ],
            max_tokens=200,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        import re
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        return []
    except Exception as exc:
        log.debug("_extract_concepts failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Per-chunk processing
# ---------------------------------------------------------------------------

def _process_primary(
    row: dict[str, Any],
    driver: Driver,
    client: OpenAI,
    model: str,
) -> TranslatePageResult:
    chunk_id = row["chunk_id"]
    text = row.get("text") or ""
    language = row.get("language") or "zh-classical"
    era = row.get("era")
    tier = row.get("tier") or "primary"
    editorial_layer = row.get("editorial_layer_type") or "pure-source"

    if not text.strip():
        ts = datetime.now(timezone.utc).isoformat()
        with driver.session() as s:
            s.run(
                "MATCH (ch:CHUNK {id: $id}) SET ch.translationStatus='skipped', ch.translationAt=$ts",
                id=chunk_id, ts=ts,
            ).consume()
        return TranslatePageResult(
            chunk_id=chunk_id, page_id=row.get("page_id") or "",
            tier=tier, language=language, status="skipped",
        )

    try:
        word_result = analyze_words(
            text, language, era, tier, driver, client=client, model=model
        )
        para_result = translate_paragraph(
            word_result, language, driver, client=client, model=model
        )
        review_result = review_translation(
            word_result, para_result, language, client=client, model=model
        )

        ts = datetime.now(timezone.utc).isoformat()
        with driver.session() as s:
            s.run(
                _WRITE_PRIMARY,
                chunk_id=chunk_id,
                text_canonical=word_result.text_canonical,
                text_vernacular=review_result.text_final,
                text_vernacular_ja=para_result.text_vernacular_ja,
                editorial_layer_type=editorial_layer,
                ts=ts,
            ).consume()

        total_p = para_result.prompt_tokens + review_result.prompt_tokens
        total_c = para_result.completion_tokens + review_result.completion_tokens
        return TranslatePageResult(
            chunk_id=chunk_id, page_id=row.get("page_id") or "",
            tier=tier, language=language, status="ok",
            prompt_tokens=total_p, completion_tokens=total_c,
        )
    except Exception as exc:
        log.warning("_process_primary: chunk=%s error=%s", chunk_id, exc)
        ts = datetime.now(timezone.utc).isoformat()
        with driver.session() as s:
            s.run(_WRITE_FAILED, chunk_id=chunk_id, error=str(exc)[:500], ts=ts).consume()
        return TranslatePageResult(
            chunk_id=chunk_id, page_id=row.get("page_id") or "",
            tier=tier, language=language, status="failed", error=str(exc),
        )


def _process_secondary(
    row: dict[str, Any],
    driver: Driver,
    client: OpenAI,
    model: str,
) -> TranslatePageResult:
    chunk_id = row["chunk_id"]
    text = row.get("text") or ""
    language = row.get("language") or "zh-modern"
    era = row.get("era")
    tier = "secondary"

    if not text.strip():
        ts = datetime.now(timezone.utc).isoformat()
        with driver.session() as s:
            s.run(
                "MATCH (ch:CHUNK {id: $id}) SET ch.translationStatus='skipped', ch.translationAt=$ts",
                id=chunk_id, ts=ts,
            ).consume()
        return TranslatePageResult(
            chunk_id=chunk_id, page_id=row.get("page_id") or "",
            tier=tier, language=language, status="skipped",
        )

    try:
        # Just normalize + concept extraction for secondary
        norm_result = normalize_canonical(
            text,
            lang=language.replace("zh-", "zh").replace("kanbun", "ja"),
            era=era or "",
            apply_loan=False,
        )
        concepts = _extract_concepts(norm_result.canonical, client, model)

        ts = datetime.now(timezone.utc).isoformat()
        with driver.session() as s:
            s.run(
                _WRITE_SECONDARY,
                chunk_id=chunk_id,
                text_canonical=norm_result.canonical,
                semantic_concepts=json.dumps(concepts, ensure_ascii=False),
                ts=ts,
            ).consume()

        return TranslatePageResult(
            chunk_id=chunk_id, page_id=row.get("page_id") or "",
            tier=tier, language=language, status="ok",
        )
    except Exception as exc:
        log.warning("_process_secondary: chunk=%s error=%s", chunk_id, exc)
        ts = datetime.now(timezone.utc).isoformat()
        with driver.session() as s:
            s.run(_WRITE_FAILED, chunk_id=chunk_id, error=str(exc)[:500], ts=ts).consume()
        return TranslatePageResult(
            chunk_id=chunk_id, page_id=row.get("page_id") or "",
            tier=tier, language=language, status="failed", error=str(exc),
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def translate_chunks(
    driver: Driver,
    *,
    limit: int = _DEFAULT_BATCH,
    tier_filter: str | None = None,
    recompute: bool = False,
    client: OpenAI | None = None,
    model: str | None = None,
) -> TranslationReport:
    """Translate a batch of CHUNK nodes.

    Args:
        driver: Open Neo4j driver.
        limit: Maximum chunks to process per call.
        tier_filter: If set, only process ``'primary'`` or ``'secondary'``.
        recompute: If True, re-process already-translated chunks.
        client: Optional Silra client.
        model: Override chat model.

    Returns:
        :class:`TranslationReport` with aggregate counts.
    """
    c = client or get_silra_client()
    m = model or os.getenv("CHAT_LLM_MODEL", "deepseek-chat")

    report = TranslationReport(started_at=datetime.now(timezone.utc).isoformat())

    with driver.session() as s:
        rows = s.run(
            _FETCH_CHUNKS,
            recompute=recompute,
            tier_filter=tier_filter,
            limit=limit,
        ).data()

    report.total = len(rows)
    log.info("translate_chunks: %d chunks to process (tier=%s)", report.total, tier_filter)

    for row in rows:
        tier = (row.get("tier") or "secondary").lower()
        if tier == "primary":
            result = _process_primary(row, driver, c, m)
        else:
            result = _process_secondary(row, driver, c, m)

        report.results.append(result)
        if result.status == "ok":
            report.ok += 1
        elif result.status == "failed":
            report.failed += 1
        else:
            report.skipped += 1
        report.prompt_tokens += result.prompt_tokens
        report.completion_tokens += result.completion_tokens

    report.ended_at = datetime.now(timezone.utc).isoformat()
    log.info(
        "translate_chunks: done — ok=%d failed=%d skipped=%d",
        report.ok, report.failed, report.skipped,
    )
    return report
