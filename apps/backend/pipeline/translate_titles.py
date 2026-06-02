"""Document title translation pipeline — translates DOCUMENT.title into 4 languages.

Reads every DOCUMENT node whose title has not yet been translated and writes:
  titleZh  — modern Chinese (simplified)
  titleEn  — English
  titleJa  — Japanese
  titleAr  — Arabic

The original ``title`` field is preserved unchanged.

Public API
----------
translate_document_titles(driver, *, ...) -> TitleTranslationReport
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver
from openai import OpenAI

from apps.backend.llm.silra import get_silra_client

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a scholarly translator specialising in East Asian history. "
    "Given a document title (which may be in Classical Chinese, modern Chinese, Japanese, or another language), "
    "translate it into the four requested languages. "
    "Return a JSON object with exactly these keys: zh, en, ja, ar. "
    "zh = modern simplified Chinese, en = English, ja = Japanese, ar = Arabic. "
    "Use concise, accurate scholarly translations. Output ONLY valid JSON, no markdown fences."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TitleTranslationResult:
    doc_id: str
    original_title: str
    status: str          # 'ok' | 'failed' | 'skipped'
    title_zh: str = ""
    title_en: str = ""
    title_ja: str = ""
    title_ar: str = ""
    error: str | None = None


@dataclass
class TitleTranslationReport:
    total: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    started_at: str = ""
    ended_at: str = ""
    results: list[TitleTranslationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "ok": self.ok,
            "failed": self.failed,
            "skipped": self.skipped,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

_FETCH_DOCS = """
MATCH (d:DOCUMENT)
WHERE d.title IS NOT NULL AND d.title <> ''
  AND ($recompute OR d.titleEn IS NULL)
RETURN d.id AS doc_id, d.title AS title
ORDER BY d.id
"""

_WRITE_TITLES = """
MATCH (d:DOCUMENT {id: $doc_id})
SET
    d.titleZh          = $title_zh,
    d.titleEn          = $title_en,
    d.titleJa          = $title_ja,
    d.titleAr          = $title_ar,
    d.titleTranslatedAt = $ts
"""

_WRITE_FAILED = """
MATCH (d:DOCUMENT {id: $doc_id})
SET d.titleTranslationError = $error,
    d.titleTranslatedAt     = $ts
"""


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------

import json
import re


def _translate_title(
    title: str,
    client: OpenAI,
    model: str,
) -> dict[str, str]:
    """Call deepseek-chat to translate a document title into zh/en/ja/ar."""
    prompt = f'Translate this document title into modern simplified Chinese (zh), English (en), Japanese (ja), and Arabic (ar).\n\nTitle: "{title}"'
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.1,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences if present
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        data = json.loads(m.group())
    else:
        data = json.loads(raw)
    return {
        "zh": str(data.get("zh", "")),
        "en": str(data.get("en", "")),
        "ja": str(data.get("ja", "")),
        "ar": str(data.get("ar", "")),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def translate_document_titles(
    driver: Driver,
    *,
    recompute: bool = False,
    client: OpenAI | None = None,
    model: str | None = None,
) -> TitleTranslationReport:
    """Translate all DOCUMENT.title fields into four languages.

    Args:
        driver: Open Neo4j driver.
        recompute: If True, re-translate already-done documents.
        client: Optional Silra client.
        model: Override chat model.

    Returns:
        :class:`TitleTranslationReport` with aggregate counts.
    """
    c = client or get_silra_client()
    m = model or os.getenv("CHAT_LLM_MODEL", "deepseek-chat")

    report = TitleTranslationReport(started_at=datetime.now(timezone.utc).isoformat())

    with driver.session() as s:
        rows = s.run(_FETCH_DOCS, recompute=recompute).data()

    report.total = len(rows)
    log.info("translate_document_titles: %d documents to translate", report.total)

    for row in rows:
        doc_id = row["doc_id"]
        title = row.get("title") or ""

        if not title.strip():
            report.skipped += 1
            report.results.append(TitleTranslationResult(
                doc_id=doc_id, original_title=title, status="skipped",
            ))
            continue

        try:
            translations = _translate_title(title, c, m)
            ts = datetime.now(timezone.utc).isoformat()
            with driver.session() as s:
                s.run(
                    _WRITE_TITLES,
                    doc_id=doc_id,
                    title_zh=translations["zh"],
                    title_en=translations["en"],
                    title_ja=translations["ja"],
                    title_ar=translations["ar"],
                    ts=ts,
                ).consume()

            log.info("Translated [%s] → en=%r", title, translations["en"])
            report.ok += 1
            report.results.append(TitleTranslationResult(
                doc_id=doc_id,
                original_title=title,
                status="ok",
                title_zh=translations["zh"],
                title_en=translations["en"],
                title_ja=translations["ja"],
                title_ar=translations["ar"],
            ))
        except Exception as exc:
            log.warning("translate_document_titles: doc=%s title=%r error=%s", doc_id, title, exc)
            ts = datetime.now(timezone.utc).isoformat()
            with driver.session() as s:
                s.run(_WRITE_FAILED, doc_id=doc_id, error=str(exc)[:500], ts=ts).consume()
            report.failed += 1
            report.results.append(TitleTranslationResult(
                doc_id=doc_id, original_title=title, status="failed", error=str(exc),
            ))

    report.ended_at = datetime.now(timezone.utc).isoformat()
    log.info(
        "translate_document_titles: done — ok=%d failed=%d skipped=%d",
        report.ok, report.failed, report.skipped,
    )
    return report
