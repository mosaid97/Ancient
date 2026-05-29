"""Phase 1b orchestrator: classify native-text PAGE nodes (plan §6 Stage 1b).

Iterates over ``(:PAGE {mode: 'native_text'})`` nodes that don't already
carry a ``language`` property, runs :func:`apps.backend.lang.detect_language`
on ``PAGE.text``, and writes back ``language`` / ``scriptMix`` /
``kuntenMarks`` / ``langConfidence`` / ``langDetectionRule``.

OCR pages (``mode='ocr'``) are deliberately skipped — they have no
``text`` yet; Phase 3 will run the authoritative detector on the fused
OCR output and overwrite these properties.

The orchestrator is idempotent + restartable: callers pass
``recompute_existing=False`` (the default) and only PAGEs missing the
``language`` property are processed; passing ``True`` reclassifies every
native page (used after threshold tuning in the notebook).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from neo4j import Driver

from apps.backend.lang.detector import LanguageProfile, detect_language

logger = logging.getLogger(__name__)


# Pull every native page that needs (re)classification. ``$recompute=True``
# returns every native page; ``$recompute=False`` returns only those with
# no ``language`` set yet.
_SELECT_CYPHER = """
MATCH (p:PAGE)
WHERE p.mode = 'native_text'
  AND p.text IS NOT NULL
  AND ($recompute = true OR p.language IS NULL)
RETURN p.id AS page_id, p.text AS text, p.documentId AS document_id,
       p.tier AS tier, p.docPageIndex AS page_index
ORDER BY p.documentId, p.docPageIndex
"""


# Single-page write. ``MERGE`` isn't needed (the page already exists from
# Phase 1); we just ``SET`` the language fields. ``scriptMix`` is JSON
# because Neo4j can't store nested maps as a property (plan §5 + AGENTS
# .md §11 entry on editorialLayers).
_UPDATE_CYPHER = """
MATCH (p:PAGE {id: $page_id})
SET p.language = $language,
    p.scriptMix = $script_mix_json,
    p.kuntenMarks = $kunten_marks,
    p.langConfidence = $lang_confidence,
    p.langDetectionRule = $lang_detection_rule,
    p.langDetectionAt = timestamp()
RETURN p.id AS id
"""


@dataclass
class PageDecision:
    """Single page → language outcome (kept for notebook summaries + tests)."""

    page_id: str
    document_id: str
    tier: str
    page_index: int
    profile: LanguageProfile

    def to_dict(self) -> dict[str, Any]:
        return {
            "pageId": self.page_id,
            "documentId": self.document_id,
            "tier": self.tier,
            "pageIndex": self.page_index,
            **self.profile.to_dict(),
        }


@dataclass
class LangDetectReport:
    """Aggregate run report (suitable for ``json.dump``)."""

    pages_processed: int = 0
    pages_written: int = 0
    pages_skipped: int = 0
    by_language: dict[str, int] = field(default_factory=dict)
    by_rule: dict[str, int] = field(default_factory=dict)
    by_tier_language: dict[str, dict[str, int]] = field(default_factory=dict)
    mean_confidence: float = 0.0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    sample_decisions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_pages(
    driver: Driver,
    *,
    recompute_existing: bool = False,
    batch_size: int = 200,
    max_pages: int | None = None,
    sample_size: int = 5,
) -> LangDetectReport:
    """Run the Phase 1b detector over native pages in Neo4j.

    Args:
        driver: Open Neo4j driver.
        recompute_existing: If ``True``, re-classify every native page;
            else only pages with ``language IS NULL``.
        batch_size: Write batch size — every ``batch_size`` pages we
            consume the write transaction so a partial failure doesn't
            cost the whole run.
        max_pages: Optional cap (for notebook smoke tests).
        sample_size: How many per-page decisions to retain in the
            ``sample_decisions`` array of the report (for inspection).

    Returns:
        :class:`LangDetectReport`.
    """
    report = LangDetectReport()
    started = time.monotonic()

    decisions: list[PageDecision] = []
    confidences: list[float] = []

    with driver.session() as session:
        rows = list(session.run(_SELECT_CYPHER, recompute=recompute_existing))
        if max_pages is not None:
            rows = rows[:max_pages]

        for row in rows:
            page_id = row["page_id"]
            text = row.get("text")
            if not text:
                report.pages_skipped += 1
                continue

            profile = detect_language(text)
            decision = PageDecision(
                page_id=page_id,
                document_id=row.get("document_id") or "",
                tier=row.get("tier") or "",
                page_index=row.get("page_index") or 0,
                profile=profile,
            )
            decisions.append(decision)
            confidences.append(profile.confidence)

            report.pages_processed += 1
            report.by_language[profile.language] = (
                report.by_language.get(profile.language, 0) + 1
            )
            report.by_rule[profile.rule] = report.by_rule.get(profile.rule, 0) + 1
            tier_bucket = report.by_tier_language.setdefault(decision.tier or "?", {})
            tier_bucket[profile.language] = tier_bucket.get(profile.language, 0) + 1

        # Idempotent writes, batched. Each batch runs in its own tx.
        for start in range(0, len(decisions), batch_size):
            chunk = decisions[start : start + batch_size]
            try:
                with session.begin_transaction() as tx:
                    for d in chunk:
                        payload = d.profile.to_dict()
                        tx.run(
                            _UPDATE_CYPHER,
                            page_id=d.page_id,
                            language=payload["language"],
                            script_mix_json=json.dumps(
                                payload["scriptMix"], ensure_ascii=False
                            ),
                            kunten_marks=payload["kuntenMarks"],
                            lang_confidence=payload["langConfidence"],
                            lang_detection_rule=payload["langDetectionRule"],
                        ).consume()
                    tx.commit()
                    report.pages_written += len(chunk)
            except Exception as exc:  # noqa: BLE001
                msg = (
                    f"batch [{start}:{start + len(chunk)}] write failed "
                    f"({type(exc).__name__}): {exc}"
                )
                logger.exception(msg)
                report.errors.append(msg)

    if confidences:
        report.mean_confidence = round(sum(confidences) / len(confidences), 4)

    report.sample_decisions = [d.to_dict() for d in decisions[:sample_size]]
    report.duration_seconds = round(time.monotonic() - started, 3)
    return report


def language_summary(driver: Driver) -> dict[str, Any]:
    """Roll up post-detection language counts (used by the notebook).

    Returns three slices: by language (overall), by ``tier × language``,
    and by ``rule``. Skips pages that have never been classified — those
    show up under the ``language='(unset)'`` key so it's obvious in the
    notebook how much of the corpus the detector hasn't touched yet.
    """
    cypher = """
    MATCH (p:PAGE)
    WITH coalesce(p.language, '(unset)') AS language,
         coalesce(p.tier, '(unset)') AS tier,
         coalesce(p.langDetectionRule, '(unset)') AS rule
    RETURN language, tier, rule, count(*) AS n
    """
    by_language: dict[str, int] = {}
    by_tier_language: dict[str, dict[str, int]] = {}
    by_rule: dict[str, int] = {}
    with driver.session() as session:
        for row in session.run(cypher):
            lang = row["language"]
            tier = row["tier"]
            rule = row["rule"]
            n = row["n"]
            by_language[lang] = by_language.get(lang, 0) + n
            tier_bucket = by_tier_language.setdefault(tier, {})
            tier_bucket[lang] = tier_bucket.get(lang, 0) + n
            by_rule[rule] = by_rule.get(rule, 0) + n
    return {
        "by_language": by_language,
        "by_tier_language": by_tier_language,
        "by_rule": by_rule,
    }
