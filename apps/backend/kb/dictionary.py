"""Bilingual Dictionary KB — Phase 6 (plan §6 Stage 6).

Wraps ``(:DICTIONARY_ENTRY)`` Neo4j nodes for lookup by the Translation Agent.
Also seeds the dictionary from ``data/seeds/dictionary_seed.jsonl``.

``DICTIONARY_ENTRY`` properties (plan §5):
  id, term, readings, meaning, dynasty, source, language, createdFromCorrectionId

Public API
----------
seed_dictionary(driver, path) -> int          (upserts from JSONL seed)
lookup_term(driver, term, language) -> list[DictEntry]
upsert_entry(driver, entry: DictEntry) -> str
DictEntry                                     — data class
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neo4j import Driver

log = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).parents[3] / "data" / "seeds" / "dictionary_seed.jsonl"
_SEED_PATHS_BY_LANG: dict[str, Path] = {
    "zh": _SEED_PATH,
    "ja": Path(__file__).parents[3] / "data" / "seeds" / "dictionary_seed_ja.jsonl",
    "en": Path(__file__).parents[3] / "data" / "seeds" / "dictionary_seed_en.jsonl",
    "ar": Path(__file__).parents[3] / "data" / "seeds" / "dictionary_seed_ar.jsonl",
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class DictEntry:
    """One bilingual dictionary entry.

    Attributes:
        term: The canonical CJK term (traditional form).
        meaning: Definition in modern Chinese (always zh for consistency).
        language: Source language of the term — ``'zh'`` or ``'ja'``.
        readings: Pronunciation(s): pinyin for zh, kana for ja.
        dynasty: Dynasty context (e.g. ``'Tang'``, ``'Han'``).
        source: Citation (e.g. ``'Kangxi'``, ``'JMdict'``, ``'HITL'``).
        created_from_correction_id: Audit link to a CORRECTION node.
    """

    term: str
    meaning: str
    language: str = "zh"
    readings: list[str] = field(default_factory=list)
    dynasty: str = ""
    source: str = ""
    created_from_correction_id: str | None = None

    @property
    def id(self) -> str:
        return "dict_" + hashlib.sha1(
            f"{self.language}::{self.term}".encode()
        ).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "term": self.term,
            "meaning": self.meaning,
            "language": self.language,
            "readings": self.readings,
            "dynasty": self.dynasty,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

_UPSERT = """
MERGE (d:DICTIONARY_ENTRY {id: $id})
ON CREATE SET
    d.term       = $term,
    d.meaning    = $meaning,
    d.language   = $language,
    d.readings   = $readings,
    d.dynasty    = $dynasty,
    d.source     = $source,
    d.createdAt  = $ts
ON MATCH SET
    d.meaning    = $meaning,
    d.readings   = $readings
"""

_LOOKUP = """
MATCH (d:DICTIONARY_ENTRY)
WHERE d.term = $term AND ($language IS NULL OR d.language = $language)
RETURN d.id AS id, d.term AS term, d.meaning AS meaning,
       d.language AS language, d.readings AS readings,
       d.dynasty AS dynasty, d.source AS source
LIMIT 10
"""

_LOOKUP_FUZZY = """
MATCH (d:DICTIONARY_ENTRY)
WHERE d.term CONTAINS $term AND ($language IS NULL OR d.language = $language)
RETURN d.id AS id, d.term AS term, d.meaning AS meaning,
       d.language AS language, d.readings AS readings,
       d.dynasty AS dynasty, d.source AS source
ORDER BY size(d.term) ASC
LIMIT 5
"""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def seed_dictionary(driver: Driver, path: Path | None = None) -> int:
    """Upsert DICTIONARY_ENTRY nodes from a JSONL seed file.

    Each line: ``{"term": ..., "meaning": ..., "language": ..., ...}``.

    Args:
        driver: Open Neo4j driver.
        path: Override seed file path.

    Returns:
        Number of entries upserted.
    """
    p = path or _SEED_PATH
    if not p.exists():
        log.warning("seed_dictionary: seed file not found at %s", p)
        return 0

    ts = datetime.now(timezone.utc).isoformat()
    count = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry_data = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry = DictEntry(
                term=entry_data["term"],
                meaning=entry_data.get("meaning", ""),
                language=entry_data.get("language", "zh"),
                readings=entry_data.get("readings", []),
                dynasty=entry_data.get("dynasty", ""),
                source=entry_data.get("source", "seed"),
            )
            with driver.session() as s:
                s.run(
                    _UPSERT,
                    id=entry.id,
                    term=entry.term,
                    meaning=entry.meaning,
                    language=entry.language,
                    readings=entry.readings,
                    dynasty=entry.dynasty,
                    source=entry.source,
                    ts=ts,
                ).consume()
            count += 1

    log.info("seed_dictionary: upserted %d entries from %s", count, p)
    return count


def lookup_term(
    driver: Driver,
    term: str,
    language: str | None = None,
    *,
    fuzzy: bool = False,
) -> list[DictEntry]:
    """Look up a term in the DICTIONARY_ENTRY KB.

    Args:
        driver: Open Neo4j driver.
        term: Exact term (or substring if ``fuzzy=True``).
        language: Filter by language (``'zh'`` or ``'ja'``); None = any.
        fuzzy: If True, use ``CONTAINS`` match instead of exact.

    Returns:
        List of matching :class:`DictEntry` objects.
    """
    query = _LOOKUP_FUZZY if fuzzy else _LOOKUP
    with driver.session() as s:
        rows = s.run(query, term=term, language=language).data()

    return [
        DictEntry(
            term=r["term"],
            meaning=r.get("meaning") or "",
            language=r.get("language") or "zh",
            readings=r.get("readings") or [],
            dynasty=r.get("dynasty") or "",
            source=r.get("source") or "",
        )
        for r in rows
    ]


def seed_all_dictionaries(driver: Driver) -> dict[str, int]:
    """Upsert all language dictionary seed files.

    Returns:
        Mapping of language code → number of entries upserted.
    """
    totals: dict[str, int] = {}
    for lang, path in _SEED_PATHS_BY_LANG.items():
        totals[lang] = seed_dictionary(driver, path)
    return totals


def upsert_entry(driver: Driver, entry: DictEntry) -> str:
    """Insert or update a single DICTIONARY_ENTRY.

    Args:
        driver: Open Neo4j driver.
        entry: :class:`DictEntry` to write.

    Returns:
        The node's ``id`` property.
    """
    ts = datetime.now(timezone.utc).isoformat()
    with driver.session() as s:
        s.run(
            _UPSERT,
            id=entry.id,
            term=entry.term,
            meaning=entry.meaning,
            language=entry.language,
            readings=entry.readings,
            dynasty=entry.dynasty,
            source=entry.source,
            ts=ts,
        ).consume()
    log.info("upsert_entry: %s [%s]", entry.term, entry.language)
    return entry.id
