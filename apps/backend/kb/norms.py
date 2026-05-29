"""Norms KB — Phase 6 (plan §6 Stage 6).

Manages ``(:NORM)`` nodes representing classical-Chinese and kanbun
translation/reading conventions. The Translation Agent (``paragraph.py``)
pulls applicable norms via Cypher RAG to guide vernacular production.

``NORM`` properties (plan §5):
  id, rule, scope, tradition

``tradition ∈ {'zh-classical', 'kanbun-kundoku', 'ja-modern', 'shared'}``

Public API
----------
seed_norms(driver, path) -> int
get_norms(driver, tradition, *, limit) -> list[NormEntry]
upsert_norm(driver, norm: NormEntry) -> str
NormEntry                   — data class
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neo4j import Driver

log = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).parents[3] / "data" / "seeds" / "norms_seed.jsonl"

_VALID_TRADITIONS = frozenset({"zh-classical", "kanbun-kundoku", "ja-modern", "shared"})


@dataclass
class NormEntry:
    """One translation/reading norm.

    Attributes:
        rule: The norm text (e.g. ``'「之」作為結構助詞時譯為「的」'``).
        scope: Application scope (e.g. ``'structural_particle'``).
        tradition: One of ``zh-classical``, ``kanbun-kundoku``, ``ja-modern``, ``shared``.
    """

    rule: str
    scope: str = ""
    tradition: str = "zh-classical"

    @property
    def id(self) -> str:
        return "norm_" + hashlib.sha1(
            f"{self.tradition}::{self.rule[:80]}".encode()
        ).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "rule": self.rule, "scope": self.scope, "tradition": self.tradition}


_UPSERT = """
MERGE (n:NORM {id: $id})
ON CREATE SET n.rule = $rule, n.scope = $scope, n.tradition = $tradition,
              n.createdAt = $ts
ON MATCH SET  n.rule = $rule, n.scope = $scope
"""

_GET_NORMS = """
MATCH (n:NORM)
WHERE ($tradition IS NULL OR n.tradition = $tradition)
RETURN n.id AS id, n.rule AS rule, n.scope AS scope, n.tradition AS tradition
ORDER BY n.tradition, n.scope
LIMIT $limit
"""


def seed_norms(driver: Driver, path: Path | None = None) -> int:
    """Upsert NORM nodes from a JSONL seed file.

    Each line: ``{"rule": ..., "scope": ..., "tradition": ...}``.
    """
    p = path or _SEED_PATH
    if not p.exists():
        log.warning("seed_norms: seed file not found at %s", p)
        return 0

    ts = datetime.now(timezone.utc).isoformat()
    count = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            norm = NormEntry(
                rule=d["rule"],
                scope=d.get("scope", ""),
                tradition=d.get("tradition", "zh-classical"),
            )
            if norm.tradition not in _VALID_TRADITIONS:
                log.warning("seed_norms: unknown tradition %r — skipping", norm.tradition)
                continue
            with driver.session() as s:
                s.run(_UPSERT, id=norm.id, rule=norm.rule, scope=norm.scope,
                      tradition=norm.tradition, ts=ts).consume()
            count += 1

    log.info("seed_norms: upserted %d norms from %s", count, p)
    return count


def get_norms(
    driver: Driver,
    tradition: str | None = None,
    *,
    limit: int = 20,
) -> list[NormEntry]:
    """Fetch applicable norms for a given tradition.

    Args:
        driver: Open Neo4j driver.
        tradition: One of ``zh-classical``, ``kanbun-kundoku``, ``ja-modern``,
            ``shared``; ``None`` returns all.
        limit: Maximum number of norms to return.

    Returns:
        List of :class:`NormEntry` objects.
    """
    with driver.session() as s:
        rows = s.run(_GET_NORMS, tradition=tradition, limit=limit).data()
    return [
        NormEntry(rule=r["rule"], scope=r.get("scope") or "", tradition=r.get("tradition") or "zh-classical")
        for r in rows
    ]


def upsert_norm(driver: Driver, norm: NormEntry) -> str:
    """Insert or update a single NORM node."""
    ts = datetime.now(timezone.utc).isoformat()
    with driver.session() as s:
        s.run(_UPSERT, id=norm.id, rule=norm.rule, scope=norm.scope,
              tradition=norm.tradition, ts=ts).consume()
    return norm.id
