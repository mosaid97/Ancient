"""紀年 → CE date conversion.

Converts strings like ``貞觀十九年``, ``武德元年``, ``開元二十年甲子`` into a
CE year (and an ``EraMatch`` record with provenance). Used by:

- The Translation Agent (Phase 6) to extract ``Temporal`` entities.
- The KG (Phase 8) to wire ``Temporal`` nodes to ``DOCUMENT.publicationPeriod``.
- The era classifier (Phase 6 stub) to tag ``PAGE.detectedEra``, which then
  drives :mod:`apps.backend.normalize.taboo` selection.

Algorithm:

1. Find every ``<era_name><era_year>年`` match (era_year may be a Chinese
   numeral, an Arabic numeral, or 元 = 1).
2. Look up ``era_name`` in ``data/seeds/era_calendar.yaml`` (the same name
   may appear in multiple dynasties — return all matches; caller chooses by
   surrounding context).
3. For each match, CE = era.start_year + year_offset - 1.
4. Optionally cross-check against any nearby 干支 token: if the computed CE
   year doesn't match the 干支 declared in the source, downgrade
   ``confidence``. Anchor: ``ganzhi_anchor_year`` from the YAML.

The era classifier — a small ``deepseek-chat`` few-shot wrapper that decides
``PAGE.detectedEra`` for an arbitrary page — is intentionally OUT of this
module (lives in ``apps.backend.agents.translation`` in Phase 6). This module
is purely deterministic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SEED = Path(__file__).resolve().parents[3] / "data" / "seeds" / "era_calendar.yaml"

CHINESE_DIGIT = {
    "〇": 0,
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "壹": 1,
    "貳": 2,
    "弍": 2,
    "參": 3,
    "肆": 4,
    "伍": 5,
    "陸": 6,
    "柒": 7,
    "捌": 8,
    "玖": 9,
    "兩": 2,
}


def _cn_to_int(token: str) -> int | None:
    """Convert a Chinese-numeral year fragment (1..99) to an int.

    Handles ``元`` (=1), Arabic digits, and standard 1-2 digit forms like
    ``十``/``十九``/``二十``/``二十三``. Returns ``None`` if unparseable.
    """
    if not token:
        return None
    token = token.strip()
    if token.isdigit():
        return int(token)
    if token == "元":
        return 1
    n = 0
    if "十" in token:
        head, _, tail = token.partition("十")
        head_n = CHINESE_DIGIT.get(head, 1) if head else 1
        tail_n = CHINESE_DIGIT.get(tail, 0) if tail else 0
        return head_n * 10 + tail_n
    if len(token) == 1 and token in CHINESE_DIGIT:
        return CHINESE_DIGIT[token]
    n = 0
    for ch in token:
        if ch not in CHINESE_DIGIT:
            return None
        n = n * 10 + CHINESE_DIGIT[ch]
    return n if n > 0 else None


@dataclass(frozen=True)
class EraEntry:
    era_name: str
    dynasty: str
    emperor: str
    start_year: int
    end_year: int
    notes: str = ""


@dataclass
class EraMatch:
    raw: str
    era_name: str
    year_offset: int
    candidates: list[EraEntry]
    ce_year: int | None
    ganzhi_in_text: str | None = None
    ganzhi_expected: str | None = None
    confidence: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml not installed; uv add pyyaml") from exc
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=2)
def load_calendar(seed_path: str | None = None) -> dict[str, Any]:
    """Load and cache the era calendar YAML."""
    path = Path(seed_path) if seed_path else DEFAULT_SEED
    return _load_yaml(path)


@lru_cache(maxsize=2)
def _era_index(seed_path: str | None = None) -> dict[str, list[EraEntry]]:
    """Build ``{era_name: [EraEntry, ...]}`` index for fast lookup.

    Same era name (e.g. ``上元``) can recur across dynasties / emperors;
    return all matches. Caller resolves ambiguity by context.
    """
    cal = load_calendar(seed_path)
    idx: dict[str, list[EraEntry]] = {}
    for dyn in cal.get("dynasties", []):
        dyn_name = dyn.get("name", "")
        for era in dyn.get("eras", []):
            entry = EraEntry(
                era_name=era.get("era_name", ""),
                dynasty=dyn_name,
                emperor=era.get("emperor", ""),
                start_year=int(era.get("start_year", 0)),
                end_year=int(era.get("end_year", 0)),
                notes=era.get("notes", "") or "",
            )
            if entry.era_name:
                idx.setdefault(entry.era_name, []).append(entry)
    return idx


def _ganzhi_for_year(year: int, cal: dict[str, Any]) -> str | None:
    cycle = cal.get("ganzhi_cycle") or []
    anchor = int(cal.get("ganzhi_anchor_year", 4))
    if not cycle:
        return None
    return cycle[(year - anchor) % 60]


_ERA_YEAR_CHARS = (
    "一二三四五六七八九十元壹貳弍參肆伍陸柒捌玖兩〇零"
    "0123456789"
)


@lru_cache(maxsize=2)
def _era_regex(seed_path: str | None = None) -> re.Pattern[str]:
    """Build a regex that anchors on KNOWN era names (longest-first).

    The era prefix is exactly one of the YAML's ``era_name`` values, so the
    matcher avoids the shortest-prefix ambiguity of a generic CJK regex
    (e.g. 開元二十年 needs to bind ``era=開元`` not ``era=開``).
    """
    idx = _era_index(seed_path)
    names = sorted(idx.keys(), key=lambda n: (-len(n), n))
    if not names:
        return re.compile(r"(?!x)x")
    alternation = "|".join(re.escape(n) for n in names)
    pattern = (
        r"(?P<era>" + alternation + r")"
        r"(?P<year>[" + _ERA_YEAR_CHARS + r"]{1,4})年"
    )
    return re.compile(pattern)


@lru_cache(maxsize=2)
def _ganzhi_re(seed_path: str | None = None) -> re.Pattern[str]:
    cal = load_calendar(seed_path)
    cycle = cal.get("ganzhi_cycle") or []
    if not cycle:
        return re.compile(r"(?!x)x")
    return re.compile("|".join(re.escape(g) for g in cycle))


def find_eras(text: str, *, seed_path: str | None = None) -> list[EraMatch]:
    """Scan ``text`` for ``<era_name><year>年`` patterns.

    Args:
        text: Input string. May be classical or modern Chinese.
        seed_path: Optional override for the era calendar YAML.

    Returns:
        List of :class:`EraMatch`, one per recognized pattern. The list may
        be empty.
    """
    if not text:
        return []
    cal = load_calendar(seed_path)
    idx = _era_index(seed_path)
    matches: list[EraMatch] = []
    pattern = _era_regex(seed_path)
    for m in pattern.finditer(text):
        era_name = m.group("era")
        year_token = m.group("year")
        candidates = idx.get(era_name, [])
        if not candidates:
            continue
        year_offset = _cn_to_int(year_token)
        if year_offset is None or year_offset <= 0:
            continue
        fitting = [
            c for c in candidates if c.start_year + year_offset - 1 <= c.end_year
        ]
        chosen = fitting[0] if fitting else candidates[0]
        ce_year = chosen.start_year + year_offset - 1

        window = text[max(0, m.start() - 6) : min(len(text), m.end() + 6)]
        gz_re = _ganzhi_re(seed_path)
        gz_match = gz_re.search(window)
        gz_in_text = gz_match.group() if gz_match else None
        gz_expected = _ganzhi_for_year(ce_year, cal)
        confidence = 1.0
        if gz_in_text and gz_expected and gz_in_text != gz_expected:
            confidence = 0.6
        if len(fitting) > 1:
            confidence = min(confidence, 0.85)
        if not fitting:
            confidence = min(confidence, 0.5)

        matches.append(
            EraMatch(
                raw=m.group(0),
                era_name=era_name,
                year_offset=year_offset,
                candidates=fitting or candidates,
                ce_year=ce_year,
                ganzhi_in_text=gz_in_text,
                ganzhi_expected=gz_expected,
                confidence=confidence,
            )
        )
    return matches


def to_ce(reign: str, *, seed_path: str | None = None) -> int | None:
    """Convert a single ``<era_name><year>年`` token to a CE year.

    Args:
        reign: e.g. ``"貞觀十九年"`` or ``"開元20年"``.
        seed_path: Override.

    Returns:
        CE year, or ``None`` if unparseable / unknown.
    """
    matches = find_eras(reign, seed_path=seed_path)
    if not matches:
        return None
    return matches[0].ce_year


def detect_era_dynasty(
    text: str, *, seed_path: str | None = None
) -> str | None:
    """Best-effort dynasty inference from a single text fragment.

    Looks at the first :class:`EraMatch` and returns its dynasty (one of
    ``唐``, ``北宋``, ``隋``, ...). ``None`` if no recognizable reign is found.
    Tang-era era names dominate the seed, so this is most reliable for Tang.
    """
    matches = find_eras(text, seed_path=seed_path)
    if not matches:
        return None
    for m in matches:
        if m.candidates:
            return m.candidates[0].dynasty
    return None
