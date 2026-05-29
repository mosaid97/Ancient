"""Step 5 of the canonical pipeline: 避諱 (era-conditional taboo) unification.

Loads ``data/seeds/taboo_{tang,song,ming,qing}.yaml`` and applies a per-era
substitution map. Only fires when an ``era`` is supplied (typically derived
from ``DOCUMENT.publicationPeriod`` or ``PAGE.detectedEra``).

Two safety controls (per the YAML schema's ``safe`` flag):

- ``safe: true``   — applied unconditionally when the era matches.
- ``safe: false``  — high-collision pairs (e.g. 世->代, 民->人); only applied
  when ``unsafe=True`` is passed explicitly. Default OFF.

Direction: the seed records 「avoidance was: original -> substitution」.
Canonicalization rewrites ``substitution -> original`` so the verifier can
match a query for the original character against a Tang doc that wrote the
substitution.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parents[3] / "data" / "seeds"

KNOWN_DYNASTIES: tuple[str, ...] = ("tang", "song", "ming", "qing")

# Aliases for era specifiers that callers may pass.
ERA_ALIASES: dict[str, str] = {
    "tang": "tang",
    "唐": "tang",
    "song": "song",
    "宋": "song",
    "ming": "ming",
    "明": "ming",
    "qing": "qing",
    "清": "qing",
}


def _seed_path(dynasty: str) -> Path:
    return SEED_DIR / f"taboo_{dynasty}.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml not installed; uv add pyyaml") from exc
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=4)
def load_table(dynasty: str, *, seed_dir: str | None = None) -> dict[str, Any]:
    """Load and cache one dynasty's taboo table.

    Args:
        dynasty: Lowercase short name (``tang``, ``song``, ``ming``, ``qing``).
        seed_dir: Override directory; defaults to ``data/seeds/``.

    Returns:
        The raw YAML dict, or ``{}`` if the file is missing.
    """
    base = Path(seed_dir) if seed_dir else SEED_DIR
    path = base / f"taboo_{dynasty}.yaml"
    return _load_yaml(path)


def _resolve_era(era: str | None) -> str | None:
    if era is None:
        return None
    key = era.strip().lower()
    return ERA_ALIASES.get(key) or ERA_ALIASES.get(era.strip())


def build_substitution_map(
    dynasty: str, *, unsafe: bool = False, seed_dir: str | None = None
) -> dict[str, str]:
    """Build ``{substitution_char: original_char}`` for the given dynasty.

    Args:
        dynasty: Lowercase short name (``tang``, ``song``, ``ming``, ``qing``).
        unsafe: Include rows marked ``safe: false`` (high-collision pairs).
        seed_dir: Override directory.

    Returns:
        A flat char->char map. Empty if the seed is missing.
    """
    table = load_table(dynasty, seed_dir=seed_dir)
    out: dict[str, str] = {}
    for emperor in table.get("emperors", []):
        for entry in emperor.get("taboo", []) or []:
            original = (entry.get("original") or "").strip()
            if not original or len(original) != 1:
                continue
            safe = bool(entry.get("safe", True))
            if not safe and not unsafe:
                continue
            for sub in entry.get("substitutions", []) or []:
                sub = (sub or "").strip()
                if sub and len(sub) == 1 and sub != original:
                    out.setdefault(sub, original)
    return out


def normalize(
    text: str,
    *,
    era: str | None,
    unsafe: bool = False,
    seed_dir: str | None = None,
) -> str:
    """Apply the era-conditional taboo substitution map.

    Args:
        text: Input string.
        era: Era keyword (``Tang``/``唐``/``tang``/...); ``None`` -> no-op.
        unsafe: Include high-collision pairs (default off).
        seed_dir: Override directory for the seed YAMLs.

    Returns:
        Text with taboo substitutions rewritten back to original characters.
    """
    if not text:
        return text
    dynasty = _resolve_era(era)
    if dynasty is None:
        return text
    if dynasty not in KNOWN_DYNASTIES:
        logger.debug("unknown dynasty %r; taboo step is a no-op", era)
        return text
    char_map = build_substitution_map(dynasty, unsafe=unsafe, seed_dir=seed_dir)
    if not char_map:
        return text
    return "".join(char_map.get(ch, ch) for ch in text)
