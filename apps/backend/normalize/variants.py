"""Step 4 of the canonical pipeline: 異體字 (variant character) unification.

Loads ``data/seeds/variants_unihan.tsv`` and applies a single-char substitution
map. The seed is small (~50 pairs); Phase 11 HITL grows it. Full Unihan ingest
is tracked in ``scripts/seed_variants.py``.

Mapping semantics:

- The TSV is a directed map ``variant -> canonical``.
- Bidirectional pairs are encoded by both rows (``A -> B`` and ``B -> A``).
- The loader builds an equivalence union-find and picks a single canonical
  representative per class (the one that appears most often as a target).
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SEED = Path(__file__).resolve().parents[3] / "data" / "seeds" / "variants_unihan.tsv"


def _parse_tsv(path: Path) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if cols[0] == "variant" and cols[1] == "canonical":
                continue
            if len(cols) < 2:
                continue
            variant = cols[0].strip()
            canonical = cols[1].strip()
            source = cols[2].strip() if len(cols) > 2 else ""
            notes = cols[3].strip() if len(cols) > 3 else ""
            if variant and canonical and variant != canonical:
                rows.append((variant, canonical, source, notes))
    return rows


def _build_canonical_map(rows: Iterable[tuple[str, str, str, str]]) -> dict[str, str]:
    """Build a per-character substitution map.

    Strategy: union-find over equivalence pairs, then pick the canonical for
    each class as the most frequent right-hand-side in the seed. Single-char
    rules only — multi-char patterns are not supported in v1.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    rhs_counter: Counter[str] = Counter()
    for variant, canonical, _src, _notes in rows:
        if len(variant) != 1 or len(canonical) != 1:
            logger.debug("skipping multi-char variant pair: %r -> %r", variant, canonical)
            continue
        parent.setdefault(variant, variant)
        parent.setdefault(canonical, canonical)
        union(variant, canonical)
        rhs_counter[canonical] += 1

    classes: dict[str, list[str]] = {}
    for ch in parent:
        classes.setdefault(find(ch), []).append(ch)

    char_map: dict[str, str] = {}
    for members in classes.values():
        if len(members) <= 1:
            continue
        canonical = max(members, key=lambda c: (rhs_counter.get(c, 0), c))
        for m in members:
            if m != canonical:
                char_map[m] = canonical
    return char_map


@lru_cache(maxsize=4)
def load_map(seed_path: str | None = None) -> dict[str, str]:
    """Load and cache the 異體字 substitution map.

    Args:
        seed_path: Override path; defaults to
            ``data/seeds/variants_unihan.tsv``.

    Returns:
        Dict ``{variant_char: canonical_char}``.
    """
    path = Path(seed_path) if seed_path else DEFAULT_SEED
    if not path.exists():
        logger.warning("variants seed not found at %s; returning empty map.", path)
        return {}
    rows = _parse_tsv(path)
    return _build_canonical_map(rows)


def normalize(text: str, *, seed_path: str | None = None) -> str:
    """Substitute every variant character by its canonical form.

    Args:
        text: Input string.
        seed_path: Optional override for the seed TSV.

    Returns:
        Text with each known variant character replaced by its canonical form.
    """
    if not text:
        return text
    char_map = load_map(seed_path)
    if not char_map:
        return text
    return "".join(char_map.get(ch, ch) for ch in text)
