"""Step 6 of the canonical pipeline: 通假字 (phonetic loan) unification.

OFF by default in the verifier (plan §2.7) — over-normalization risk. The
translator uses it always; the verifier flips it on only when a query carries
``loose=true``.

Loads ``data/seeds/loan_chars.tsv``. The TSV's ``direction`` column controls
how each pair is realized in the substitution map:

- ``a>b`` — canonical = ``b``; the map records ``a -> b``.
- ``b>a`` — canonical = ``a``; the map records ``b -> a``.
- ``bi``  — both directions; the loader skips them by default (since either
  side is a defensible choice). Pass ``include_bi=True`` to fold them into
  the canonical map (``a -> b``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SEED = Path(__file__).resolve().parents[3] / "data" / "seeds" / "loan_chars.tsv"


def _parse_tsv(path: Path) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if cols[0] == "a" and cols[1] == "b":
                continue
            if len(cols) < 3:
                continue
            a = cols[0].strip()
            b = cols[1].strip()
            direction = cols[2].strip()
            context = cols[3].strip() if len(cols) > 3 else ""
            notes = cols[4].strip() if len(cols) > 4 else ""
            if a and b and a != b:
                rows.append((a, b, direction, context, notes))
    return rows


def _build_map(
    rows: Iterable[tuple[str, str, str, str, str]], *, include_bi: bool
) -> dict[str, str]:
    out: dict[str, str] = {}
    for a, b, direction, _ctx, _notes in rows:
        if len(a) != 1 or len(b) != 1:
            continue
        if direction == "a>b":
            out.setdefault(a, b)
        elif direction == "b>a":
            out.setdefault(b, a)
        elif direction == "bi" and include_bi:
            out.setdefault(a, b)
        else:
            continue
    return out


@lru_cache(maxsize=4)
def load_map(seed_path: str | None = None, include_bi: bool = False) -> dict[str, str]:
    """Load and cache the 通假字 substitution map.

    Args:
        seed_path: Override path; defaults to ``data/seeds/loan_chars.tsv``.
        include_bi: Fold symmetric pairs into the map (``a -> b``).

    Returns:
        Dict ``{char: canonical_char}``.
    """
    path = Path(seed_path) if seed_path else DEFAULT_SEED
    if not path.exists():
        logger.warning("loan_chars seed not found at %s; returning empty map.", path)
        return {}
    rows = _parse_tsv(path)
    return _build_map(rows, include_bi=include_bi)


def normalize(
    text: str, *, seed_path: str | None = None, include_bi: bool = False
) -> str:
    """Apply 通假字 substitutions.

    Args:
        text: Input string.
        seed_path: Optional override for the seed TSV.
        include_bi: Fold symmetric pairs into the map.

    Returns:
        Text with phonetic-loan substitutions applied.
    """
    if not text:
        return text
    char_map = load_map(seed_path, include_bi=include_bi)
    if not char_map:
        return text
    return "".join(char_map.get(ch, ch) for ch in text)
