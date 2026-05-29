"""Centralized philological normalization layer (plan §2.7).

The single highest-leverage architectural decision in the project: every
consumer (indexer, translator, citation linker, verifier) imports
:func:`apps.backend.normalize.pipeline.normalize_canonical` rather than
re-implementing per stage.

7-step canonical pipeline (per plan §2.7):

1. NFC                       (always)
2. whitespace collapse       (always)
3. T-S unification           (always)
4. 異體字 (variant chars)     (always)
5. 避諱 (era-conditional)     (default ON if era is provided)
6. 通假字                     (default OFF — opt-in via ``apply_loan=True``)
7. mojimoji (half/full kana) (always for ``ja``/``mixed`` languages)

Public API mirrors the plan: ``normalize_canonical(text, *, lang, era, ...)``.
"""

from apps.backend.normalize.pipeline import (
    CanonicalResult,
    NormalizeStep,
    normalize_canonical,
)

__all__ = ["CanonicalResult", "NormalizeStep", "normalize_canonical"]
