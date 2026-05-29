"""Language detection + (later) tokenizer routing for the KG pipeline.

Phase 1b (plan §6 Stage 1b) ships :mod:`detector` — script-class heuristics
over Unicode blocks that classify NATIVE-TEXT pages into one of
``zh-classical``, ``zh-modern``, ``ja``, ``kanbun``, ``mixed``, or
``unknown``. Phase 3 will add the post-OCR authoritative detector and
Phase 6 will add the tokenizer router (``tokenizer.py``).

The detector is intentionally dependency-light (stdlib + ``unicodedata``)
so the Phase 1 worker can run it inline without pulling lingua / langid /
fasttext model weights. When Phase 3 needs higher precision on fused OCR
output, that path is free to add a heavier disambiguator behind the same
:class:`LanguageProfile` contract.
"""

from apps.backend.lang.detector import (
    Language,
    LanguageProfile,
    ScriptMix,
    detect_language,
    has_kunten_marks,
    script_mix,
)

__all__ = [
    "Language",
    "LanguageProfile",
    "ScriptMix",
    "detect_language",
    "has_kunten_marks",
    "script_mix",
]
