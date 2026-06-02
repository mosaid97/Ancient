"""Language-rules registry (Track E, Feature 3).

The translation + normalization stack was originally hard-wired to Chinese
(plus Japanese kanbun). This registry makes the *language axis* explicit so
new languages can be added without touching the pipeline: each registered
language declares how to normalize its text, which ``DICTIONARY_ENTRY``
``language`` tag it uses, and where its dictionary seed lives.

Scope (per the project's "Chinese-first" directive):

- ``zh`` (Chinese): **fully wired** — delegates to the canonical 7-step
  :func:`apps.backend.normalize.normalize_canonical` pipeline.
- ``ja`` (Japanese / kanbun): partially wired — uses the same canonical
  pipeline with ``lang='ja'`` (mojimoji width folding kicks in).
- ``en`` (English), ``ar`` (Arabic): **registered stubs** — identity
  normalization (NFC + whitespace only) + empty seed files + documented
  extension points. They exist so the dictionary KB, the upload-language
  selector, and the UI locale switcher all have a real language to bind to;
  full normalization rules are future work.

Public API:
    get_language_rule(code) -> LanguageRule
    normalize_for_language(text, code, **kw) -> str
    REGISTRY: dict[str, LanguageRule]
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_SEEDS = Path(__file__).parents[3] / "data" / "seeds"


@dataclass
class LanguageRule:
    """Declarative config for one language's normalization + KB wiring.

    Attributes:
        code: UI language code (``zh`` / ``ja`` / ``en`` / ``ar``).
        display_name: Human-readable name.
        dict_language: The ``DICTIONARY_ENTRY.language`` tag this language
            looks up (zh and ja share the CJK dictionary tags).
        seed_file: Path to this language's dictionary seed (may not exist
            yet for stubs).
        normalize: ``fn(text, **kwargs) -> str`` normalization callable.
        rtl: Whether the script is right-to-left (Arabic).
        fully_wired: ``True`` only for languages with real normalization
            rules; ``False`` marks a registered stub.
    """

    code: str
    display_name: str
    dict_language: str
    seed_file: Path
    normalize: Callable[..., str]
    rtl: bool = False
    fully_wired: bool = False
    notes: str = ""


def _identity_normalize(text: str, **_: object) -> str:
    """NFC + whitespace collapse only — the safe default for stub languages."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def _chinese_normalize(text: str, *, era: str | None = None, **_: object) -> str:
    """Full canonical 7-step pipeline (Chinese)."""
    from apps.backend.normalize import normalize_canonical

    return normalize_canonical(text, lang="zh", era=era).canonical


def _japanese_normalize(text: str, *, era: str | None = None, **_: object) -> str:
    """Canonical pipeline with ``lang='ja'`` (enables mojimoji width folding)."""
    from apps.backend.normalize import normalize_canonical

    return normalize_canonical(text, lang="ja", era=era).canonical


REGISTRY: dict[str, LanguageRule] = {
    "zh": LanguageRule(
        code="zh",
        display_name="中文 Chinese",
        dict_language="zh",
        seed_file=_SEEDS / "dictionary_seed.jsonl",
        normalize=_chinese_normalize,
        fully_wired=True,
        notes="Canonical 7-step pipeline (NFC→whitespace→T-S→異體字→避諱→通假字→mojimoji).",
    ),
    "ja": LanguageRule(
        code="ja",
        display_name="日本語 Japanese",
        dict_language="ja",
        seed_file=_SEEDS / "dictionary_seed_ja.jsonl",
        normalize=_japanese_normalize,
        fully_wired=False,
        notes="Shares the canonical pipeline (lang='ja'); kanbun handled at OCR time.",
    ),
    "en": LanguageRule(
        code="en",
        display_name="English",
        dict_language="en",
        seed_file=_SEEDS / "dictionary_seed_en.jsonl",
        normalize=_identity_normalize,
        fully_wired=False,
        notes="STUB: NFC + whitespace only. Extension point: add lemmatization / "
              "stemming here and an English glossary seed.",
    ),
    "ar": LanguageRule(
        code="ar",
        display_name="العربية Arabic",
        dict_language="ar",
        seed_file=_SEEDS / "dictionary_seed_ar.jsonl",
        normalize=_identity_normalize,
        rtl=True,
        fully_wired=False,
        notes="STUB: NFC + whitespace only. Extension point: add tatweel removal, "
              "diacritic (harakat) stripping, alef/ya normalization, and an Arabic seed.",
    ),
}

# Map the upload-form language labels to registry codes.
UPLOAD_LANGUAGE_TO_CODE: dict[str, str] = {
    "chinese": "zh",
    "japanese": "ja",
    "english": "en",
    "arabic": "ar",
}


def get_language_rule(code: str) -> LanguageRule:
    """Return the :class:`LanguageRule` for ``code`` (falls back to Chinese)."""
    return REGISTRY.get(code, REGISTRY["zh"])


def normalize_for_language(text: str, code: str, **kwargs: object) -> str:
    """Normalize ``text`` using the rule registered for ``code``."""
    return get_language_rule(code).normalize(text, **kwargs)
