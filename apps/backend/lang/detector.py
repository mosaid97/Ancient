"""Script-class language detector for native-text pages (plan §6 Stage 1b).

The detector runs on a per-page ``PAGE.text`` string and produces a
:class:`LanguageProfile` carrying:

- ``language`` ∈ ``{zh-classical, zh-modern, ja, kanbun, mixed, unknown}``
- ``script_mix`` — fractions of hiragana / katakana / kanji / kanbun_marks /
  latin / digits / cjk_punct / whitespace / other (sums to ~1.0)
- ``kunten_marks`` — boolean, True iff one or more chars from the
  CJK-radicals supplement block ``U+3190..U+319F`` (返り点 / 訓点)
- ``confidence`` — heuristic, in ``[0.0, 1.0]`` (set by the deciding rule)

Decision rules (plan §6 Stage 1b — kept deterministic and explainable, so
Phase 11 HITL labels can audit / contest a single rule rather than a
black-box classifier):

1. **kanbun** — ``kunten_marks > 0`` or ``kanji + kanbun_marks > 0`` AND
   ``hiragana + katakana >= 0.05``. Confidence depends on whether 訓点
   marks are explicit (high) vs inferred from kana density alone (medium).
2. **ja** — ``hiragana + katakana >= 0.05`` and no kunten marks. Modern
   Japanese has more kana than kanbun; we deliberately don't fight that
   distinction here because Phase 3's authoritative detector will refine
   it post-OCR.
3. **mixed** — ``latin >= 0.30`` (academic translit, romanization, or a
   secondary paper interleaving English) AND CJK is non-trivial
   (``cjk >= 0.10``).
4. **zh-modern** — kanji-dominant with explicit modern markers (Arabic
   digits ≥ 5% or simplified-only forms ≥ 50% of kanji). Tang-era 古籍
   corpora keep traditional + few Arabic digits, so this branch is
   pragmatically conservative.
5. **zh-classical** — kanji-dominant, no kana, low Arabic digits. Default
   for the corpus.
6. **unknown** — empty / whitespace-only / sub-threshold (``char_count <
   _MIN_CHARS_FOR_CLASSIFICATION``).

The thresholds live as module-level constants so unit tests + the
notebook can tune them. Plan §11 (Move 6) feeds back per-class
mis-classification rates from HITL labels — when that pipeline lands,
the thresholds become a single-source-of-truth knob.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Language = Literal[
    "zh-classical",
    "zh-modern",
    "ja",
    "kanbun",
    "mixed",
    "unknown",
]

# Unicode ranges used for script classification. Tuples are inclusive on
# both ends. Source: Unicode 16.0 block list (uniblock-2024) + the
# ISO 15924 'Hira'/'Kana'/'Hani' aliases.
_HIRAGANA_RANGES: tuple[tuple[int, int], ...] = ((0x3040, 0x309F),)
_KATAKANA_RANGES: tuple[tuple[int, int], ...] = (
    (0x30A0, 0x30FF),
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0xFF66, 0xFF9F),  # Halfwidth Katakana
)
_KANJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # Ext B
    (0x2A700, 0x2B73F),  # Ext C
    (0x2B740, 0x2B81F),  # Ext D
    (0x2B820, 0x2CEAF),  # Ext E
    (0x2CEB0, 0x2EBEF),  # Ext F
    (0x30000, 0x3134F),  # Ext G
    (0x31350, 0x323AF),  # Ext H (Unicode 15)
)
_KANBUN_MARKS_RANGES: tuple[tuple[int, int], ...] = (
    (0x3190, 0x319F),  # Kanbun (返り点 / 訓点 / レ点)
)
_LATIN_RANGES: tuple[tuple[int, int], ...] = (
    (0x0041, 0x005A),  # A-Z
    (0x0061, 0x007A),  # a-z
    (0x00C0, 0x024F),  # Latin Extended-A + B + supplementary
)
_DIGIT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0030, 0x0039),  # ASCII digits
    (0xFF10, 0xFF19),  # Fullwidth digits
)
_CJK_PUNCT_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),  # CJK Symbols and Punctuation
    (0xFE30, 0xFE4F),  # CJK Compatibility Forms
    (0xFF00, 0xFF0F),  # Fullwidth ASCII punctuation block A
    (0xFF1A, 0xFF20),  # block B
    (0xFF3B, 0xFF40),  # block C
    (0xFF5B, 0xFF65),  # block D
)


def _in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    for start, end in ranges:
        if start <= codepoint <= end:
            return True
    return False


# Minimum text length below which the detector returns ``unknown`` rather
# than guess. Empty pages (e.g. blank leaves in scanned EPUBs) and very
# short headings shouldn't drive a confident language decision.
_MIN_CHARS_FOR_CLASSIFICATION: int = 8

# Thresholds (kept tunable for HITL feedback in Phase 11).
_KANA_THRESHOLD: float = 0.05  # >= 5% kana → ja/kanbun branch
_LATIN_MIXED_THRESHOLD: float = 0.30  # >= 30% latin AND >= 5% CJK → mixed
_CJK_NONTRIVIAL_THRESHOLD: float = 0.05  # used together with latin
_MODERN_DIGIT_THRESHOLD: float = 0.03  # >= 3% Arabic digits suggests zh-modern
_MODERN_SIMPLIFIED_LOW_THRESHOLD: float = 0.03  # paired with digit threshold
_MODERN_SIMPLIFIED_HIGH_THRESHOLD: float = 0.15  # standalone marker

# Simplified-only character set we sniff for zh-modern hints. Each glyph is
# a *high-precision* indicator (i.e. unambiguously absent from Tang-era
# 古籍 corpora): a shape introduced by the PRC's 1956/1964 simplification
# tables that did not exist as a variant before. Ambiguous shapes
# (后/後, 几/幾, 还/還, 长/長, 处/處 — all of which appear in classical
# texts) are deliberately excluded to keep precision high.
_SIMPLIFIED_SNIFF: frozenset[str] = frozenset(
    "国学这个们时实选举论经历来关说让见业应么发东车马鸟鱼鸡龙龟"
    "为从对开门间问没传话头条种试爱设认请书会议体礼乐医纸厂买卖"
    "热爱写读练习汉语义师术现现状权"
)


@dataclass
class ScriptMix:
    """Fractional script-class composition of a single page.

    All fields are in ``[0.0, 1.0]`` and the sum across non-``other``
    classes is normalised to ``char_count`` (so ``other`` covers anything
    not classified). Stored as JSON on ``PAGE.scriptMix`` (plan §5).
    """

    hiragana: float = 0.0
    katakana: float = 0.0
    kanji: float = 0.0
    kanbun_marks: float = 0.0
    latin: float = 0.0
    digits: float = 0.0
    cjk_punct: float = 0.0
    whitespace: float = 0.0
    other: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


@dataclass
class LanguageProfile:
    """Per-page detection result. Maps 1:1 to the ``PAGE`` properties.

    Attributes:
        language: One of the :data:`Language` literals.
        script_mix: :class:`ScriptMix`.
        kunten_marks: ``True`` iff the page contains at least one char in
            the Kanbun block ``U+3190..U+319F``.
        confidence: Heuristic in ``[0.0, 1.0]`` — set by the rule that
            decided ``language``. Not a probability; a sentinel for the
            verifier and the Phase-3 re-detector ("am I confident enough
            to skip post-OCR re-detection?").
        char_count: ``len(text.strip())``; gates the ``unknown`` branch.
        rule: Name of the deciding rule (``'kanbun-marks'``, ``'kana-density'``,
            ``'latin-mix'``, ``'modern-digit'``, ``'simplified-sniff'``,
            ``'classical-default'``, ``'too-short'``, ``'empty'``).
    """

    language: Language
    script_mix: ScriptMix
    kunten_marks: bool
    confidence: float
    char_count: int
    rule: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "scriptMix": self.script_mix.to_dict(),
            "kuntenMarks": self.kunten_marks,
            "langConfidence": round(self.confidence, 4),
            "charCount": self.char_count,
            "langDetectionRule": self.rule,
        }


# Pre-compiled regex for kunten-mark detection (faster than per-char loop
# for the common case where the page has none).
_KUNTEN_RE: re.Pattern[str] = re.compile(r"[\u3190-\u319F]")


def has_kunten_marks(text: str) -> bool:
    """Return ``True`` iff ``text`` contains any Kanbun (訓点) reading mark."""
    return bool(_KUNTEN_RE.search(text))


def script_mix(text: str) -> ScriptMix:
    """Compute the fractional script-class composition of ``text``.

    Whitespace + control chars go to ``whitespace``; everything outside
    the known script blocks lands in ``other``. Total of all fields
    equals 1.0 (modulo float rounding) when ``len(text) > 0``.
    """
    if not text:
        return ScriptMix()

    counts = {
        "hiragana": 0,
        "katakana": 0,
        "kanji": 0,
        "kanbun_marks": 0,
        "latin": 0,
        "digits": 0,
        "cjk_punct": 0,
        "whitespace": 0,
        "other": 0,
    }
    for ch in text:
        cp = ord(ch)
        if ch.isspace() or unicodedata.category(ch).startswith("C"):
            counts["whitespace"] += 1
            continue
        if _in_ranges(cp, _KANBUN_MARKS_RANGES):
            counts["kanbun_marks"] += 1
            continue
        if _in_ranges(cp, _HIRAGANA_RANGES):
            counts["hiragana"] += 1
            continue
        if _in_ranges(cp, _KATAKANA_RANGES):
            counts["katakana"] += 1
            continue
        if _in_ranges(cp, _KANJI_RANGES):
            counts["kanji"] += 1
            continue
        if _in_ranges(cp, _DIGIT_RANGES):
            counts["digits"] += 1
            continue
        if _in_ranges(cp, _LATIN_RANGES):
            counts["latin"] += 1
            continue
        if _in_ranges(cp, _CJK_PUNCT_RANGES) or unicodedata.category(ch).startswith("P"):
            counts["cjk_punct"] += 1
            continue
        counts["other"] += 1

    total = sum(counts.values())
    if total == 0:
        return ScriptMix()
    return ScriptMix(**{k: v / total for k, v in counts.items()})


def _simplified_sniff_ratio(text: str) -> float:
    """Fraction of characters in ``_SIMPLIFIED_SNIFF`` over the kanji subset.

    Used to nudge a kanji-dominant page from ``zh-classical`` to
    ``zh-modern`` only when there is clear evidence of vernacular
    simplification. Returns ``0.0`` if there are no kanji at all.
    """
    if not text:
        return 0.0
    kanji_total = 0
    simplified = 0
    for ch in text:
        cp = ord(ch)
        if _in_ranges(cp, _KANJI_RANGES):
            kanji_total += 1
            if ch in _SIMPLIFIED_SNIFF:
                simplified += 1
    return (simplified / kanji_total) if kanji_total > 0 else 0.0


def detect_language(text: str | None) -> LanguageProfile:
    """Classify ``text`` and return a complete :class:`LanguageProfile`.

    Idempotent + side-effect-free. Safe to call from a Celery worker, a
    notebook, or a unit test.

    Args:
        text: The page text (typically ``PAGE.text``). ``None`` or empty
            input yields a ``unknown`` profile so the caller can treat
            it as missing without a try/except.

    Returns:
        :class:`LanguageProfile` whose ``to_dict()`` is suitable for direct
        use as Cypher parameters (matches the ``PAGE`` properties listed
        in plan §5).
    """
    if not text:
        return LanguageProfile(
            language="unknown",
            script_mix=ScriptMix(),
            kunten_marks=False,
            confidence=0.0,
            char_count=0,
            rule="empty",
        )

    text_str = text.strip()
    char_count = len(text_str)
    mix = script_mix(text_str)
    kunten = mix.kanbun_marks > 0.0

    if char_count < _MIN_CHARS_FOR_CLASSIFICATION:
        return LanguageProfile(
            language="unknown",
            script_mix=mix,
            kunten_marks=kunten,
            confidence=0.0,
            char_count=char_count,
            rule="too-short",
        )

    kana_ratio = mix.hiragana + mix.katakana
    cjk_ratio = mix.kanji + mix.kanbun_marks
    has_explicit_kunten = kunten

    # Rule 1: kanbun — explicit reading marks OR kanji-heavy text that still
    # carries enough kana to be Japanese-reading-of-classical-Chinese.
    if has_explicit_kunten and cjk_ratio > 0.0:
        return LanguageProfile(
            language="kanbun",
            script_mix=mix,
            kunten_marks=True,
            confidence=0.95,
            char_count=char_count,
            rule="kunten-marks",
        )

    # Rule 2: ja — kana density without explicit kunten marks. Modern
    # Japanese typically has 30%+ kana; even formal kanbun-style writing
    # without 訓点 keeps some kana for verb endings.
    if kana_ratio >= _KANA_THRESHOLD:
        return LanguageProfile(
            language="ja",
            script_mix=mix,
            kunten_marks=False,
            confidence=min(0.95, 0.6 + 2.0 * kana_ratio),
            char_count=char_count,
            rule="kana-density",
        )

    # Rule 3: mixed — latin-heavy with non-trivial CJK (academic papers,
    # bibliography pages, romanized indexes).
    if mix.latin >= _LATIN_MIXED_THRESHOLD and cjk_ratio >= _CJK_NONTRIVIAL_THRESHOLD:
        return LanguageProfile(
            language="mixed",
            script_mix=mix,
            kunten_marks=False,
            confidence=0.85,
            char_count=char_count,
            rule="latin-mix",
        )

    # Rule 4: zh-modern — kanji-dominant with modern markers. We accept
    # EITHER (a) digit-citation density paired with at least a weak
    # simplified signal, OR (b) a strong simplified signal alone. The
    # corpus mostly ingests Tang-era 古籍 (zh-classical), so we keep
    # precision higher than recall: a borderline zh-modern page that
    # falls through to ``classical-default`` will still be handled
    # correctly by the downstream normalizer (T-S unification is
    # idempotent on already-traditional text).
    if cjk_ratio >= 0.30:
        simplified_ratio = _simplified_sniff_ratio(text_str)
        if (
            mix.digits >= _MODERN_DIGIT_THRESHOLD
            and simplified_ratio >= _MODERN_SIMPLIFIED_LOW_THRESHOLD
        ):
            return LanguageProfile(
                language="zh-modern",
                script_mix=mix,
                kunten_marks=False,
                confidence=0.85,
                char_count=char_count,
                rule="modern-digit",
            )
        if simplified_ratio >= _MODERN_SIMPLIFIED_HIGH_THRESHOLD:
            return LanguageProfile(
                language="zh-modern",
                script_mix=mix,
                kunten_marks=False,
                confidence=0.80,
                char_count=char_count,
                rule="simplified-sniff",
            )

    # Rule 5: zh-classical — kanji-dominant default. Confidence scales with
    # how dominant the kanji are vs CJK-punctuation + other noise.
    if cjk_ratio >= 0.30:
        return LanguageProfile(
            language="zh-classical",
            script_mix=mix,
            kunten_marks=False,
            confidence=min(0.95, 0.5 + cjk_ratio),
            char_count=char_count,
            rule="classical-default",
        )

    # Fallback: text is non-empty but has very little CJK and not enough
    # latin to be 'mixed'. Mark as unknown so Phase 3 / HITL can re-look.
    return LanguageProfile(
        language="unknown",
        script_mix=mix,
        kunten_marks=False,
        confidence=0.0,
        char_count=char_count,
        rule="no-rule-matched",
    )


# Lightweight self-test fixtures (used by the notebook and tests/). Keep
# them inline so the module is self-documenting; tests import the names
# rather than the strings.
SELF_TEST_FIXTURES: dict[str, tuple[str, Language]] = {
    "tang-classical-prose": (
        "貞觀十一年正月辛丑詔曰朕聞天地之大德曰生君人者必當以德爲本",
        "zh-classical",
    ),
    "modern-vernacular-zh": (
        "唐代的科举制度是中国古代选拔官吏的一种重要制度,起源于隋朝。"
        "本文将从制度史和社会史两个角度分析其演变过程,涉及2001年以来的研究成果。",
        "zh-modern",
    ),
    "kanbun-with-marks": (
        "天命㆐之謂㆑性、率㆒性之謂㆓道、修㆓道之謂㆔教。",
        "kanbun",
    ),
    "japanese-modern": (
        "唐代の科挙制度は中国古代における官吏選抜の重要な制度であり、"
        "隋代に起源を持つ。本論では制度史と社会史の両側面から分析する。",
        "ja",
    ),
    "mixed-academic": (
        "Liu Houbin (刘后滨), The Bureaucratic Selection System of the Tang Dynasty "
        "唐代选官政务研究, Beijing University Press, 2005, pp. 12-45.",
        "mixed",
    ),
    "empty": ("", "unknown"),
    "too-short": ("貞觀", "unknown"),
}
