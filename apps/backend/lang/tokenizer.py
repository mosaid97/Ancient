"""Tokenizer router — Phase 6 (plan §6 Stage 6).

Routes text to the appropriate tokenizer based on ``PAGE.language``:
- ``zh-classical`` / ``zh-modern`` → ``jieba`` (Chinese word segmentation)
- ``ja`` / ``kanbun``              → ``fugashi`` + ``unidic-lite`` (Japanese MeCab)
- ``mixed``                        → both-and-merge (tokenize with both, merge result)
- ``unknown`` / default            → jieba fallback

Each tokenized word carries a ``Token`` record:
  surface, lemma, pos, reading (ja only), language

Public API
----------
Token                       — per-token record
TokenizerRouter             — main class with tokenize() method
tokenize(text, language)    — convenience function
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Lazy-import guards — deps are optional at import time; missing deps
# fall back gracefully rather than crashing on import.
_jieba_ok: bool = False
_fugashi_ok: bool = False

try:
    import jieba
    import jieba.posseg as pseg
    _jieba_ok = True
except ImportError:
    log.debug("jieba not installed — zh tokenizer will use char-by-char fallback")

try:
    import fugashi
    _fugashi_ok = True
except ImportError:
    log.debug("fugashi not installed — ja tokenizer will use char-by-char fallback")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class Token:
    """A single tokenized unit."""

    surface: str            # the actual text span
    lemma: str              # dictionary form
    pos: str                # part-of-speech tag
    reading: str = ""       # kana reading (ja only)
    language: str = "zh"    # 'zh' or 'ja'

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "surface": self.surface,
            "lemma": self.lemma,
            "pos": self.pos,
            "language": self.language,
        }
        if self.reading:
            d["reading"] = self.reading
        return d


# ---------------------------------------------------------------------------
# Per-language tokenizers
# ---------------------------------------------------------------------------


def _tokenize_zh(text: str) -> list[Token]:
    """Tokenize Chinese text with jieba (posseg mode)."""
    if not _jieba_ok:
        # Char-by-char fallback
        return [Token(surface=c, lemma=c, pos="CHAR", language="zh") for c in text if c.strip()]
    try:
        tokens: list[Token] = []
        for word, flag in pseg.cut(text):
            if word.strip():
                tokens.append(Token(surface=word, lemma=word, pos=str(flag), language="zh"))
        return tokens
    except Exception as exc:
        log.warning("_tokenize_zh failed: %s — using char fallback", exc)
        return [Token(surface=c, lemma=c, pos="CHAR", language="zh") for c in text if c.strip()]


def _tokenize_ja(text: str) -> list[Token]:
    """Tokenize Japanese text with fugashi + unidic-lite."""
    if not _fugashi_ok:
        return [Token(surface=c, lemma=c, pos="CHAR", language="ja") for c in text if c.strip()]
    try:
        import unidic_lite
        tagger = fugashi.GenericTagger(f"-d {unidic_lite.DICDIR}")
        tokens: list[Token] = []
        for word in tagger(text):
            surface = word.surface
            if not surface.strip():
                continue
            feature = word.feature
            # unidic feature fields: pos1, pos2, pos3, pos4, cType, cForm,
            #   lForm, lemma, orth, pron, orthBase, pronBase, ...
            try:
                pos = feature.pos1 or "UNK"
                lemma = feature.lemma or surface
                reading = getattr(feature, "pron", "") or getattr(feature, "reading", "") or ""
            except Exception:
                pos, lemma, reading = "UNK", surface, ""
            tokens.append(Token(surface=surface, lemma=lemma, pos=pos, reading=reading, language="ja"))
        return tokens
    except Exception as exc:
        log.warning("_tokenize_ja failed: %s — using char fallback", exc)
        return [Token(surface=c, lemma=c, pos="CHAR", language="ja") for c in text if c.strip()]


def _tokenize_en(text: str) -> list[Token]:
    """Tokenize English text — whitespace + regex word splitting."""
    import re
    tokens: list[Token] = []
    for word in re.findall(r"[A-Za-z](?:[A-Za-z'\-]*[A-Za-z])?", text):
        w = word.lower()
        if w:
            tokens.append(Token(surface=word, lemma=w, pos="WORD", language="en"))
    return tokens


def _tokenize_ar(text: str) -> list[Token]:
    """Tokenize Arabic text — whitespace splitting with punctuation strip."""
    import re
    tokens: list[Token] = []
    for word in re.split(r"[\s،؛؟؍،؛؟]+", text):
        word = word.strip("‏‎.,;:!?()\"""''")
        if word:
            tokens.append(Token(surface=word, lemma=word, pos="WORD", language="ar"))
    return tokens


def _is_cjk(char: str) -> bool:
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF
    )


def _is_kana(char: str) -> bool:
    cp = ord(char)
    return 0x3040 <= cp <= 0x30FF


def _tokenize_mixed(text: str) -> list[Token]:
    """Both-and-merge for mixed zh/ja text.

    Strategy: split on script boundaries, route each segment to zh or ja,
    then concatenate the token lists.
    """
    # Split into runs of CJK, kana, latin, and other
    segments: list[tuple[str, str]] = []  # (segment_text, lang)
    buf = ""
    lang = "zh"
    for ch in text:
        if _is_kana(ch):
            if buf:
                segments.append((buf, lang))
                buf = ""
            lang = "ja"
        elif _is_cjk(ch):
            if buf and lang == "ja":
                segments.append((buf, lang))
                buf = ""
                lang = "zh"
        buf += ch
    if buf:
        segments.append((buf, lang))

    tokens: list[Token] = []
    for seg_text, seg_lang in segments:
        if seg_lang == "ja":
            tokens.extend(_tokenize_ja(seg_text))
        else:
            tokens.extend(_tokenize_zh(seg_text))
    return tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TokenizerRouter:
    """Thread-safe tokenizer that dispatches to jieba or fugashi."""

    def tokenize(self, text: str, language: str) -> list[Token]:
        """Tokenize text according to ``language``.

        Args:
            text: Input text string.
            language: ``PAGE.language`` value from Neo4j.

        Returns:
            List of :class:`Token` objects.
        """
        if not text or not text.strip():
            return []
        lang = (language or "zh-classical").lower()
        if lang.startswith("ja") or lang == "kanbun":
            return _tokenize_ja(text)
        if lang == "mixed":
            return _tokenize_mixed(text)
        if lang == "en":
            return _tokenize_en(text)
        if lang == "ar":
            return _tokenize_ar(text)
        # zh-classical, zh-modern, unknown → jieba
        return _tokenize_zh(text)


# Module-level singleton
_router = TokenizerRouter()


def tokenize(text: str, language: str) -> list[Token]:
    """Convenience wrapper around :class:`TokenizerRouter`.

    Args:
        text: Input text.
        language: ``PAGE.language`` value.

    Returns:
        List of :class:`Token` objects.
    """
    return _router.tokenize(text, language)
