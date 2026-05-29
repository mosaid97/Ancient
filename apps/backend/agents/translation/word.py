"""Word-level analysis agent — Phase 6 (plan §6 Stage 6).

Step 1 of the primary-tier translation pipeline:
1. Tokenize via :func:`apps.backend.lang.tokenizer.tokenize`.
2. Apply the §2.7 normalization pipeline (异體字 + 避諱 per era + 通假字 ON).
3. Look up every token in DICTIONARY_ENTRY (language-filtered).
4. Resolve polysemy via ``deepseek-chat`` for tokens with ≥2 candidate meanings.

Returns ``WordAnalysisResult``:
  tokens       — list of ``AnnotatedToken`` (surface, lemma, pos, definition, reading)
  text_canonical — post-§2.7 normalized text (also used by paragraph.py)

This module is skipped for ``tier == 'secondary'``.

Public API
----------
AnnotatedToken     — token + dict entry + polysemy resolution
WordAnalysisResult — result dataclass
analyze_words(text, language, era, tier, driver, client) -> WordAnalysisResult
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from neo4j import Driver
from openai import OpenAI

from apps.backend.kb.dictionary import DictEntry, lookup_term
from apps.backend.lang.tokenizer import Token, tokenize
from apps.backend.llm.silra import ANCIENT_CHINA_SYSTEM_PROMPT, get_silra_client
from apps.backend.normalize.pipeline import normalize_canonical

log = logging.getLogger(__name__)

_POLYSEMY_SYSTEM = (
    ANCIENT_CHINA_SYSTEM_PROMPT
    + "\n\n你是一位古漢語詞義辨析專家。根據給定的上下文句子，從候選義項中選出最符合語境的詞義。"
    "必須從候選列表中選擇，不能創造新義項。"
)

_MAX_CANDIDATES = 5   # max dict lookups to send to LLM for polysemy


@dataclass
class AnnotatedToken:
    """A tokenized word enriched with dictionary and polysemy information."""

    surface: str
    lemma: str
    pos: str
    reading: str = ""
    language: str = "zh"
    definition: str = ""          # resolved meaning (empty if not in dict)
    dict_candidates: list[str] = field(default_factory=list)
    polysemy_resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "surface": self.surface,
            "lemma": self.lemma,
            "pos": self.pos,
            "language": self.language,
        }
        if self.reading:
            d["reading"] = self.reading
        if self.definition:
            d["definition"] = self.definition
        if self.polysemy_resolved:
            d["polysemy_resolved"] = True
        return d


@dataclass
class WordAnalysisResult:
    """Result of word-level analysis for one text passage."""

    text_canonical: str               # §2.7 normalized text
    tokens: list[AnnotatedToken]
    normalization_steps: list[str]    # steps applied by normalize_canonical

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_canonical": self.text_canonical,
            "token_count": len(self.tokens),
            "normalization_steps": self.normalization_steps,
            "tokens": [t.to_dict() for t in self.tokens],
        }


# ---------------------------------------------------------------------------
# Polysemy resolution
# ---------------------------------------------------------------------------

def _resolve_polysemy(
    token: Token,
    candidates: list[DictEntry],
    context: str,
    client: OpenAI,
    model: str,
) -> str:
    """Ask deepseek-chat to pick the best meaning given sentence context."""
    options = [
        f"{i + 1}. {e.meaning}"
        for i, e in enumerate(candidates[:_MAX_CANDIDATES])
    ]
    prompt = (
        f"上下文句子：\n「{context[:300]}」\n\n"
        f"詞語：「{token.surface}」\n\n"
        f"候選義項：\n" + "\n".join(options) + "\n\n"
        "請回答數字（如 1、2、3），選出最符合上下文的義項。只回答數字，不要解釋。"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _POLYSEMY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=8,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        idx = int(raw.strip("1234567890")) if raw.strip() not in "1234567890" else int(raw) - 1
        # More robust: extract first digit
        import re
        m = re.search(r"\d", raw)
        if m:
            idx = int(m.group()) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx].meaning
    except Exception as exc:
        log.debug("_resolve_polysemy failed for %r: %s", token.surface, exc)
    return candidates[0].meaning  # fallback to first candidate


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def analyze_words(
    text: str,
    language: str,
    era: str | None,
    tier: str,
    driver: Driver,
    *,
    client: OpenAI | None = None,
    model: str | None = None,
) -> WordAnalysisResult:
    """Run word-level analysis on a text passage.

    Args:
        text: Raw text (post-OCR or native).
        language: ``PAGE.language`` value.
        era: Era tag for 避諱 selection (e.g. ``'Tang'``); ``None`` skips taboo.
        tier: ``'primary'`` or ``'secondary'``.
        driver: Open Neo4j driver.
        client: Optional Silra client (new one created if None).
        model: Override chat model.

    Returns:
        :class:`WordAnalysisResult` with canonical text and annotated tokens.
    """
    import os
    c = client or get_silra_client()
    m = model or os.getenv("CHAT_LLM_MODEL", "deepseek-chat")

    # Step 1: §2.7 normalization (通假字 ON for translator per AGENTS.md rule)
    norm_result = normalize_canonical(
        text,
        lang=language.replace("zh-", "zh").replace("kanbun", "ja"),
        era=era or "",
        apply_loan=True,
    )
    canonical = norm_result.canonical
    steps_applied = [step.name for step in norm_result.steps if step.applied]

    # Step 2: tokenize canonical text
    raw_tokens = tokenize(canonical, language)

    # Step 3 + 4: dict lookup + polysemy resolution
    lang_key = "ja" if language.startswith("ja") or language == "kanbun" else "zh"
    annotated: list[AnnotatedToken] = []
    for tok in raw_tokens:
        entries = lookup_term(driver, tok.lemma, lang_key)
        if not entries:
            # fallback: fuzzy for 2+ char CJK tokens
            if len(tok.lemma) >= 2:
                entries = lookup_term(driver, tok.lemma, lang_key, fuzzy=True)

        if not entries:
            annotated.append(AnnotatedToken(
                surface=tok.surface,
                lemma=tok.lemma,
                pos=tok.pos,
                reading=tok.reading,
                language=tok.language,
            ))
        elif len(entries) == 1:
            annotated.append(AnnotatedToken(
                surface=tok.surface,
                lemma=tok.lemma,
                pos=tok.pos,
                reading=tok.reading,
                language=tok.language,
                definition=entries[0].meaning,
                dict_candidates=[e.meaning for e in entries],
            ))
        else:
            # Multiple candidates — resolve polysemy
            resolved = _resolve_polysemy(tok, entries, canonical, c, m)
            annotated.append(AnnotatedToken(
                surface=tok.surface,
                lemma=tok.lemma,
                pos=tok.pos,
                reading=tok.reading,
                language=tok.language,
                definition=resolved,
                dict_candidates=[e.meaning for e in entries],
                polysemy_resolved=True,
            ))

    polysemy_count = sum(1 for t in annotated if t.polysemy_resolved)
    log.info(
        "analyze_words: %d tokens, %d polysemy resolved, era=%s, tier=%s",
        len(annotated), polysemy_count, era, tier,
    )

    return WordAnalysisResult(
        text_canonical=canonical,
        tokens=annotated,
        normalization_steps=steps_applied,
    )
