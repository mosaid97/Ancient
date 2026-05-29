"""Paragraph-level vernacular translation — Phase 6 (plan §6 Stage 6).

Step 2 of the primary-tier pipeline:
1. Receives canonical text + word annotations from ``word.py``.
2. Fetches applicable NORM rules from the KB via Cypher RAG.
3. Produces ``text_vernacular`` (always modern Chinese).
4. For ja/kanbun pages: ALSO produces ``text_vernacular_ja`` (display-only).

The LLM call uses a single ``deepseek-chat`` completion with:
- The word annotations as structured context.
- The relevant NORM rules as a guideline list.
- Strict instructions: preserve 避諱-markers in parentheses; do NOT add
  content beyond what the source text says; stay factual.

Public API
----------
ParagraphResult        — result with text_vernacular + optional text_vernacular_ja
translate_paragraph(word_result, language, driver, client, model) -> ParagraphResult
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from neo4j import Driver
from openai import OpenAI

from apps.backend.agents.translation.word import WordAnalysisResult
from apps.backend.kb.norms import NormEntry, get_norms
from apps.backend.llm.silra import ANCIENT_CHINA_SYSTEM_PROMPT, get_silra_client

log = logging.getLogger(__name__)

_MAX_NORMS = 12
_MAX_DEFINITION_TOKENS = 600   # cap the word-def block fed to LLM

_SYSTEM_ZH = (
    ANCIENT_CHINA_SYSTEM_PROMPT
    + "\n\n你是一位古漢語白話翻譯專家，專門將唐代古籍文言文譯成準確、流暢的現代漢語。"
    "原則：①忠實原文，不增不減；②保留避諱字標注（括號內）；③官職、律文術語加括號注解；"
    "④結合給定的詞義注釋和翻譯準則；⑤輸出純白話文，不含原文。"
)

_SYSTEM_JA = (
    ANCIENT_CHINA_SYSTEM_PROMPT
    + "\n\nあなたは漢文・古典中国語の専門家です。与えられた文章を現代日本語に訳してください。"
    "方針：①原文に忠実で過不足なく；②官職名・法律用語は括弧内に説明を付加；"
    "③送り仮名・訓点の指示に従って語順を調整する；④現代語として自然な日本語で出力する。"
)


@dataclass
class ParagraphResult:
    """Result of paragraph-level translation."""

    text_vernacular: str              # always modern Chinese
    text_vernacular_ja: str | None    # Japanese gloss (only for ja/kanbun pages)
    norms_applied: int                # how many NORM rules were in the prompt
    prompt_tokens: int                # LLM prompt tokens (0 if not available)
    completion_tokens: int            # LLM completion tokens

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "text_vernacular": self.text_vernacular,
            "norms_applied": self.norms_applied,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
        if self.text_vernacular_ja:
            d["text_vernacular_ja"] = self.text_vernacular_ja
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_definition_block(word_result: WordAnalysisResult, max_chars: int) -> str:
    """Format the word-annotation context block for the LLM prompt."""
    lines = []
    for tok in word_result.tokens:
        if tok.definition:
            lines.append(f"- 「{tok.surface}」：{tok.definition}")
    raw = "\n".join(lines)
    if len(raw) > max_chars:
        raw = raw[:max_chars] + "\n...（以下略）"
    return raw


def _build_norms_block(norms: list[NormEntry]) -> str:
    lines = [f"{i + 1}. {n.rule}" for i, n in enumerate(norms)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Translation call
# ---------------------------------------------------------------------------

def _call_vernacular(
    canonical_text: str,
    word_defs: str,
    norms_block: str,
    client: OpenAI,
    model: str,
    system: str,
) -> tuple[str, int, int]:
    """Call deepseek-chat for vernacular translation; returns (text, prompt_tok, completion_tok)."""
    user_parts = [f"原文（規範化後）：\n「{canonical_text}」"]
    if word_defs:
        user_parts.append(f"\n詞義注釋：\n{word_defs}")
    if norms_block:
        user_parts.append(f"\n翻譯準則：\n{norms_block}")
    user_parts.append("\n請輸出白話文翻譯（僅輸出譯文，不附加任何解釋）：")
    user_msg = "\n".join(user_parts)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=min(len(canonical_text) * 3, 2048),
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    usage = resp.usage
    p_tok = usage.prompt_tokens if usage else 0
    c_tok = usage.completion_tokens if usage else 0
    return text, p_tok, c_tok


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def translate_paragraph(
    word_result: WordAnalysisResult,
    language: str,
    driver: Driver,
    *,
    client: OpenAI | None = None,
    model: str | None = None,
) -> ParagraphResult:
    """Produce vernacular translation from word analysis.

    Args:
        word_result: Output of :func:`apps.backend.agents.translation.word.analyze_words`.
        language: ``PAGE.language`` value (determines norm selection + JA gloss).
        driver: Open Neo4j driver (for norm RAG).
        client: Optional Silra client.
        model: Override chat model.

    Returns:
        :class:`ParagraphResult` with ``text_vernacular`` and optional
        ``text_vernacular_ja``.
    """
    c = client or get_silra_client()
    m = model or os.getenv("CHAT_LLM_MODEL", "deepseek-chat")

    # Fetch norms for this tradition
    tradition = "zh-classical"
    is_ja = language.startswith("ja") or language == "kanbun"
    if is_ja:
        tradition = "kanbun-kundoku"

    norms_zh = get_norms(driver, "zh-classical", limit=_MAX_NORMS)
    norms_specific = get_norms(driver, tradition, limit=6) if is_ja else []
    norms = norms_zh + norms_specific

    def_block = _build_definition_block(word_result, _MAX_DEFINITION_TOKENS)
    norms_block = _build_norms_block(norms)

    # Produce modern Chinese vernacular (always)
    text_zh, p_tok, c_tok = _call_vernacular(
        word_result.text_canonical,
        def_block,
        norms_block,
        c,
        m,
        _SYSTEM_ZH,
    )

    # Optionally produce Japanese gloss for ja/kanbun pages
    text_ja: str | None = None
    if is_ja:
        try:
            ja_norms = get_norms(driver, "kanbun-kundoku", limit=_MAX_NORMS)
            text_ja, p2, c2 = _call_vernacular(
                word_result.text_canonical,
                def_block,
                _build_norms_block(ja_norms),
                c,
                m,
                _SYSTEM_JA,
            )
            p_tok += p2
            c_tok += c2
        except Exception as exc:
            log.warning("translate_paragraph: JA gloss failed: %s", exc)

    log.info(
        "translate_paragraph: lang=%s norms=%d p_tok=%d c_tok=%d",
        language, len(norms), p_tok, c_tok,
    )
    return ParagraphResult(
        text_vernacular=text_zh,
        text_vernacular_ja=text_ja,
        norms_applied=len(norms),
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
    )
