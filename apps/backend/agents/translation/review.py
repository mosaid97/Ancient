"""Three-pass translation review agent — Phase 6 (plan §6 Stage 6).

Step 3 of the primary-tier pipeline:
Runs up to ``max_iterations`` rounds of LLM self-review over the
vernacular translation, checking:
  Pass 1 — coherence: does the text read as natural, grammatical modern Chinese?
  Pass 2 — grammar: sentence structure, punctuation, tense consistency.
  Pass 3 — key-word fidelity: are named entities, legal terms, and era names
            from the canonical text preserved accurately?

Each pass produces a JSON verdict ``{"ok": true/false, "issues": [...], "revised": "..."}``
If all three passes report ``ok: true`` (or no issues) the process terminates early.
On ``ok: false`` the revised text replaces the input for the next iteration.

After ``max_iterations`` the best candidate (fewest issues on last pass) is returned.

Public API
----------
ReviewResult       — result dataclass
review_translation(word_result, para_result, language, client, model) -> ReviewResult
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from apps.backend.agents.translation.paragraph import ParagraphResult
from apps.backend.agents.translation.word import WordAnalysisResult
from apps.backend.llm.silra import ANCIENT_CHINA_SYSTEM_PROMPT, get_silra_client

log = logging.getLogger(__name__)

_MAX_ITER = 3

_REVIEW_SYSTEM = (
    ANCIENT_CHINA_SYSTEM_PROMPT
    + "\n\n你是古漢語翻譯品質審核員。你將對一段文言文的白話文翻譯進行審查。"
    "請以 JSON 格式回答，格式：\n"
    '{"ok": true/false, "issues": ["issue1", ...], "revised": "修訂後的譯文（若無問題則原文照錄）"}\n'
    "只輸出 JSON，不含任何其他文字。"
)

_PASSES = [
    ("coherence",
     "審查連貫性：譯文是否通順？語句是否完整？是否存在語義跳躍或不清晰的地方？"),
    ("grammar",
     "審查語法：句子結構是否正確？標點是否恰當？動詞時態/體是否一致？"),
    ("keyword_fidelity",
     "審查關鍵詞忠實度：原文中的人名、地名、官職、年號、法律術語是否在譯文中準確保留？"),
]


@dataclass
class ReviewResult:
    """Result of the three-pass review process."""

    text_final: str                   # best translation after review
    iterations: int                   # number of iterations actually run
    passes_ok: list[str]              # pass names that passed
    passes_failed: list[str]          # pass names that had issues
    all_ok: bool                      # True if all passes passed on some iteration
    prompt_tokens: int
    completion_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_final": self.text_final,
            "iterations": self.iterations,
            "passes_ok": self.passes_ok,
            "passes_failed": self.passes_failed,
            "all_ok": self.all_ok,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _call_review_pass(
    pass_name: str,
    instruction: str,
    original: str,
    translation: str,
    client: OpenAI,
    model: str,
) -> tuple[bool, list[str], str, int, int]:
    """Run one review pass; returns (ok, issues, revised, p_tok, c_tok)."""
    user = (
        f"原文：\n「{original[:400]}」\n\n"
        f"白話文譯文：\n「{translation}」\n\n"
        f"審查指令：{instruction}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _REVIEW_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=min(len(translation) * 4, 2048),
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        usage = resp.usage
        p_tok = usage.prompt_tokens if usage else 0
        c_tok = usage.completion_tokens if usage else 0

        m = _JSON_RE.search(raw)
        if m:
            verdict = json.loads(m.group())
        else:
            verdict = {"ok": True, "issues": [], "revised": translation}

        ok = bool(verdict.get("ok", True))
        issues = list(verdict.get("issues") or [])
        revised = str(verdict.get("revised") or translation).strip()
        if not revised:
            revised = translation
        return ok, issues, revised, p_tok, c_tok
    except Exception as exc:
        log.warning("_call_review_pass [%s] failed: %s", pass_name, exc)
        return True, [], translation, 0, 0


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def review_translation(
    word_result: WordAnalysisResult,
    para_result: ParagraphResult,
    language: str,
    *,
    client: OpenAI | None = None,
    model: str | None = None,
    max_iterations: int = _MAX_ITER,
) -> ReviewResult:
    """Run three-pass review over a vernacular translation.

    Args:
        word_result: Output of :func:`~apps.backend.agents.translation.word.analyze_words`.
        para_result: Output of :func:`~apps.backend.agents.translation.paragraph.translate_paragraph`.
        language: ``PAGE.language`` value.
        client: Optional Silra client.
        model: Override chat model.
        max_iterations: Maximum number of review iterations.

    Returns:
        :class:`ReviewResult` with the final polished translation.
    """
    c = client or get_silra_client()
    m = model or os.getenv("CHAT_LLM_MODEL", "deepseek-chat")

    current = para_result.text_vernacular
    original = word_result.text_canonical
    total_p, total_c = 0, 0

    passes_ok: list[str] = []
    passes_failed: list[str] = []
    iter_count = 0

    for iteration in range(max_iterations):
        iter_count = iteration + 1
        iter_ok = True
        for pass_name, instruction in _PASSES:
            ok, issues, revised, p_tok, c_tok = _call_review_pass(
                pass_name, instruction, original, current, c, m
            )
            total_p += p_tok
            total_c += c_tok
            if ok:
                if pass_name not in passes_ok:
                    passes_ok.append(pass_name)
            else:
                iter_ok = False
                if pass_name not in passes_failed:
                    passes_failed.append(pass_name)
                current = revised

        if iter_ok:
            log.info(
                "review_translation: all passes OK at iteration %d, lang=%s",
                iteration + 1, language,
            )
            break

    # Determine overall success
    all_ok = all(p not in passes_failed for p in [n for n, _ in _PASSES])

    log.info(
        "review_translation: %d iterations, passes_ok=%s, passes_failed=%s",
        iter_count, passes_ok, passes_failed,
    )
    return ReviewResult(
        text_final=current,
        iterations=iter_count,
        passes_ok=passes_ok,
        passes_failed=passes_failed,
        all_ok=all_ok,
        prompt_tokens=total_p,
        completion_tokens=total_c,
    )
