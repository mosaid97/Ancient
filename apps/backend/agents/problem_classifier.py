"""Problem Classifier — Phase 4 (plan §6 Stage 4).

Given the OCR outputs from all three engines and the fused text for a PAGE,
calls ``deepseek-chat`` with a few-shot prompt drawn from
``data/seeds/problem_classes.yaml`` to classify the disagreement into one
of the canonical PROBLEM_CLASS codes.

The classifier is called by :mod:`apps.backend.agents.evaluator` only when
inter-engine CER exceeds a threshold, keeping Silra API costs low.

Public API
----------
load_problem_classes(path) -> list[ProblemClassDef]
ProblemClassDef                   — data class for one class definition
ClassificationResult              — output of classify_page()
classify_page(paddle, fused, qwen, deepseek, *, language, context, client)
    -> ClassificationResult
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from apps.backend.llm.silra import ANCIENT_CHINA_SYSTEM_PROMPT, chat_completion, get_silra_client

log = logging.getLogger(__name__)

_SEEDS_PATH = Path(__file__).parents[3] / "data" / "seeds" / "problem_classes.yaml"

# All valid class codes — used to coerce out-of-vocabulary LLM responses.
_ZH_CLASSES = frozenset({
    "OK", "RARE_GLYPH", "DEGRADATION", "LAYOUT_AMBIGUITY",
    "READING_ORDER", "ANNOTATION", "POLYSEMY", "CULTURAL_REFERENCE",
    "EDITORIAL_VS_SOURCE",
})
_JP_CLASSES = frozenset({
    "KANBUN_KUNTEN", "OKURIGANA", "HENTAIGANA", "MIXED_SCRIPT",
})
_ALL_CLASSES = _ZH_CLASSES | _JP_CLASSES


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FewShotExample:
    """One few-shot example for the classifier prompt."""
    context: str
    paddle_excerpt: str
    fused_excerpt: str
    problem_class: str
    reasoning: str


@dataclass
class ProblemClassDef:
    """Definition of a single problem class."""
    code: str
    label: str
    description: str
    language: str       # 'zh' or 'ja'
    routing: str        # 'pass' | 'needs_review' | 'failed'
    few_shots: list[FewShotExample] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Output of :func:`classify_page`."""
    problem_class: str                  # one of _ALL_CLASSES
    confidence: float                   # 0–1 (LLM self-reported, treated as ordinal only)
    reasoning: str                      # LLM's one-sentence justification
    raw_response: str = ""             # raw LLM output for audit


# ---------------------------------------------------------------------------
# Seed loader
# ---------------------------------------------------------------------------


def load_problem_classes(path: Path | None = None) -> list[ProblemClassDef]:
    """Load problem class definitions from the YAML seed file.

    Args:
        path: Override path to ``problem_classes.yaml``.

    Returns:
        List of :class:`ProblemClassDef` objects.
    """
    p = path or _SEEDS_PATH
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    defs: list[ProblemClassDef] = []
    for entry in raw.get("classes", []):
        few_shots = [
            FewShotExample(
                context=ex.get("context", ""),
                paddle_excerpt=ex.get("paddle_excerpt", ""),
                fused_excerpt=ex.get("fused_excerpt", ""),
                problem_class=ex.get("class", "OK"),
                reasoning=ex.get("reasoning", ""),
            )
            for ex in entry.get("few_shots", [])
        ]
        defs.append(
            ProblemClassDef(
                code=entry["code"],
                label=entry["label"],
                description=entry.get("description", ""),
                language=entry.get("language", "zh"),
                routing=entry.get("routing", "pass"),
                few_shots=few_shots,
            )
        )
    return defs


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = (
    ANCIENT_CHINA_SYSTEM_PROMPT
    + "\n\n"
    "你是古代漢語OCR品質分類助手，負責對比多引擎OCR結果並識別問題類型。\n"
    "分析提供的OCR文本片段，返回最符合的問題類別代碼。\n"
    "只返回 JSON，不要包含任何 markdown 代碼塊。"
)

_CLASSIFIER_USER_TEMPLATE = """\
文檔背景：{context}
頁面語言：{language}

PaddleOCR結果（前200字）：
{paddle}

融合結果（前200字）：
{fused}

{qwen_section}

{deepseek_section}

問題類別選項（按語言 {language}）：
{class_options}

少量樣本示例：
{few_shot_block}

請返回 JSON：
{{
  "problem_class": "<代碼>",
  "confidence": <0.0–1.0>,
  "reasoning": "<一句話理由（中文）>"
}}
"""


def _build_few_shot_block(defs: list[ProblemClassDef], language: str) -> str:
    """Build a few-shot block from relevant class definitions."""
    lines: list[str] = []
    for d in defs:
        if d.language not in (language[:2], "zh"):
            continue
        for ex in d.few_shots[:1]:  # one shot per class keeps the prompt short
            lines.append(
                f"示例 [{d.code}] 背景：{ex.context}\n"
                f"  PaddleOCR：{ex.paddle_excerpt[:60]}\n"
                f"  Fused：{ex.fused_excerpt[:60]}\n"
                f"  → 類別：{d.code}（{ex.reasoning}）"
            )
    return "\n\n".join(lines) if lines else "（無示例）"


def _build_class_options(defs: list[ProblemClassDef], language: str) -> str:
    """Build a compact class-options block for the prompt."""
    lines: list[str] = []
    for d in defs:
        if language.startswith("ja"):
            if d.language not in ("ja", "zh"):
                continue
        else:
            if d.language == "ja":
                continue
        lines.append(f"  {d.code}: {d.label} — {d.description[:80].strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classification call
# ---------------------------------------------------------------------------


def classify_page(
    paddle_text: str,
    fused_text: str,
    qwen_text: str | None,
    deepseek_text: str | None,
    *,
    language: str = "zh-classical",
    context: str = "",
    defs: list[ProblemClassDef] | None = None,
    client: OpenAI | None = None,
    model: str | None = None,
) -> ClassificationResult:
    """Classify an OCR page's problem class via deepseek-chat.

    Args:
        paddle_text: PaddleOCR output for the page.
        fused_text: Fused output (authoritative text).
        qwen_text: Qwen-VL-OCR output (optional).
        deepseek_text: DeepSeek-OCR output (optional).
        language: ``PAGE.language`` value (e.g. ``'zh-classical'``, ``'ja'``).
        context: Brief document context (e.g. title + chapter).
        defs: Pre-loaded :class:`ProblemClassDef` list; loaded from seed if None.
        client: Re-use an existing Silra client.
        model: Override chat model.

    Returns:
        :class:`ClassificationResult` with ``problem_class``, ``confidence``,
        ``reasoning``.
    """
    if defs is None:
        defs = load_problem_classes()

    llm_client = client or get_silra_client()
    lang_prefix = language[:2]  # 'zh' or 'ja'

    qwen_section = (
        f"Qwen-VL-OCR結果（前200字）：\n{qwen_text[:200]}"
        if qwen_text
        else ""
    )
    deepseek_section = (
        f"DeepSeek-OCR結果（前200字）：\n{deepseek_text[:200]}"
        if deepseek_text
        else ""
    )

    user_msg = _CLASSIFIER_USER_TEMPLATE.format(
        context=context[:120] or "（未知文檔）",
        language=language,
        paddle=paddle_text[:200] if paddle_text else "（空）",
        fused=fused_text[:200] if fused_text else "（空）",
        qwen_section=qwen_section,
        deepseek_section=deepseek_section,
        class_options=_build_class_options(defs, lang_prefix),
        few_shot_block=_build_few_shot_block(defs, lang_prefix),
    )

    messages = [
        {"role": "system", "content": _CLASSIFIER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    raw = ""
    try:
        resp = chat_completion(messages, model=model, client=llm_client, temperature=0.0, max_tokens=256)
        raw = resp.choices[0].message.content or ""
    except Exception as exc:
        log.warning("classify_page: LLM call failed: %s — defaulting to OK", exc)
        return ClassificationResult(
            problem_class="OK",
            confidence=0.5,
            reasoning=f"LLM call failed: {exc}",
            raw_response="",
        )

    # Parse JSON from response
    result = _parse_classification(raw)
    result.raw_response = raw
    return result


def _parse_classification(raw: str) -> ClassificationResult:
    """Extract classification JSON from LLM response."""
    import json

    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        log.warning("classify_page: no JSON found in response: %r", raw[:100])
        return ClassificationResult(problem_class="OK", confidence=0.5, reasoning="parse failed")

    try:
        obj = json.loads(text[start : end + 1])
    except Exception as exc:
        log.warning("classify_page: JSON parse error: %s — raw: %r", exc, raw[:100])
        return ClassificationResult(problem_class="OK", confidence=0.5, reasoning="JSON parse error")

    code = obj.get("problem_class", "OK").strip().upper()
    if code not in _ALL_CLASSES:
        log.debug("classify_page: unknown class %r, coercing to OK", code)
        code = "OK"

    return ClassificationResult(
        problem_class=code,
        confidence=max(0.0, min(1.0, float(obj.get("confidence", 0.5)))),
        reasoning=str(obj.get("reasoning", "")).strip()[:300],
    )
