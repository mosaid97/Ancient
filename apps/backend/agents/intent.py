"""C3: Query intent classifier (plan §0.6 C3).

Classifies queries into:
  'factual'    — specific fact lookup; expects primary-tier chunks; tier-boost +0.15 primary
  'interpretive' — scholarly interpretation; expects secondary-tier; tier-boost +0.15 secondary
  'synthesis'  — broad thematic query; routes via community summaries
  'unknown'    — no strong signal; neutral retrieval

Classification uses a fast LLM call (deepseek-chat) with a structured prompt.
Falls back to 'unknown' on any error so retrieval is never blocked.
"""
from __future__ import annotations

import logging
import os
from typing import Literal

from openai import OpenAI

log = logging.getLogger(__name__)

IntentType = Literal["factual", "interpretive", "synthesis", "unknown"]

_INTENT_PROMPT = """\
你是一个古代汉语文献检索系统的查询分类器。请将以下查询分类为以下四种意图之一：
- factual: 查询特定的历史事实、法律条文、人物信息或年代记录（期望从原典中找到直接证据）
- interpretive: 查询学术解读、评注、考证或现代学者的分析（期望从二手文献中找到解释）
- synthesis: 宽泛的主题性查询，需要综合多篇文献的信息（如某朝代的整体制度）
- unknown: 无法判断或混合类型

仅回复一个词：factual、interpretive、synthesis 或 unknown。

查询：{query}"""


def classify_intent(query: str, *, model: str | None = None) -> IntentType:
    """Classify query intent via LLM call.

    Returns 'unknown' on any API error so retrieval is never blocked.
    """
    llm_model = model or os.getenv("LLM_MODEL", "deepseek-chat")
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.silra.cn/v1/"),
    )
    try:
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": _INTENT_PROMPT.format(query=query)}],
            max_tokens=10,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip().lower()
        if raw in ("factual", "interpretive", "synthesis"):
            return raw  # type: ignore[return-value]
        return "unknown"
    except Exception as exc:
        log.warning("Intent classification failed (defaulting to unknown): %s", exc)
        return "unknown"


def tier_boost(
    hits: list[tuple[str, float]],
    chunk_tiers: dict[str, str | None],
    intent: IntentType,
    *,
    boost: float = 0.15,
) -> list[tuple[str, float]]:
    """Apply tier-boost to scores based on query intent.

    factual     → +boost for primary chunks
    interpretive → +boost for secondary chunks
    synthesis / unknown → no boost
    """
    if intent not in ("factual", "interpretive"):
        return hits
    favoured_tier = "primary" if intent == "factual" else "secondary"
    return [
        (cid, score + (boost if chunk_tiers.get(cid) == favoured_tier else 0.0))
        for cid, score in hits
    ]
