"""C3: HyDE (Hypothetical Document Embedding) query expansion (plan §0.6 C3).

Generates a ~150-token hypothetical Classical-Chinese passage that would
answer the query, then embeds it. The HyDE vector is used alongside the
original query vector in dense retrieval for better recall on short queries.

Reference: Gao et al. 2022 "Precise Zero-Shot Dense Retrieval without
Relevance Labels" (HyDE). Adapted for classical Chinese 古籍 retrieval.
"""
from __future__ import annotations

import logging
import os

from openai import OpenAI

log = logging.getLogger(__name__)

_HYDE_PROMPT = """\
请你扮演一位唐代法学专家，根据以下现代汉语问题，写一段约150字的古典汉语文献原文，\
该文献内容应能直接回答这个问题。只输出古典汉语原文，不要加任何解释。

问题：{query}

古典汉语原文："""

_HYDE_MAX_TOKENS = 200


def generate_hyde_passage(query: str, *, model: str | None = None) -> str | None:
    """Generate a hypothetical Classical-Chinese passage for HyDE.

    Returns the generated passage string, or None on failure.
    """
    llm_model = model or os.getenv("LLM_MODEL", "deepseek-chat")
    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.silra.cn/v1/"),
    )
    try:
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
            max_tokens=_HYDE_MAX_TOKENS,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.warning("HyDE generation failed (will use original query only): %s", exc)
        return None
