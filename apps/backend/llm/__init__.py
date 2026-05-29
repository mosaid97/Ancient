"""LLM clients (Silra, OpenAI-compatible)."""

from apps.backend.llm.silra import (
    ANCIENT_CHINA_SYSTEM_PROMPT,
    chat_completion,
    embed,
    get_silra_client,
    ping,
)

__all__ = [
    "ANCIENT_CHINA_SYSTEM_PROMPT",
    "chat_completion",
    "embed",
    "get_silra_client",
    "ping",
]
