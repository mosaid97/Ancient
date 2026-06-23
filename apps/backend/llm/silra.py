"""Silra (OpenAI-compatible) client per AGENTS.md §5.

This module is the single point of contact with the Silra API. All other backend
code should import :func:`get_silra_client`, :func:`chat_completion`,
:func:`embed`, or :func:`ping` rather than instantiating ``openai.OpenAI``
directly.

Environment variables consumed (all required, loaded by the caller via
``python-dotenv`` per AGENTS.md §2):

- ``LLM_API_KEY`` — Silra API key.
- ``LLM_BASE_URL`` — e.g. ``https://api.silra.cn/v1/``.
- ``CHAT_LLM_MODEL`` — chat / reasoning model (default ``deepseek-chat``).
- ``OCR_LLM_MODEL`` — OCR-capable VLM (default ``deepseek-ocr``).
- ``EMBED_LLM_MODEL`` — embedding model (default ``text-embedding-v4``).
- ``EMBEDDING_DIMS`` — expected embedding dimension (default ``1024``).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Sequence
from typing import Any

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

ANCIENT_CHINA_SYSTEM_PROMPT: str = (
    "You are a domain-anchored assistant for the Ancient China Knowledge Graph "
    "project. The corpus is Tang-era primary sources (旧唐书, 新唐书, 通典, "
    "册府元龟, 唐律疏議箋解, 唐令拾遗 …) plus modern academic monographs that "
    "cite them. Treat classical Chinese, kanbun (Japanese-authored editorial "
    "works such as 唐令拾遗補 by 仁井田陞 / 池田温), and modern Chinese as "
    "first-class. Be conservative: when uncertain about a historical fact, say "
    "so rather than fabricate. Preserve 異體字, 避諱, and 通假字 when they are "
    "philologically meaningful — never silently rewrite source text."
)

_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def get_silra_client(
    timeout: float = 60.0,
    *,
    api_key: str | None = None,
) -> OpenAI:
    """Build an OpenAI-SDK client pointed at Silra.

    Args:
        timeout: Per-request timeout in seconds.
        api_key: Override the API key (used by per-user request handlers
            that decrypt the key from the vault). When None, falls back
            to the ``LLM_API_KEY`` env var so offline pipeline scripts
            (translation, embedding, OCR) keep working.

    Returns:
        A configured :class:`openai.OpenAI` client.

    Raises:
        RuntimeError: If no API key is available.
    """
    api_key = api_key or os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key:
        raise RuntimeError(
            "No Silra API key available. Either pass api_key=, or set "
            "LLM_API_KEY in .env for offline scripts."
        )
    if not base_url:
        raise RuntimeError("LLM_BASE_URL is not set (e.g. https://api.silra.cn/v1/).")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _retry(func, *args, max_retries: int = 3, base_delay: float = 1.0, **kwargs):
    """Retry ``func`` with exponential backoff on transient Silra errors.

    Per AGENTS.md §5: max 3 retries, exponential backoff. Honours the
    ``Retry-After`` response header on :class:`openai.RateLimitError` (HTTP 429)
    so parallel workers don't hammer the API after a rate-limit response.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                break
            delay = base_delay * (2**attempt)
            # Honour Retry-After header when the server tells us exactly how long
            # to wait (common on 429 responses from Silra under parallel load).
            if isinstance(exc, openai.RateLimitError):
                response = getattr(exc, "response", None)
                if response is not None:
                    ra = getattr(response, "headers", {}).get("retry-after")
                    if ra:
                        try:
                            delay = max(delay, float(ra))
                        except ValueError:
                            pass
            logger.warning(
                "Silra transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _ensure_system_prompt(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepend the Ancient-China system prompt if the caller didn't supply one."""
    if messages and messages[0].get("role") == "system":
        return list(messages)
    return [{"role": "system", "content": ANCIENT_CHINA_SYSTEM_PROMPT}, *messages]


def chat_completion(
    messages: Sequence[dict[str, Any]],
    model: str | None = None,
    *,
    client: OpenAI | None = None,
    max_retries: int = 3,
    **kwargs: Any,
) -> Any:
    """Send a chat-completion request with the canonical system prompt + retry.

    Args:
        messages: OpenAI-style chat messages.
        model: Override; defaults to ``CHAT_LLM_MODEL`` env var.
        client: Re-use an existing client; otherwise one is built.
        max_retries: Forwarded to :func:`_retry`.
        **kwargs: Forwarded to ``client.chat.completions.create``.

    Returns:
        The raw OpenAI ``ChatCompletion`` response.
    """
    client = client or get_silra_client()
    model_name = model or os.getenv("CHAT_LLM_MODEL", "deepseek-chat")
    full_messages = _ensure_system_prompt(messages)
    response = _retry(
        client.chat.completions.create,
        model=model_name,
        messages=full_messages,
        max_retries=max_retries,
        **kwargs,
    )
    usage = getattr(response, "usage", None)
    if usage is not None:
        logger.info(
            "silra.chat model=%s prompt=%s completion=%s total=%s",
            model_name,
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "completion_tokens", "?"),
            getattr(usage, "total_tokens", "?"),
        )
    return response


def embed(
    texts: str | Iterable[str],
    model: str | None = None,
    *,
    client: OpenAI | None = None,
    max_retries: int = 3,
    **kwargs: Any,
) -> list[list[float]]:
    """Embed one or more strings via Silra.

    Args:
        texts: One string or an iterable of strings.
        model: Override; defaults to ``EMBED_LLM_MODEL`` env var.
        client: Re-use an existing client; otherwise one is built.
        max_retries: Forwarded to :func:`_retry`.
        **kwargs: Forwarded to ``client.embeddings.create``.

    Returns:
        List of embedding vectors (always a list of lists, even for a single input).
    """
    client = client or get_silra_client()
    model_name = model or os.getenv("EMBED_LLM_MODEL", "text-embedding-v4")
    inputs = [texts] if isinstance(texts, str) else list(texts)
    if not inputs:
        return []
    response = _retry(
        client.embeddings.create,
        model=model_name,
        input=inputs,
        max_retries=max_retries,
        **kwargs,
    )
    usage = getattr(response, "usage", None)
    if usage is not None:
        logger.info(
            "silra.embed model=%s prompt=%s total=%s",
            model_name,
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "total_tokens", "?"),
        )
    return [item.embedding for item in response.data]


def ping(*, client: OpenAI | None = None) -> dict[str, Any]:
    """Lightweight health probe for Silra: 1-token chat + 1-text embed.

    Both calls are cheap (a few tokens each). Confirms the API is
    reachable and that the configured chat + embedding models exist.

    Returns:
        A dict with shape::

            {
                "ok": bool,
                "base_url": str,
                "chat_model": str,
                "embed_model": str,
                "ocr_model": str,           # only configured, not exercised
                "embedding_dims_expected": int,
                "embedding_dims_observed": int,
                "chat_sample": str,
                "errors": list[str],
            }
    """
    errors: list[str] = []
    base_url = os.getenv("LLM_BASE_URL", "")
    chat_model = os.getenv("CHAT_LLM_MODEL", "deepseek-chat")
    embed_model = os.getenv("EMBED_LLM_MODEL", "text-embedding-v4")
    ocr_model = os.getenv("OCR_LLM_MODEL", "deepseek-ocr")
    embedding_dims_expected = int(os.getenv("EMBEDDING_DIMS", "1024"))

    chat_sample: str = ""
    embedding_dims_observed: int = 0

    try:
        client = client or get_silra_client(timeout=30.0)
    except RuntimeError as exc:
        errors.append(f"client init failed: {exc}")
        return {
            "ok": False,
            "base_url": base_url,
            "chat_model": chat_model,
            "embed_model": embed_model,
            "ocr_model": ocr_model,
            "embedding_dims_expected": embedding_dims_expected,
            "embedding_dims_observed": 0,
            "chat_sample": "",
            "errors": errors,
        }

    try:
        resp = chat_completion(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            client=client,
            max_tokens=8,
            temperature=0.0,
        )
        chat_sample = (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 — surface every reason in errors[]
        errors.append(f"chat probe failed ({type(exc).__name__}): {exc}")

    try:
        vectors = embed(["唐"], client=client)
        embedding_dims_observed = len(vectors[0]) if vectors else 0
        if (
            embedding_dims_observed
            and embedding_dims_observed != embedding_dims_expected
        ):
            errors.append(
                "EMBEDDING_DIMS mismatch: env says "
                f"{embedding_dims_expected} but {embed_model} returned "
                f"{embedding_dims_observed}; update .env or schema."
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"embed probe failed ({type(exc).__name__}): {exc}")

    return {
        "ok": not errors,
        "base_url": base_url,
        "chat_model": chat_model,
        "embed_model": embed_model,
        "ocr_model": ocr_model,
        "embedding_dims_expected": embedding_dims_expected,
        "embedding_dims_observed": embedding_dims_observed,
        "chat_sample": chat_sample,
        "errors": errors,
    }
