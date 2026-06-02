"""User-supplied custom OCR engine (generic OpenAI-compatible vision endpoint).

Track E feature: when a user uploads a document and provides their own
``base_url`` + ``api_key`` + ``model`` for an OpenAI-compatible vision model,
that engine is run as an **additional** OCR comparison column alongside the
built-in Paddle / Qwen-VL / DeepSeek engines.

This module deliberately mirrors :mod:`apps.backend.ocr.qwen_vl` so the
fusion + HITL surfaces treat the custom output identically:

- same :class:`OCRPageResult` shape, engine label ``'custom_ocr'``;
- the inline hallucination guard :func:`apps.backend.ocr.validate.apply_validation`
  is **always** applied before returning (AGENTS.md 2026-05-19 rule: every
  LLM OCR engine must call ``apply_validation``);
- the same script-aware prompt selection as Qwen-VL.

Credentials are **never persisted**: the caller passes them per-call (the
upload runner reads them from the job record / env and discards them after).
"""
from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any

import cv2
import numpy as np
from openai import OpenAI

from apps.backend.ocr.base import OCRLine, OCRPageResult
from apps.backend.ocr.qwen_vl import _system_prompt
from apps.backend.ocr.validate import apply_validation

logger = logging.getLogger(__name__)

_ENGINE = "custom_ocr"


def _encode_image(image: np.ndarray | bytes, *, fmt: str = "png") -> str:
    """Return a ``data:image/<fmt>;base64,...`` URI."""
    if isinstance(image, np.ndarray):
        ok, buf = cv2.imencode(f".{fmt}", image)
        if not ok:
            raise RuntimeError(f"cv2.imencode failed for ({fmt})")
        payload = buf.tobytes()
    elif isinstance(image, (bytes, bytearray, memoryview)):
        payload = bytes(image)
    elif isinstance(image, io.IOBase):
        payload = image.read()
    else:
        raise TypeError(f"_encode_image: unsupported input type {type(image).__name__}")
    b64 = base64.b64encode(payload).decode("ascii")
    return f"data:image/{fmt};base64,{b64}"


def make_custom_client(*, base_url: str, api_key: str, timeout: float = 180.0) -> OpenAI:
    """Build an OpenAI-compatible client for a user-supplied endpoint."""
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def custom_ocr_page(
    image: np.ndarray | bytes,
    *,
    page_id: str,
    model: str,
    client: OpenAI,
    language_hint: str | None = "zh-classical",
    script_hint: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> OCRPageResult:
    """OCR one page via a user-supplied OpenAI-compatible vision model.

    Args:
        image: Preprocessed page image (BGR ndarray or PNG/JPEG bytes).
        page_id: Neo4j ``PAGE.id`` (stored on the result).
        model: User-supplied model name.
        client: A pre-built client from :func:`make_custom_client`.
        language_hint: Used for prompt selection when ``script_hint`` is None.
        script_hint: ``'traditional'`` | ``'simplified'`` | ``'classical'``.
        max_tokens: Response cap.
        temperature: 0.0 for deterministic transcription.

    Returns:
        An :class:`OCRPageResult` with ``engine='custom_ocr'``.
    """
    started = time.monotonic()
    data_uri = _encode_image(image, fmt="png")
    system_prompt = _system_prompt(language_hint, script_hint=script_hint)

    if script_hint == "traditional" or (
        script_hint is None
        and language_hint not in {"ja", "kanbun", "japan", "jpn", "zh-modern"}
    ):
        user_instruction = "請按上述要求識別本頁面所有字符，只輸出原文。"
    else:
        user_instruction = "请按上述要求识别本页面所有字符，只输出原文。"

    user_content = [
        {"type": "image_url", "image_url": {"url": data_uri}},
        {"type": "text", "text": user_instruction},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - started
        logger.warning("custom_ocr_page %s failed: %s", page_id, exc)
        return OCRPageResult(
            engine=_ENGINE,
            model_version=model,
            page_id=page_id,
            text="",
            confidence=0.0,
            char_count=0,
            language_hint=language_hint,
            duration_seconds=round(elapsed, 3),
            metadata={"custom": True},
            error=f"{type(exc).__name__}: {exc}",
        )

    elapsed = time.monotonic() - started
    raw_text = (response.choices[0].message.content or "").strip()

    # Inline hallucination guard — mandatory for every LLM OCR engine.
    text, validation_error = apply_validation(
        raw_text,
        expected_script=script_hint if script_hint == "traditional" else None,
    )

    lines: list[OCRLine] = []
    for idx, line in enumerate(text.split("\n")):
        chunk = line.strip()
        if chunk:
            lines.append(OCRLine(text=chunk, confidence=1.0, order=idx))

    metadata: dict[str, Any] = {"custom": True, "model": model}
    if validation_error:
        metadata["validation_error"] = validation_error

    confidence = 0.80 if text else 0.0

    return OCRPageResult(
        engine=_ENGINE,
        model_version=model,
        page_id=page_id,
        text=text,
        lines=lines,
        confidence=confidence,
        char_count=len(text),
        language_hint=language_hint,
        duration_seconds=round(elapsed, 3),
        metadata=metadata,
        error=validation_error,
    )
