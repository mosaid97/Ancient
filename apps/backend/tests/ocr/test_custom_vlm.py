"""Tests for the user-supplied custom OCR engine (ocr/custom_vlm.py, Track E).

No real network calls — the OpenAI-compatible client is faked. Covers the
success path, the mandatory ``apply_validation`` hallucination guard, the
API-error path, and image encoding.
"""
from __future__ import annotations

import numpy as np
import pytest

from apps.backend.ocr.custom_vlm import _ENGINE, _encode_image, custom_ocr_page


# ─────────────────────────────────────────────────────────────────────────────
# Fake OpenAI-compatible client
# ─────────────────────────────────────────────────────────────────────────────

class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, *, content: str = "", raise_exc: Exception | None = None) -> None:
        self._content = content
        self._raise = raise_exc
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):  # noqa: ANN003
        self.last_kwargs = kwargs
        if self._raise is not None:
            raise self._raise
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, *, content: str = "", raise_exc: Exception | None = None) -> None:
        self.chat = _FakeChat(_FakeCompletions(content=content, raise_exc=raise_exc))


def _blank_image() -> np.ndarray:
    return np.zeros((12, 12, 3), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# _encode_image
# ─────────────────────────────────────────────────────────────────────────────

def test_encode_image_ndarray_returns_data_uri():
    uri = _encode_image(_blank_image(), fmt="png")
    assert uri.startswith("data:image/png;base64,")
    assert len(uri) > len("data:image/png;base64,")


def test_encode_image_bytes_passthrough():
    raw = b"\x89PNG\r\n\x1a\n" + b"0" * 16
    uri = _encode_image(raw, fmt="png")
    assert uri.startswith("data:image/png;base64,")


def test_encode_image_rejects_unknown_type():
    with pytest.raises(TypeError):
        _encode_image(12345)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# custom_ocr_page — success
# ─────────────────────────────────────────────────────────────────────────────

def test_custom_ocr_page_success_clean_cjk():
    client = _FakeClient(content="唐律疏議卷第一名例律")
    result = custom_ocr_page(
        _blank_image(), page_id="doc::p00000", model="my-vlm", client=client,
    )
    assert result.engine == _ENGINE
    assert result.model_version == "my-vlm"
    assert result.error is None
    assert result.text == "唐律疏議卷第一名例律"
    assert result.char_count == len(result.text)
    assert result.confidence == pytest.approx(0.80)
    assert result.metadata.get("custom") is True


def test_custom_ocr_page_applies_validation_guard():
    """An English image-description response must be zeroed by apply_validation."""
    client = _FakeClient(content="This image shows a page of classical Chinese text.")
    result = custom_ocr_page(
        _blank_image(), page_id="doc::p1", model="my-vlm", client=client,
    )
    assert result.text == ""
    assert result.error is not None
    assert result.error.startswith("validation_failed")
    assert result.confidence == 0.0


def test_custom_ocr_page_api_error_returns_error_result():
    client = _FakeClient(raise_exc=RuntimeError("boom"))
    result = custom_ocr_page(
        _blank_image(), page_id="doc::p2", model="my-vlm", client=client,
    )
    assert result.text == ""
    assert result.char_count == 0
    assert result.confidence == 0.0
    assert result.error is not None
    assert "RuntimeError" in result.error


def test_custom_ocr_page_passes_model_to_client():
    fake = _FakeClient(content="名例律")
    custom_ocr_page(_blank_image(), page_id="p", model="vendor/model-x", client=fake)
    assert fake.chat.completions.last_kwargs is not None
    assert fake.chat.completions.last_kwargs["model"] == "vendor/model-x"
    assert fake.chat.completions.last_kwargs["temperature"] == 0.0
