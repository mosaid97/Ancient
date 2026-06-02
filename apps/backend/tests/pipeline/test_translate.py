"""Tests for pipeline/translate.py — serial and parallel paths.

Covers:
  - _process_primary writes textCanonical + textVernacular on success
  - _process_secondary writes textCanonical (no vernacular) on success
  - Both functions write translationStatus='failed' on exception
  - translate_chunks serial path (workers=1) aggregates results
  - translate_chunks parallel path (workers>1) produces same aggregate
  - translate_chunks skips empty-text chunks
  - silra._retry honours Retry-After header on RateLimitError

All Neo4j and OpenAI calls are mocked; no real API or DB required.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest

from apps.backend.pipeline.translate import (
    TranslatePageResult,
    TranslationReport,
    _process_primary,
    _process_secondary,
    translate_chunks,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_driver(fetch_rows: list[dict] | None = None) -> MagicMock:
    """Return a mocked Neo4j driver.

    The first session().run() returns fetch_rows (for translate_chunks).
    All subsequent session().run().consume() calls succeed silently.
    """
    session_mock = MagicMock()
    run_mock = MagicMock()
    run_mock.data.return_value = fetch_rows or []
    run_mock.consume.return_value = None
    session_mock.run.return_value = run_mock
    session_mock.__enter__ = lambda s: s
    session_mock.__exit__ = MagicMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = session_mock
    return driver


def _make_client(
    canonical: str = "规范文本",
    vernacular: str = "现代译文",
    concepts: str = '["人名", "地名"]',
) -> MagicMock:
    """Return a mocked OpenAI client that always returns success responses."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = vernacular
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    client.chat.completions.create.return_value = resp
    return client


def _primary_row(**kwargs) -> dict:
    base = {
        "chunk_id": "chunk-001",
        "text": "均田制之法，自北魏始。",
        "page_id": "page-001",
        "tier": "primary",
        "language": "zh-classical",
        "era": "Tang",
        "doc_id": "doc-001",
        "editorial_layer_type": "pure-source",
    }
    base.update(kwargs)
    return base


def _secondary_row(**kwargs) -> dict:
    base = {
        "chunk_id": "chunk-002",
        "text": "本文考察唐代均田制度的实施情况。",
        "page_id": "page-002",
        "tier": "secondary",
        "language": "zh-modern",
        "era": None,
        "doc_id": "doc-002",
        "editorial_layer_type": None,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# _process_primary tests
# ---------------------------------------------------------------------------

class TestProcessPrimary:
    def test_success_writes_canonical_and_vernacular(self):
        """Primary chunk writes textCanonical and textVernacular on success."""
        driver = _make_driver()
        client = _make_client()
        row = _primary_row()

        # Patch at the import site in translate.py (not at the source module)
        with (
            patch("apps.backend.pipeline.translate.analyze_words") as mock_word,
            patch("apps.backend.pipeline.translate.translate_paragraph") as mock_para,
            patch("apps.backend.pipeline.translate.review_translation") as mock_review,
        ):
            word_result = MagicMock()
            word_result.text_canonical = "均田制之法，自北魏始。"
            mock_word.return_value = word_result

            para_result = MagicMock()
            para_result.text_vernacular_ja = None
            para_result.prompt_tokens = 5
            para_result.completion_tokens = 10
            mock_para.return_value = para_result

            review_result = MagicMock()
            review_result.text_final = "均田制度的法规，从北魏开始实施。"
            review_result.prompt_tokens = 5
            review_result.completion_tokens = 10
            mock_review.return_value = review_result

            result = _process_primary(row, driver, client, "deepseek-chat")

        assert result.status == "ok"
        assert result.chunk_id == "chunk-001"
        assert result.tier == "primary"

        # Confirm Neo4j write was called with textVernacular parameter
        run_args = driver.session().__enter__().run.call_args_list
        assert any(
            "text_vernacular" in str(c) or "textVernacular" in str(c)
            for c in run_args
        ), "Expected textVernacular in write call"

    def test_empty_text_returns_skipped(self):
        """Primary chunk with no text is skipped without LLM calls."""
        driver = _make_driver()
        client = _make_client()
        row = _primary_row(text="")

        with patch("apps.backend.agents.translation.word.analyze_words") as mock_word:
            result = _process_primary(row, driver, client, "deepseek-chat")
            mock_word.assert_not_called()

        assert result.status == "skipped"

    def test_llm_error_writes_failed(self):
        """LLM exception marks chunk as failed and writes WRITE_FAILED."""
        driver = _make_driver()
        client = _make_client()
        row = _primary_row()

        with patch("apps.backend.pipeline.translate.analyze_words") as mock_word:
            mock_word.side_effect = RuntimeError("API timeout")
            result = _process_primary(row, driver, client, "deepseek-chat")

        assert result.status == "failed"
        assert "API timeout" in (result.error or "")


# ---------------------------------------------------------------------------
# _process_secondary tests
# ---------------------------------------------------------------------------

class TestProcessSecondary:
    def test_success_writes_canonical_only(self):
        """Secondary chunk writes textCanonical but not textVernacular."""
        driver = _make_driver()
        client = _make_client()
        row = _secondary_row()

        with patch("apps.backend.pipeline.translate.normalize_canonical") as mock_norm:
            norm_result = MagicMock()
            norm_result.canonical = "本文考察唐代均田制度的实施情况。"
            mock_norm.return_value = norm_result

            result = _process_secondary(row, driver, client, "deepseek-chat")

        assert result.status == "ok"
        assert result.tier == "secondary"
        # Confirm NO textVernacular in write calls
        run_args = driver.session().__enter__().run.call_args_list
        assert not any("text_vernacular" in str(c) for c in run_args), \
            "Secondary should not write textVernacular"

    def test_empty_text_returns_skipped(self):
        row = _secondary_row(text="")
        driver = _make_driver()
        result = _process_secondary(row, driver, _make_client(), "deepseek-chat")
        assert result.status == "skipped"

    def test_exception_writes_failed(self):
        driver = _make_driver()
        client = _make_client()
        row = _secondary_row()

        with patch("apps.backend.pipeline.translate.normalize_canonical") as mock_norm:
            mock_norm.side_effect = ValueError("normalize error")
            result = _process_secondary(row, driver, client, "deepseek-chat")

        assert result.status == "failed"


# ---------------------------------------------------------------------------
# translate_chunks — serial path
# ---------------------------------------------------------------------------

class TestTranslateChunksSerial:
    def test_empty_batch_returns_zero_counts(self):
        driver = _make_driver(fetch_rows=[])
        report = translate_chunks(driver, client=_make_client(), workers=1)
        assert report.total == 0
        assert report.ok == 0

    def test_mixed_tier_batch_aggregates_correctly(self):
        rows = [_primary_row(), _secondary_row()]
        driver = _make_driver(fetch_rows=rows)

        with (
            patch("apps.backend.pipeline.translate._process_primary") as mp,
            patch("apps.backend.pipeline.translate._process_secondary") as ms,
        ):
            mp.return_value = TranslatePageResult(
                chunk_id="chunk-001", page_id="page-001",
                tier="primary", language="zh-classical", status="ok",
            )
            ms.return_value = TranslatePageResult(
                chunk_id="chunk-002", page_id="page-002",
                tier="secondary", language="zh-modern", status="ok",
            )
            report = translate_chunks(driver, client=_make_client(), workers=1)

        assert report.total == 2
        assert report.ok == 2
        assert report.failed == 0

    def test_tier_filter_is_passed_to_query(self):
        driver = _make_driver(fetch_rows=[])
        translate_chunks(driver, client=_make_client(), tier_filter="primary", workers=1)
        run_call = driver.session().__enter__().run.call_args
        assert run_call.kwargs.get("tier_filter") == "primary"

    def test_failed_chunk_counted(self):
        rows = [_primary_row()]
        driver = _make_driver(fetch_rows=rows)

        with patch("apps.backend.pipeline.translate._process_primary") as mp:
            mp.return_value = TranslatePageResult(
                chunk_id="chunk-001", page_id="page-001",
                tier="primary", language="zh-classical", status="failed",
                error="timeout",
            )
            report = translate_chunks(driver, client=_make_client(), workers=1)

        assert report.failed == 1
        assert report.ok == 0


# ---------------------------------------------------------------------------
# translate_chunks — parallel path
# ---------------------------------------------------------------------------

class TestTranslateChunksParallel:
    def test_parallel_same_aggregate_as_serial(self):
        """Workers>1 produces identical aggregate counts as workers=1."""
        rows = [
            _primary_row(chunk_id=f"chunk-{i}", page_id=f"page-{i}")
            for i in range(6)
        ]
        driver = _make_driver(fetch_rows=rows)
        client = _make_client()

        ok_result = lambda cid: TranslatePageResult(
            chunk_id=cid, page_id=cid, tier="primary",
            language="zh-classical", status="ok",
        )

        with patch("apps.backend.pipeline.translate._process_primary") as mp:
            mp.side_effect = lambda row, *a, **kw: ok_result(row["chunk_id"])
            report_parallel = translate_chunks(driver, client=client, workers=3)

        # Reset and run serially for comparison
        driver2 = _make_driver(fetch_rows=rows)
        with patch("apps.backend.pipeline.translate._process_primary") as mp2:
            mp2.side_effect = lambda row, *a, **kw: ok_result(row["chunk_id"])
            report_serial = translate_chunks(driver2, client=client, workers=1)

        assert report_parallel.ok == report_serial.ok == 6
        assert report_parallel.failed == report_serial.failed == 0

    def test_parallel_worker_exception_is_caught(self):
        """An unhandled worker exception is recorded as failed, not raised."""
        rows = [_primary_row(chunk_id="chunk-err")]
        driver = _make_driver(fetch_rows=rows)

        with patch("apps.backend.pipeline.translate._process_primary") as mp:
            mp.side_effect = RuntimeError("worker crash")
            report = translate_chunks(driver, client=_make_client(), workers=2)

        assert report.failed == 1
        assert report.ok == 0

    def test_parallel_workers_parameter_defaults_to_one(self):
        """Default workers=1 uses serial path (no ThreadPoolExecutor)."""
        rows = [_primary_row()]
        driver = _make_driver(fetch_rows=rows)

        with (
            patch("apps.backend.pipeline.translate._process_primary") as mp,
            patch("apps.backend.pipeline.translate.ThreadPoolExecutor") as mock_pool,
        ):
            mp.return_value = TranslatePageResult(
                chunk_id="chunk-001", page_id="page-001",
                tier="primary", language="zh-classical", status="ok",
            )
            translate_chunks(driver, client=_make_client(), workers=1)
            mock_pool.assert_not_called()


# ---------------------------------------------------------------------------
# silra._retry — Retry-After backoff
# ---------------------------------------------------------------------------

class TestSilraRetryAfter:
    def test_retry_after_header_is_honoured(self):
        """_retry sleeps for the Retry-After value on a 429 response."""
        import openai as _openai
        from apps.backend.llm.silra import _retry

        # Build a fake RateLimitError with Retry-After=5
        fake_response = MagicMock()
        fake_response.headers = {"retry-after": "0.01"}  # tiny value for speed
        exc = _openai.RateLimitError(
            message="rate limited",
            response=fake_response,
            body=None,
        )

        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise exc
            return "ok"

        result = _retry(flaky, max_retries=3, base_delay=0.001)
        assert result == "ok"
        assert call_count == 2

    def test_retry_after_absent_falls_back_to_exponential(self):
        """When Retry-After header is absent, normal exponential backoff is used."""
        import openai as _openai
        from apps.backend.llm.silra import _retry

        fake_response = MagicMock()
        fake_response.headers = {}   # no Retry-After
        exc = _openai.RateLimitError(
            message="rate limited",
            response=fake_response,
            body=None,
        )

        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise exc
            return "done"

        with patch("apps.backend.llm.silra.time.sleep") as mock_sleep:
            result = _retry(flaky, max_retries=3, base_delay=1.0)

        assert result == "done"
        # Should have slept exactly once (first retry)
        mock_sleep.assert_called_once()
        # Delay should be the base 1.0 (2^0 * 1.0)
        assert mock_sleep.call_args[0][0] == pytest.approx(1.0)

    def test_max_retries_exhausted_raises(self):
        """_retry re-raises the last exception after max_retries."""
        import openai as _openai
        from apps.backend.llm.silra import _retry

        fake_response = MagicMock()
        fake_response.headers = {}
        exc = _openai.APIConnectionError(request=MagicMock())

        with (
            patch("apps.backend.llm.silra.time.sleep"),
            pytest.raises(_openai.APIConnectionError),
        ):
            _retry(lambda: (_ for _ in ()).throw(exc), max_retries=2, base_delay=0.001)
