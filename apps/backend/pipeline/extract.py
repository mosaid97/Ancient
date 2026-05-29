"""Phase-3 dual-OCR orchestrator (plan §6 Stage 3a + 3b + 3c).

Walks every preprocessed PAGE node (``mode='ocr'`` AND
``preprocessedImageUri IS NOT NULL``) and runs one or both OCR engines,
storing per-engine results back on the PAGE.

Two entry points are exposed for the offline + online flight split:

- :func:`run_paddle_pages` — local PaddleOCR PP-OCRv5. Works offline
  once the model is downloaded; safe to ``caffeinate -dimsu`` during a
  flight.
- :func:`run_deepseek_pages` — DeepSeek-OCR via Silra. Requires
  internet; run after landing.

Both runners are idempotent and resumable: ``(<engine>)OcrStatus`` is
checked per page so re-running picks up where the previous run failed
or stopped.

Per-engine PAGE properties (camelCase per AGENTS.md §4):

- ``paddleOcrText`` / ``deepseekOcrText`` — full-page transcription.
- ``paddleOcrLinesJson`` / ``deepseekOcrLinesJson`` — per-line JSON
  blob (text + bbox + score).
- ``paddleOcrConfidence`` / ``deepseekOcrConfidence`` — aggregate
  per-page confidence in ``[0, 1]``.
- ``paddleOcrModelVersion`` / ``deepseekOcrModelVersion`` — engine
  identifier (e.g. ``PP-OCRv5-mobile/ch`` or ``deepseek-ocr``).
- ``paddleOcrDurationSeconds`` / ``deepseekOcrDurationSeconds`` —
  per-page wall-clock (for runtime projections).
- ``paddleOcrStatus`` / ``deepseekOcrStatus`` —
  ``'ok' | 'failed' | 'skipped'``.
- ``paddleOcrError`` / ``deepseekOcrError`` — error string when
  status='failed'.
- ``paddleOcrAt`` / ``deepseekOcrAt`` — Neo4j ``timestamp()``.

The fusion step (Phase 3d) lives in :mod:`apps.backend.pipeline.fusion`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np
from neo4j import Driver

from apps.backend.ocr.base import OCRPageResult, serialize_lines
from apps.backend.ocr.qwen_vl import qwen_vl_ocr_page
from apps.backend.ocr.silra_deepseek import deepseek_ocr_page

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transient-error retry policy for MinIO downloads.
# ---------------------------------------------------------------------------

DOWNLOAD_RETRY_ATTEMPTS: int = 4
"""Maximum attempts per object including the initial call.

Empirically, the dominant transient is ``RequestTimeTooSkewed`` caused by
Docker-for-Mac VM clock drift after a host sleep/wake; that condition
typically self-heals within 10-30 seconds once the VM re-syncs against
the host clock. Four attempts at 1s/2s/4s back-off gives ~7s of total
sleep — long enough for the common case, short enough that an actually-
missing object still fails fast."""

DOWNLOAD_RETRY_BASE_SLEEP: float = 1.0
"""Base sleep seconds between retries (doubled on each attempt)."""

_TRANSIENT_ERROR_MARKERS: tuple[str, ...] = (
    "RequestTimeTooSkewed",
    "InternalError",
    "SlowDown",
    "ServiceUnavailable",
    "Connection",
    "timed out",
    "timeout",
    "Read timed out",
    "EOF occurred",
)


def _is_transient_download_error(exc: BaseException) -> bool:
    """Heuristic: should we retry this MinIO error?

    Matches by error-code substring rather than exception class so we
    cover ``minio.error.S3Error`` (which carries the code in ``.code``
    and ``str(exc)``) and the underlying ``urllib3`` / network errors
    uniformly. ``NoSuchKey`` / ``NoSuchBucket`` / ``AccessDenied`` are
    deliberately *not* in the list — those are permanent failures and
    retrying would just waste wall-clock.
    """

    msg = str(exc)
    code = getattr(exc, "code", "") or ""
    haystack = f"{code} {msg}"
    return any(marker in haystack for marker in _TRANSIENT_ERROR_MARKERS)


# ---------------------------------------------------------------------------
# Cypher (plan §5 v2.1 spine; Phase-3 adds per-engine properties to PAGE).
# ---------------------------------------------------------------------------


def _select_pages_cypher(
    engine: str,
    *,
    include_empty: bool = False,
    require_paddle_text: int = 0,
) -> str:
    """Pages eligible for ``engine`` OCR; column name matches engine name.

    Args:
        engine: One of ``'paddleocr'``, ``'deepseek_ocr'``, ``'qwen_vl_ocr'``.
        include_empty: When ``True``, also re-process pages that currently
            have ``status='empty'`` (e.g. pages cleaned of hallucinations
            that deserve a fresh attempt with the new inline validator).
        require_paddle_text: When > 0, only select pages where
            ``paddleOcrCharCount >= require_paddle_text``.  Use this when
            re-running an LLM engine on its empty pages to skip pages that
            Paddle also couldn't read (genuinely blank folios).
    """
    status_col = {
        "paddleocr": "paddleOcrStatus",
        "deepseek_ocr": "deepseekOcrStatus",
        "qwen_vl_ocr": "qwenVlOcrStatus",
    }[engine]
    empty_clause = f" OR p.{status_col} = 'empty'" if include_empty else ""
    paddle_clause = (
        f"\n      AND p.paddleOcrCharCount >= {require_paddle_text}"
        if require_paddle_text > 0
        else ""
    )
    return f"""
    MATCH (p:PAGE)
    WHERE p.mode = 'ocr'
      AND p.preprocessedImageUri IS NOT NULL
      AND p.role IN $roles
      AND ($recompute = true OR p.{status_col} IS NULL OR p.{status_col} = 'failed'{empty_clause})
      AND ($document_id IS NULL OR p.documentId = $document_id){paddle_clause}
    RETURN p.id AS page_id,
           p.documentId AS document_id,
           p.sectionId AS section_id,
           p.docPageIndex AS page_index,
           p.tier AS tier,
           p.preprocessedImageUri AS image_uri,
           p.language AS language,
           p.role AS role
    ORDER BY p.documentId, p.docPageIndex, p.id
    """


_PADDLE_PAGE_UPDATE = """
MATCH (p:PAGE {id: $page_id})
SET p.paddleOcrText = $text,
    p.paddleOcrLinesJson = $lines_json,
    p.paddleOcrConfidence = $confidence,
    p.paddleOcrModelVersion = $model_version,
    p.paddleOcrDurationSeconds = $duration_seconds,
    p.paddleOcrCharCount = $char_count,
    p.paddleOcrStatus = $status,
    p.paddleOcrError = $error,
    p.paddleOcrAt = timestamp()
RETURN p.id AS id
"""


_DEEPSEEK_PAGE_UPDATE = """
MATCH (p:PAGE {id: $page_id})
SET p.deepseekOcrText = $text,
    p.deepseekOcrLinesJson = $lines_json,
    p.deepseekOcrConfidence = $confidence,
    p.deepseekOcrModelVersion = $model_version,
    p.deepseekOcrDurationSeconds = $duration_seconds,
    p.deepseekOcrCharCount = $char_count,
    p.deepseekOcrStatus = $status,
    p.deepseekOcrError = $error,
    p.deepseekOcrAt = timestamp()
RETURN p.id AS id
"""

_QWEN_VL_PAGE_UPDATE = """
MATCH (p:PAGE {id: $page_id})
SET p.qwenVlOcrText = $text,
    p.qwenVlOcrLinesJson = $lines_json,
    p.qwenVlOcrConfidence = $confidence,
    p.qwenVlOcrModelVersion = $model_version,
    p.qwenVlOcrDurationSeconds = $duration_seconds,
    p.qwenVlOcrCharCount = $char_count,
    p.qwenVlOcrStatus = $status,
    p.qwenVlOcrError = $error,
    p.qwenVlOcrAt = timestamp()
RETURN p.id AS id
"""


# ---------------------------------------------------------------------------
# Reporting dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class ExtractOutcome:
    """Per-page result returned by an engine runner.

    Status values: ``"ok"`` (text extracted), ``"empty"`` (engine ran,
    no text on page — e.g. blank covers, photo plates), ``"failed"``
    (engine or download raised), ``"skipped"`` (excluded by filter).
    """

    page_id: str
    document_id: str
    engine: str
    status: str
    char_count: int = 0
    confidence: float = 0.0
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractRunReport:
    """Aggregate report for one :func:`run_paddle_pages` / :func:`run_deepseek_pages` invocation."""

    engine: str = "paddleocr"
    pages_total: int = 0
    pages_processed: int = 0
    pages_empty: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0
    total_chars: int = 0
    duration_seconds: float = 0.0
    avg_seconds_per_page: float = 0.0
    by_document: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    sample_outcomes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# MinIO helpers (mirror apps/backend/pipeline/preprocess.py).
# ---------------------------------------------------------------------------


def _get_object_with_retry(minio_client, bucket: str, key: str) -> bytes:
    """Fetch one object with bounded retry on transient MinIO errors.

    Re-raises the last exception when retries are exhausted, so callers
    can record the failure normally. Permanent errors (NoSuchKey, etc.)
    fail fast on the first attempt.
    """

    last_exc: BaseException | None = None
    for attempt in range(1, DOWNLOAD_RETRY_ATTEMPTS + 1):
        try:
            response = minio_client.get_object(bucket, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception as exc:  # noqa: BLE001 — re-raise after retry
            last_exc = exc
            if attempt == DOWNLOAD_RETRY_ATTEMPTS or not _is_transient_download_error(exc):
                raise
            sleep = DOWNLOAD_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            logger.warning(
                "MinIO transient on %s/%s (attempt %d/%d): %s — retrying in %.1fs",
                bucket, key, attempt, DOWNLOAD_RETRY_ATTEMPTS, exc, sleep,
            )
            time.sleep(sleep)
    # Unreachable, but keeps type-checkers happy.
    raise last_exc  # type: ignore[misc]


def _download_image(minio_client, bucket: str, key: str) -> np.ndarray:
    data = _get_object_with_retry(minio_client, bucket, key)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to decode image at {bucket}/{key} ({len(data)} bytes)")
    return img


def _download_bytes(minio_client, bucket: str, key: str) -> bytes:
    return _get_object_with_retry(minio_client, bucket, key)


# ---------------------------------------------------------------------------
# Status classification.
#
# A page where the OCR engine ran cleanly but found no text regions (e.g.
# blank dividers, photo plates, cover art) is **not** a failure — it is
# the correct outcome for that page. We separate this from ``'failed'``
# so the corpus audit cleanly distinguishes "engine had a problem" from
# "the page legitimately has no text".
# ---------------------------------------------------------------------------


def _classify_status(result: OCRPageResult) -> str:
    """Return ``'ok' | 'empty' | 'failed'`` for an :class:`OCRPageResult``.

    ``validation_failed:*`` errors from the inline hallucination guard are
    treated as ``'empty'`` (not ``'failed'``) so they don't skew the
    engine-failure metrics. The ``*OcrError`` property still records the
    reason, making them auditable.
    """
    if result.error is not None:
        if result.error.startswith("validation_failed:"):
            return "empty"
        return "failed"
    if not (result.text and result.text.strip()):
        return "empty"
    return "ok"


# ---------------------------------------------------------------------------
# Write helpers.
# ---------------------------------------------------------------------------


def _write_paddle_result(driver: Driver, page_id: str, result: OCRPageResult) -> str:
    """Persist a Paddle OCR result; returns the classified status."""

    status = _classify_status(result)
    with driver.session() as session:
        session.run(
            _PADDLE_PAGE_UPDATE,
            page_id=page_id,
            text=result.text or "",
            lines_json=serialize_lines(result.lines),
            confidence=float(result.confidence or 0.0),
            model_version=result.model_version,
            duration_seconds=float(result.duration_seconds or 0.0),
            char_count=int(result.char_count or 0),
            status=status,
            error=result.error,
        ).consume()
    return status


def _write_deepseek_result(driver: Driver, page_id: str, result: OCRPageResult) -> str:
    """Persist a DeepSeek OCR result; returns the classified status."""

    status = _classify_status(result)
    with driver.session() as session:
        session.run(
            _DEEPSEEK_PAGE_UPDATE,
            page_id=page_id,
            text=result.text or "",
            lines_json=serialize_lines(result.lines),
            confidence=float(result.confidence or 0.0),
            model_version=result.model_version,
            duration_seconds=float(result.duration_seconds or 0.0),
            char_count=int(result.char_count or 0),
            status=status,
            error=result.error,
        ).consume()
    return status


def _write_qwen_result(driver: Driver, page_id: str, result: OCRPageResult) -> str:
    """Persist a Qwen-VL-OCR result; returns the classified status."""

    status = _classify_status(result)
    with driver.session() as session:
        session.run(
            _QWEN_VL_PAGE_UPDATE,
            page_id=page_id,
            text=result.text or "",
            lines_json=serialize_lines(result.lines),
            confidence=float(result.confidence or 0.0),
            model_version=result.model_version,
            duration_seconds=float(result.duration_seconds or 0.0),
            char_count=int(result.char_count or 0),
            status=status,
            error=result.error,
        ).consume()
    return status


# ---------------------------------------------------------------------------
# Script-hint routing.
# ---------------------------------------------------------------------------


def _resolve_script_hint(tier: str | None, language: str | None) -> str | None:
    """Derive the ``script_hint`` for an LLM OCR call from page tier + language.

    Routing logic (corpus-tuned; see AGENTS.md 2026-05-23 rule):

    * ``tier='primary'`` + any classical/unknown language →
      ``'traditional'``: primary-tier facsimiles are almost always
      traditional-script woodblock prints or Republican-era 點校 editions.
    * ``tier='secondary'`` + ``language='zh-modern'`` →
      ``'simplified'``: modern scholarly commentary is written in simplified.
    * ``language='zh-modern'`` regardless of tier →
      ``'simplified'``.
    * ``language in {'ja', 'kanbun'}`` → ``None`` (kanbun prompt selected by
      ``language_hint`` path inside each OCR function).
    * Everything else → ``'classical'`` (generic fallback that asserts "output
      the script you see in the image").

    Args:
        tier: ``PAGE.tier`` from Neo4j (``'primary'``, ``'secondary'``, or
            ``None``).
        language: ``PAGE.language`` from Neo4j.

    Returns:
        One of ``'traditional'``, ``'simplified'``, ``'classical'``, or
        ``None`` (kanbun path).
    """
    if language in {"ja", "kanbun", "japan", "jpn"}:
        return None  # kanbun prompt selected via language_hint
    if language == "zh-modern":
        return "simplified"
    if tier == "primary":
        return "traditional"
    if tier == "secondary":
        return "simplified" if language == "zh-modern" else "classical"
    return "classical"


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def run_paddle_pages(
    *,
    driver: Driver,
    minio_client,
    engine,  # PaddleOCREngine (avoid hard import to keep this module light)
    bucket: str = "ancient-pages",
    document_id: str | None = None,
    max_pages: int | None = None,
    recompute_existing: bool = False,
    roles: tuple[str, ...] = ("body",),
    sample_size: int = 5,
    progress_every: int = 25,
) -> ExtractRunReport:
    """Run PaddleOCR over every eligible PAGE.

    Args:
        driver: Open Neo4j driver.
        minio_client: Open MinIO client.
        engine: A pre-built :class:`apps.backend.ocr.paddle.PaddleOCREngine`.
            Caller is expected to keep the engine alive across calls (it
            holds the loaded PP-OCRv5 model in memory).
        bucket: MinIO bucket (default ``ancient-pages``).
        document_id: Optional filter.
        max_pages: Optional cap.
        recompute_existing: Re-OCR pages that already have a result.
        roles: Which PAGE roles to OCR. Default ``('body',)``;
            ``('body', 'marginalia')`` runs the marginal annotations too
            (they are stored as derived sibling PAGEs by Phase 2).
        sample_size: Outcomes retained in ``sample_outcomes`` for the
            notebook.
        progress_every: Log every N pages.

    Returns:
        :class:`ExtractRunReport` with ``engine='paddleocr'``.
    """
    report = ExtractRunReport(engine="paddleocr")
    started = time.monotonic()

    with driver.session() as session:
        rows = list(
            session.run(
                _select_pages_cypher("paddleocr"),
                recompute=recompute_existing,
                document_id=document_id,
                roles=list(roles),
            )
        )
    if max_pages is not None:
        rows = rows[:max_pages]
    report.pages_total = len(rows)

    sample: list[ExtractOutcome] = []
    for idx, row in enumerate(rows, start=1):
        page_id = row["page_id"]
        doc_id = row["document_id"]
        image_uri = row["image_uri"]
        language_hint = row.get("language")
        doc_bucket = report.by_document.setdefault(
            doc_id, {"ok": 0, "empty": 0, "failed": 0, "skipped": 0}
        )
        try:
            image = _download_image(minio_client, bucket, image_uri)
        except Exception as exc:  # noqa: BLE001
            outcome = ExtractOutcome(
                page_id=page_id, document_id=doc_id, engine="paddleocr",
                status="failed", error=f"download failed: {exc}",
            )
            report.pages_failed += 1
            doc_bucket["failed"] += 1
            report.errors.append(f"{page_id}: {outcome.error}")
            if len(sample) < sample_size:
                sample.append(outcome)
            continue

        result = engine.ocr_page(
            image, page_id=page_id, language_hint=language_hint
        )
        try:
            status = _write_paddle_result(driver, page_id, result)
        except Exception as exc:  # noqa: BLE001
            outcome = ExtractOutcome(
                page_id=page_id, document_id=doc_id, engine="paddleocr",
                status="failed", error=f"neo4j write failed: {exc}",
                duration_seconds=result.duration_seconds,
            )
            report.pages_failed += 1
            doc_bucket["failed"] += 1
            report.errors.append(f"{page_id}: {outcome.error}")
            if len(sample) < sample_size:
                sample.append(outcome)
            continue

        outcome = ExtractOutcome(
            page_id=page_id, document_id=doc_id, engine="paddleocr",
            status=status,
            char_count=result.char_count,
            confidence=result.confidence,
            duration_seconds=result.duration_seconds,
            error=result.error,
        )
        if status == "ok":
            report.pages_processed += 1
            report.total_chars += outcome.char_count
            doc_bucket["ok"] += 1
        elif status == "empty":
            report.pages_empty += 1
            doc_bucket["empty"] += 1
        else:  # "failed"
            report.pages_failed += 1
            doc_bucket["failed"] += 1
            if outcome.error:
                report.errors.append(f"{page_id}: {outcome.error}")

        if len(sample) < sample_size:
            sample.append(outcome)

        if idx % progress_every == 0:
            so_far = time.monotonic() - started
            eta = so_far / idx * (len(rows) - idx)
            logger.info(
                "paddle progress %d/%d (%.1fs elapsed, ~%.1fs ETA, "
                "%.2fs/page avg, ok=%d empty=%d failed=%d)",
                idx, len(rows), so_far, eta, so_far / max(idx, 1),
                report.pages_processed, report.pages_empty, report.pages_failed,
            )

    elapsed = time.monotonic() - started
    report.duration_seconds = round(elapsed, 3)
    report.avg_seconds_per_page = (
        round(elapsed / max(report.pages_processed, 1), 3)
        if report.pages_processed else 0.0
    )
    report.sample_outcomes = [o.to_dict() for o in sample]
    return report


def run_deepseek_pages(
    *,
    driver: Driver,
    minio_client,
    silra_client=None,
    bucket: str = "ancient-pages",
    document_id: str | None = None,
    max_pages: int | None = None,
    recompute_existing: bool = False,
    include_empty: bool = False,
    require_paddle_text: int = 0,
    roles: tuple[str, ...] = ("body",),
    sample_size: int = 5,
    progress_every: int = 25,
    max_tokens: int = 4096,
    timeout: float = 180.0,
) -> ExtractRunReport:
    """Run DeepSeek-OCR (Silra) over every eligible PAGE.

    Args:
        driver: Open Neo4j driver.
        minio_client: Open MinIO client.
        silra_client: Optional pre-built Silra client.
        bucket: MinIO bucket.
        document_id: Optional filter.
        max_pages: Optional cap.
        recompute_existing: Re-OCR pages with an existing deepseek result.
        include_empty: Also re-process pages currently at ``status='empty'``
            (use after cleaning hallucinations to give them a fresh attempt).
        require_paddle_text: Minimum ``paddleOcrCharCount`` a page must have to
            be selected.  Setting this to e.g. 11 focuses re-runs on pages
            where Paddle confirmed there is real content, skipping genuinely
            blank folios and saving API calls.
        roles: PAGE roles to OCR.
        sample_size: Outcomes retained in ``sample_outcomes``.
        progress_every: Log every N pages.
        max_tokens: Per-page response cap.
        timeout: Per-request timeout in seconds.

    Returns:
        :class:`ExtractRunReport` with ``engine='deepseek_ocr'``.
    """
    report = ExtractRunReport(engine="deepseek_ocr")
    started = time.monotonic()

    with driver.session() as session:
        rows = list(
            session.run(
                _select_pages_cypher(
                    "deepseek_ocr",
                    include_empty=include_empty,
                    require_paddle_text=require_paddle_text,
                ),
                recompute=recompute_existing,
                document_id=document_id,
                roles=list(roles),
            )
        )
    if max_pages is not None:
        rows = rows[:max_pages]
    report.pages_total = len(rows)

    sample: list[ExtractOutcome] = []
    for idx, row in enumerate(rows, start=1):
        page_id = row["page_id"]
        doc_id = row["document_id"]
        image_uri = row["image_uri"]
        language_hint = row.get("language") or "zh-classical"
        doc_bucket = report.by_document.setdefault(
            doc_id, {"ok": 0, "empty": 0, "failed": 0, "skipped": 0}
        )
        try:
            payload = _download_bytes(minio_client, bucket, image_uri)
        except Exception as exc:  # noqa: BLE001
            outcome = ExtractOutcome(
                page_id=page_id, document_id=doc_id, engine="deepseek_ocr",
                status="failed", error=f"download failed: {exc}",
            )
            report.pages_failed += 1
            doc_bucket["failed"] += 1
            report.errors.append(f"{page_id}: {outcome.error}")
            if len(sample) < sample_size:
                sample.append(outcome)
            continue

        script_hint = _resolve_script_hint(row.get("tier"), language_hint)
        result = deepseek_ocr_page(
            payload,
            page_id=page_id,
            language_hint=language_hint,
            script_hint=script_hint,
            client=silra_client,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        try:
            status = _write_deepseek_result(driver, page_id, result)
        except Exception as exc:  # noqa: BLE001
            outcome = ExtractOutcome(
                page_id=page_id, document_id=doc_id, engine="deepseek_ocr",
                status="failed", error=f"neo4j write failed: {exc}",
                duration_seconds=result.duration_seconds,
            )
            report.pages_failed += 1
            doc_bucket["failed"] += 1
            report.errors.append(f"{page_id}: {outcome.error}")
            if len(sample) < sample_size:
                sample.append(outcome)
            continue

        outcome = ExtractOutcome(
            page_id=page_id, document_id=doc_id, engine="deepseek_ocr",
            status=status,
            char_count=result.char_count,
            confidence=result.confidence,
            duration_seconds=result.duration_seconds,
            error=result.error,
        )
        if status == "ok":
            report.pages_processed += 1
            report.total_chars += outcome.char_count
            doc_bucket["ok"] += 1
        elif status == "empty":
            report.pages_empty += 1
            doc_bucket["empty"] += 1
        else:
            report.pages_failed += 1
            doc_bucket["failed"] += 1
            if outcome.error:
                report.errors.append(f"{page_id}: {outcome.error}")

        if len(sample) < sample_size:
            sample.append(outcome)

        if idx % progress_every == 0:
            so_far = time.monotonic() - started
            eta = so_far / idx * (len(rows) - idx)
            logger.info(
                "deepseek progress %d/%d (%.1fs elapsed, ~%.1fs ETA, "
                "%.2fs/page avg)",
                idx, len(rows), so_far, eta, so_far / max(idx, 1),
            )

    elapsed = time.monotonic() - started
    report.duration_seconds = round(elapsed, 3)
    report.avg_seconds_per_page = (
        round(elapsed / max(report.pages_processed, 1), 3)
        if report.pages_processed else 0.0
    )
    report.sample_outcomes = [o.to_dict() for o in sample]
    return report


def run_qwen_pages(
    *,
    driver: Driver,
    minio_client,
    silra_client=None,
    bucket: str = "ancient-pages",
    document_id: str | None = None,
    max_pages: int | None = None,
    recompute_existing: bool = False,
    include_empty: bool = False,
    roles: tuple[str, ...] = ("body",),
    sample_size: int = 5,
    progress_every: int = 25,
    max_tokens: int = 4096,
    timeout: float = 180.0,
) -> ExtractRunReport:
    """Run Qwen-VL-OCR (Silra) over every eligible PAGE.

    Args:
        driver: Open Neo4j driver.
        minio_client: Open MinIO client.
        silra_client: Optional pre-built Silra client.
        bucket: MinIO bucket.
        document_id: Optional filter.
        max_pages: Optional cap.
        recompute_existing: Re-OCR pages with an existing Qwen result.
        include_empty: Also re-process pages currently at ``status='empty'``.
        roles: PAGE roles to OCR.
        sample_size: Outcomes retained in ``sample_outcomes``.
        progress_every: Log every N pages.
        max_tokens: Per-page response cap.
        timeout: Per-request timeout in seconds.

    Returns:
        :class:`ExtractRunReport` with ``engine='qwen_vl_ocr'``.
    """
    report = ExtractRunReport(engine="qwen_vl_ocr")
    started = time.monotonic()

    with driver.session() as session:
        rows = list(
            session.run(
                _select_pages_cypher("qwen_vl_ocr", include_empty=include_empty),
                recompute=recompute_existing,
                document_id=document_id,
                roles=list(roles),
            )
        )
    if max_pages is not None:
        rows = rows[:max_pages]
    report.pages_total = len(rows)

    sample: list[ExtractOutcome] = []
    for idx, row in enumerate(rows, start=1):
        page_id = row["page_id"]
        doc_id = row["document_id"]
        image_uri = row["image_uri"]
        language_hint = row.get("language") or "zh-classical"
        doc_bucket = report.by_document.setdefault(
            doc_id, {"ok": 0, "empty": 0, "failed": 0, "skipped": 0}
        )
        try:
            payload = _download_bytes(minio_client, bucket, image_uri)
        except Exception as exc:  # noqa: BLE001
            outcome = ExtractOutcome(
                page_id=page_id, document_id=doc_id, engine="qwen_vl_ocr",
                status="failed", error=f"download failed: {exc}",
            )
            report.pages_failed += 1
            doc_bucket["failed"] += 1
            report.errors.append(f"{page_id}: {outcome.error}")
            if len(sample) < sample_size:
                sample.append(outcome)
            continue

        script_hint = _resolve_script_hint(row.get("tier"), language_hint)
        result = qwen_vl_ocr_page(
            payload,
            page_id=page_id,
            language_hint=language_hint,
            script_hint=script_hint,
            client=silra_client,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        try:
            status = _write_qwen_result(driver, page_id, result)
        except Exception as exc:  # noqa: BLE001
            outcome = ExtractOutcome(
                page_id=page_id, document_id=doc_id, engine="qwen_vl_ocr",
                status="failed", error=f"neo4j write failed: {exc}",
                duration_seconds=result.duration_seconds,
            )
            report.pages_failed += 1
            doc_bucket["failed"] += 1
            report.errors.append(f"{page_id}: {outcome.error}")
            if len(sample) < sample_size:
                sample.append(outcome)
            continue

        outcome = ExtractOutcome(
            page_id=page_id, document_id=doc_id, engine="qwen_vl_ocr",
            status=status,
            char_count=result.char_count,
            confidence=result.confidence,
            duration_seconds=result.duration_seconds,
            error=result.error,
        )
        if status == "ok":
            report.pages_processed += 1
            report.total_chars += outcome.char_count
            doc_bucket["ok"] += 1
        elif status == "empty":
            report.pages_empty += 1
            doc_bucket["empty"] += 1
        else:
            report.pages_failed += 1
            doc_bucket["failed"] += 1
            if outcome.error:
                report.errors.append(f"{page_id}: {outcome.error}")

        if len(sample) < sample_size:
            sample.append(outcome)

        if idx % progress_every == 0:
            so_far = time.monotonic() - started
            eta = so_far / idx * (len(rows) - idx)
            logger.info(
                "qwen_vl_ocr progress %d/%d (%.1fs elapsed, ~%.1fs ETA, "
                "%.2fs/page avg)",
                idx, len(rows), so_far, eta, so_far / max(idx, 1),
            )

    elapsed = time.monotonic() - started
    report.duration_seconds = round(elapsed, 3)
    report.avg_seconds_per_page = (
        round(elapsed / max(report.pages_processed, 1), 3)
        if report.pages_processed else 0.0
    )
    report.sample_outcomes = [o.to_dict() for o in sample]
    return report


# ---------------------------------------------------------------------------
# Aggregate summary for notebook checks.
# ---------------------------------------------------------------------------


def extraction_summary(driver: Driver) -> dict[str, Any]:
    """Per-engine coverage summary across the corpus (3-engine view).

    Returns::

        {
            "by_engine_status": {
                "paddleocr": {"ok": N, "failed": N, "(unset)": N},
                "deepseek_ocr": {...},
                "qwen_vl_ocr": {...},
            },
            "by_tier": {"primary": {...}, "secondary": {...}},
            "total_ocr_pages": int,
        }
    """
    cypher = """
    MATCH (p:PAGE)
    WHERE p.mode = 'ocr' AND p.preprocessedImageUri IS NOT NULL
      AND (p.role IS NULL OR p.role = 'body')
    WITH p,
         coalesce(p.tier, '(unset)') AS tier,
         coalesce(p.paddleOcrStatus, '(unset)') AS paddle_status,
         coalesce(p.deepseekOcrStatus, '(unset)') AS deepseek_status,
         coalesce(p.qwenVlOcrStatus, '(unset)') AS qwen_status
    RETURN tier, paddle_status, deepseek_status, qwen_status, count(*) AS n
    """
    by_engine_status: dict[str, dict[str, int]] = {
        "paddleocr": {},
        "deepseek_ocr": {},
        "qwen_vl_ocr": {},
    }
    by_tier: dict[str, dict[str, dict[str, int]]] = {}
    total = 0
    with driver.session() as session:
        for row in session.run(cypher):
            tier = row["tier"]
            ps = row["paddle_status"]
            ds = row["deepseek_status"]
            qs = row["qwen_status"]
            n = row["n"]
            total += n
            by_engine_status["paddleocr"][ps] = by_engine_status["paddleocr"].get(ps, 0) + n
            by_engine_status["deepseek_ocr"][ds] = by_engine_status["deepseek_ocr"].get(ds, 0) + n
            by_engine_status["qwen_vl_ocr"][qs] = by_engine_status["qwen_vl_ocr"].get(qs, 0) + n
            tier_bucket = by_tier.setdefault(
                tier, {"paddleocr": {}, "deepseek_ocr": {}, "qwen_vl_ocr": {}}
            )
            tier_bucket["paddleocr"][ps] = tier_bucket["paddleocr"].get(ps, 0) + n
            tier_bucket["deepseek_ocr"][ds] = tier_bucket["deepseek_ocr"].get(ds, 0) + n
            tier_bucket["qwen_vl_ocr"][qs] = tier_bucket["qwen_vl_ocr"].get(qs, 0) + n

    return {
        "by_engine_status": by_engine_status,
        "by_tier": by_tier,
        "total_ocr_pages": total,
    }
