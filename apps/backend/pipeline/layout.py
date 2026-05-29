"""Phase-4 layout analysis orchestrator (plan §6 Stage 4).

Walks every PAGE node that has completed Phase-3 fusion and runs
:class:`apps.backend.ocr.structure.StructureEngine` over its preprocessed
image, writing layout regions, structured Markdown, and a page-type
classification back to Neo4j.

Two categories of pages are handled differently:

**Manuscript / cursive pages** (Dunhuang handwritten facsimiles, etc.):
Both OCR engines returned ≤10 characters combined. PP-StructureV3 is a
typeset-trained model and produces garbage on these pages. They are
pre-classified by :func:`_classify_manuscript` and receive
``layoutStatus='manuscript'`` + ``pageType='manuscript_cursive'``
without any inference.

**Typeset pages** (the vast majority of the corpus):
PP-StructureV3 runs normally. The orchestrator writes ``layoutStatus='ok'``
on success, ``'empty'`` when the pipeline ran but found no regions, and
``'failed'`` when an exception was raised (retryable on the next run).

Per-page PAGE properties (camelCase per AGENTS.md §4):

- ``layoutStatus`` — ``'ok' | 'empty' | 'failed' | 'manuscript'``
- ``layoutAt`` — Neo4j ``timestamp()``
- ``layoutModelVersion`` — engine identifier string
- ``layoutDurationSeconds`` — wall-clock per page
- ``layoutRegionCount`` — count of detected bounding boxes
- ``layoutJson`` — JSON list of ``{label, bbox, score}`` dicts
- ``structuredMarkdown`` — reading-order-recovered Markdown text
- ``tableHtmlJson`` — JSON list of ``{region_id, html}`` (``'[]'`` if none)
- ``pageType`` — ``'typeset' | 'manuscript_cursive' | 'image_only'``
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np
from neo4j import Driver

from apps.backend.ocr.structure import LayoutPageResult, StructureEngine
from apps.backend.pipeline.extract import (
    DOWNLOAD_RETRY_ATTEMPTS,
    DOWNLOAD_RETRY_BASE_SLEEP,
    _is_transient_download_error,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manuscript heuristic thresholds
# ---------------------------------------------------------------------------

MANUSCRIPT_MAX_CHARS: int = 10
"""A page with fused char count at or below this value (and both OCR engines
empty) is pre-classified as ``pageType='manuscript_cursive'``.  Chosen to
be above zero (to catch pages with 1–2 stray digits misdetected by Paddle)
while well below any real typeset page (smallest real pages have ~30 chars)."""


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

_SELECT_LAYOUT_PAGES = """
MATCH (p:PAGE)
WHERE p.mode = 'ocr'
  AND p.preprocessedImageUri IS NOT NULL
  AND (p.paddleOcrStatus = 'ok' OR p.deepseekOcrStatus = 'ok'
       OR p.fusionStatus IN ['ok', 'single'])
  AND ($recompute = true OR p.layoutStatus IS NULL OR p.layoutStatus = 'failed')
  AND ($document_id IS NULL OR p.documentId = $document_id)
RETURN p.id               AS page_id,
       p.documentId       AS document_id,
       p.tier             AS tier,
       p.preprocessedImageUri AS image_uri,
       coalesce(p.fusionCharCount, p.paddleOcrCharCount, p.deepseekOcrCharCount, 0)
                          AS fusion_char_count,
       p.paddleOcrStatus  AS paddle_status,
       p.deepseekOcrStatus AS deepseek_status,
       p.language         AS language,
       p.role             AS role
ORDER BY p.documentId, p.docPageIndex, p.id
"""

_LAYOUT_PAGE_UPDATE = """
MATCH (p:PAGE {id: $page_id})
SET p.layoutStatus          = $status,
    p.layoutAt              = timestamp(),
    p.layoutModelVersion    = $model_version,
    p.layoutDurationSeconds = $duration_seconds,
    p.layoutRegionCount     = $region_count,
    p.layoutJson            = $layout_json,
    p.structuredMarkdown    = $structured_markdown,
    p.tableHtmlJson         = $table_html_json,
    p.pageType              = $page_type,
    p.layoutError           = $error
RETURN p.id AS id
"""


# ---------------------------------------------------------------------------
# Reporting dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LayoutOutcome:
    """Per-page result from the layout orchestrator.

    Status values:
    - ``'ok'``: layout ran and found regions.
    - ``'empty'``: layout ran, no regions detected (blank / image-only pages).
    - ``'failed'``: pipeline raised an exception (retryable).
    - ``'manuscript'``: pre-classified as cursive handwriting; no inference.
    """

    page_id: str
    document_id: str
    status: str
    page_type: str = "typeset"
    region_count: int = 0
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LayoutRunReport:
    """Aggregate report for one :func:`run_layout_pages` invocation."""

    pages_total: int = 0
    pages_ok: int = 0
    pages_empty: int = 0
    pages_failed: int = 0
    pages_manuscript: int = 0
    total_duration_seconds: float = 0.0
    avg_seconds_per_page: float = 0.0
    by_document: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    sample_outcomes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Layout run: total={self.pages_total} "
            f"ok={self.pages_ok} "
            f"empty={self.pages_empty} "
            f"failed={self.pages_failed} "
            f"manuscript={self.pages_manuscript} "
            f"({self.total_duration_seconds/60:.1f} min, "
            f"{self.avg_seconds_per_page:.2f}s/page avg)"
        )


# ---------------------------------------------------------------------------
# Manuscript pre-classification
# ---------------------------------------------------------------------------


def _classify_manuscript(row: dict[str, Any]) -> bool:
    """Return True when this page should be tagged ``manuscript_cursive``.

    Criteria (all must hold):
    - ``paddleOcrStatus`` is ``'empty'`` or ``None``
    - ``deepseekOcrStatus`` is ``'empty'`` or ``None``
    - ``fusionCharCount`` is at most :data:`MANUSCRIPT_MAX_CHARS`

    Rationale: a page where both engines found ≤10 characters is almost
    certainly a handwritten cursive facsimile (or a blank divider page).
    Running PP-StructureV3 on it would produce garbage region detections.
    The ``tag_manuscript_page`` helper allows HITL override for pages that
    this heuristic misclassifies.
    """
    paddle_empty = row.get("paddle_status") in ("empty", None)
    deepseek_empty = row.get("deepseek_status") in ("empty", None)
    char_count = row.get("fusion_char_count") or 0
    return paddle_empty and deepseek_empty and char_count <= MANUSCRIPT_MAX_CHARS


# ---------------------------------------------------------------------------
# MinIO download (re-uses the retry helper from extract.py)
# ---------------------------------------------------------------------------


def _get_object_with_retry(minio_client: Any, bucket: str, key: str) -> bytes:
    last_exc: BaseException | None = None
    for attempt in range(1, DOWNLOAD_RETRY_ATTEMPTS + 1):
        try:
            response = minio_client.get_object(bucket, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == DOWNLOAD_RETRY_ATTEMPTS or not _is_transient_download_error(exc):
                raise
            sleep = DOWNLOAD_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            logger.warning(
                "MinIO transient on %s/%s (attempt %d/%d): %s — retrying in %.1fs",
                bucket, key, attempt, DOWNLOAD_RETRY_ATTEMPTS, exc, sleep,
            )
            time.sleep(sleep)
    raise last_exc  # type: ignore[misc]


def _download_image(minio_client: Any, bucket: str, key: str) -> np.ndarray:
    data = _get_object_with_retry(minio_client, bucket, key)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to decode image at {bucket}/{key} ({len(data)} bytes)")
    return img


_DEFAULT_BUCKET = "ancient-pages"


def _uri_to_bucket_key(uri: str) -> tuple[str, str]:
    """Parse the MinIO URI into (bucket, key).

    Storage convention in this project (from :mod:`apps.backend.pipeline.preprocess`):
    URIs are stored as bare ``<document_id>/page_N/<step>.png`` paths —
    the bucket is always ``ancient-pages`` and is not embedded in the URI.
    Optionally accepts ``minio://<bucket>/<key>`` for forward-compatibility.
    """
    if uri.startswith("minio://"):
        uri = uri.removeprefix("minio://")
        bucket, _, key = uri.partition("/")
        return bucket, key
    # Bare key — bucket is implicit.
    return _DEFAULT_BUCKET, uri


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------


def _classify_layout_status(result: LayoutPageResult) -> str:
    """Return ``'ok' | 'empty' | 'failed'`` for a :class:`LayoutPageResult`."""
    if result.error is not None:
        return "failed"
    if not result.regions:
        return "empty"
    return "ok"


# ---------------------------------------------------------------------------
# Neo4j write helper
# ---------------------------------------------------------------------------


def _write_layout_result(
    driver: Driver,
    page_id: str,
    result: LayoutPageResult,
    *,
    status: str,
    page_type: str,
) -> None:
    """Persist layout analysis results to the PAGE node."""
    with driver.session() as session:
        session.run(
            _LAYOUT_PAGE_UPDATE,
            page_id=page_id,
            status=status,
            model_version=result.model_version,
            duration_seconds=float(result.duration_seconds or 0.0),
            region_count=int(result.region_count),
            layout_json=result.layout_json(),
            structured_markdown=result.markdown or "",
            table_html_json=result.table_html_json(),
            page_type=page_type,
            error=result.error,
        ).consume()


def _write_manuscript_page(
    driver: Driver,
    page_id: str,
    *,
    model_version: str = "PP-StructureV3/PP-DocLayout_plus-L",
) -> None:
    """Write manuscript pre-classification without running inference."""
    with driver.session() as session:
        session.run(
            _LAYOUT_PAGE_UPDATE,
            page_id=page_id,
            status="manuscript",
            model_version=model_version,
            duration_seconds=0.0,
            region_count=0,
            layout_json="[]",
            structured_markdown="",
            table_html_json="[]",
            page_type="manuscript_cursive",
            error=None,
        ).consume()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_layout_pages(
    driver: Driver,
    minio_client: Any,
    *,
    engine: StructureEngine | None = None,
    recompute_existing: bool = False,
    document_id: str | None = None,
    max_pages: int | None = None,
    progress_every: int = 20,
) -> LayoutRunReport:
    """Run PP-StructureV3 layout analysis over all eligible PAGE nodes.

    Args:
        driver: Open Neo4j driver.
        minio_client: MinIO client from :func:`apps.backend.storage.minio_client.get_minio_client`.
        engine: Pre-loaded :class:`StructureEngine`. If ``None``, one is
            created here (which triggers model download on first run).
        recompute_existing: If ``True``, re-process pages that already
            have ``layoutStatus`` set. Default ``False``.
        document_id: Restrict to one document (useful for debugging).
        max_pages: Hard cap on pages processed. ``None`` = no cap.
        progress_every: Log a summary line every N pages.

    Returns:
        :class:`LayoutRunReport` with aggregate statistics.
    """
    if engine is None:
        engine = StructureEngine()

    report = LayoutRunReport()

    with driver.session() as session:
        rows = session.run(
            _SELECT_LAYOUT_PAGES,
            recompute=recompute_existing,
            document_id=document_id,
        ).data()

    if max_pages is not None:
        rows = rows[:max_pages]

    report.pages_total = len(rows)
    logger.info("layout: %d pages eligible", report.pages_total)

    run_start = time.monotonic()

    for i, row in enumerate(rows, 1):
        page_id: str = row["page_id"]
        document_id_row: str = row.get("document_id") or ""
        image_uri: str = row.get("image_uri") or ""

        # ---- Pre-classify manuscript pages (no inference) ----
        if _classify_manuscript(row):
            _write_manuscript_page(driver, page_id, model_version=engine._MODEL_VERSION)
            outcome = LayoutOutcome(
                page_id=page_id,
                document_id=document_id_row,
                status="manuscript",
                page_type="manuscript_cursive",
                duration_seconds=0.0,
            )
            report.pages_manuscript += 1
            _accumulate(report, outcome)
            if i % progress_every == 0 or i == report.pages_total:
                _log_progress(report, i, run_start)
            continue

        # ---- Download preprocessed image ----
        try:
            bucket, key = _uri_to_bucket_key(image_uri)
            image = _download_image(minio_client, bucket, key)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            logger.warning("layout download failed %s: %s", page_id, err)
            failed_result = LayoutPageResult(
                page_id=page_id,
                model_version=engine._MODEL_VERSION,
                error=err,
            )
            _write_layout_result(driver, page_id, failed_result, status="failed", page_type="typeset")
            outcome = LayoutOutcome(
                page_id=page_id,
                document_id=document_id_row,
                status="failed",
                error=err,
            )
            report.pages_failed += 1
            report.errors.append(f"{page_id}: {err}")
            _accumulate(report, outcome)
            if i % progress_every == 0 or i == report.pages_total:
                _log_progress(report, i, run_start)
            continue

        # ---- Run PP-StructureV3 ----
        result = engine.analyse_page(image, page_id=page_id)
        status = _classify_layout_status(result)
        page_type = "image_only" if status == "empty" else "typeset"

        _write_layout_result(driver, page_id, result, status=status, page_type=page_type)

        outcome = LayoutOutcome(
            page_id=page_id,
            document_id=document_id_row,
            status=status,
            page_type=page_type,
            region_count=result.region_count,
            duration_seconds=result.duration_seconds,
            error=result.error,
        )

        if status == "ok":
            report.pages_ok += 1
        elif status == "empty":
            report.pages_empty += 1
        else:
            report.pages_failed += 1
            report.errors.append(f"{page_id}: {result.error}")

        _accumulate(report, outcome)

        if i % progress_every == 0 or i == report.pages_total:
            _log_progress(report, i, run_start)

    elapsed = time.monotonic() - run_start
    report.total_duration_seconds = round(elapsed, 1)
    processed = report.pages_ok + report.pages_empty + report.pages_failed + report.pages_manuscript
    report.avg_seconds_per_page = round(elapsed / max(processed, 1), 3)

    logger.info("layout run complete: %s", report)
    return report


# ---------------------------------------------------------------------------
# HITL override
# ---------------------------------------------------------------------------


def tag_manuscript_page(driver: Driver, page_id: str) -> None:
    """Manually tag a PAGE as ``manuscript_cursive`` (HITL override).

    Use this in a notebook cell when ``_classify_manuscript`` has
    misclassified a page (e.g. a sparse but typeset page was tagged
    correctly by the heuristic, but a dense cursive page slipped through
    because it had unexpected OCR output). The function sets
    ``pageType='manuscript_cursive'`` and ``layoutStatus='manuscript'``
    so the page is excluded from future layout runs (unless
    ``recompute=True`` is passed).

    Args:
        driver: Open Neo4j driver.
        page_id: The ``PAGE.id`` to retag.

    Example::

        from apps.backend.pipeline.layout import tag_manuscript_page
        tag_manuscript_page(driver, "唐耕耦、陆宏基_敦煌社会经济文献真迹释录_第二辑__9c6014dd26::p00459")
    """
    _write_manuscript_page(driver, page_id)
    logger.info("tag_manuscript_page: %s → manuscript_cursive", page_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _accumulate(report: LayoutRunReport, outcome: LayoutOutcome) -> None:
    doc = outcome.document_id or "unknown"
    doc_stats = report.by_document.setdefault(doc, {"ok": 0, "empty": 0, "failed": 0, "manuscript": 0})
    doc_stats[outcome.status] = doc_stats.get(outcome.status, 0) + 1
    if len(report.sample_outcomes) < 20:
        report.sample_outcomes.append(outcome.to_dict())


def _log_progress(report: LayoutRunReport, i: int, run_start: float) -> None:
    elapsed = time.monotonic() - run_start
    processed = report.pages_ok + report.pages_empty + report.pages_failed + report.pages_manuscript
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = report.pages_total - i
    eta_min = remaining / rate / 60 if rate > 0 else float("nan")
    logger.info(
        "layout [%d/%d] ok=%d empty=%d failed=%d manuscript=%d "
        "| %.2f p/s | ETA ~%.0f min",
        i, report.pages_total,
        report.pages_ok, report.pages_empty,
        report.pages_failed, report.pages_manuscript,
        rate, eta_min,
    )
    print(
        f"  [{i}/{report.pages_total}] "
        f"ok={report.pages_ok} empty={report.pages_empty} "
        f"failed={report.pages_failed} manuscript={report.pages_manuscript} "
        f"| {rate:.2f} p/s | ETA ~{eta_min:.0f} min",
        flush=True,
    )
