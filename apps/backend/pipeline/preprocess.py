"""Phase-2 orchestrator: chain the six preprocessing steps on OCR pages.

Inputs
------
- A Neo4j-resident ``(:PAGE {mode='ocr'})`` whose ``imageUri`` points at
  the raw rasterized scan in MinIO (uploaded by Phase 1's
  :mod:`apps.backend.pipeline.ingest`).

Pipeline (plan §6 Stage 2, fixed order)
---------------------------------------
1. **Download** raw image from MinIO (``ancient-pages/<imageUri>``).
2. **Deskew** (Hough lines).
3. **Dewarp** (perspective rectification).
4. **Illumination** (flat-field).
5. **Bleed** (channel-separation removal).
6. **Split** (vertical-projection valley double-page split).
7. **Marginalia** (天頭/地腳 separation).
8. **Enhance** (smart-gated CLAHE + unsharp for low-contrast facsimiles).
9. **Upload** the final body image + every intermediate variant under
   ``ancient-pages/<document_id>/page_<n>/<variant>.png``.
9. **Write back** to Neo4j:
    - The parent ``(:PAGE)`` gets ``preprocessedImageUri``,
      ``preprocessedAt``, ``preprocessingProvenance`` (JSON), and
      ``preprocessingStatus``.
    - If the page-split fired, the *right* folio becomes a sibling
      ``(:PAGE {role:'body'})`` linked via ``(:PAGE)-[:SPLIT_FROM]->``.
    - If marginalia were detected, each band becomes a sibling
      ``(:PAGE {role:'marginalia'})`` linked via
      ``(:PAGE)-[:MARGINALIA_OF]->``.

The function is **idempotent**: re-running it on the same page only
re-uploads + re-writes when ``recompute_existing=True`` (the default is
``False`` so daily incremental runs don't churn the bucket).
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np
from neo4j import Driver

from apps.backend.preprocess.base import (
    PreprocessProvenance,
    StepResult,
    ensure_bgr,
)
from apps.backend.preprocess.bleed import remove_bleed_through
from apps.backend.preprocess.deskew import deskew_image
from apps.backend.preprocess.dewarp import dewarp_image
from apps.backend.preprocess.enhance import enhance_contrast
from apps.backend.preprocess.illumination import correct_illumination
from apps.backend.preprocess.marginalia import separate_marginalia
from apps.backend.preprocess.page_split import split_double_page

logger = logging.getLogger(__name__)


PNG_ENCODE_PARAMS = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
"""Mild PNG compression — fast on M-series CPU; size is dwarfed by the
raw scan anyway."""


# ---------------------------------------------------------------------------
# Cypher (plan §5 v2.1 spine; PAGE upsert in ingest.py is the parent).
# ---------------------------------------------------------------------------


_SELECT_OCR_PAGES = """
MATCH (p:PAGE)
WHERE p.mode = 'ocr'
  AND p.imageUri IS NOT NULL
  AND (p.role IS NULL OR p.role = 'body')
  AND ($recompute = true OR p.preprocessedImageUri IS NULL)
  AND ($document_id IS NULL OR p.documentId = $document_id)
RETURN p.id AS page_id,
       p.documentId AS document_id,
       p.chapterId AS chapter_id,
       p.sectionId AS section_id,
       p.docPageIndex AS page_index,
       p.tier AS tier,
       p.imageUri AS image_uri,
       p.language AS language
ORDER BY p.documentId, p.docPageIndex
"""


_PAGE_UPDATE = """
MATCH (p:PAGE {id: $page_id})
SET p.preprocessedImageUri = $final_uri,
    p.preprocessingProvenance = $provenance_json,
    p.preprocessingStatus = $status,
    p.preprocessedAt = timestamp()
RETURN p.id AS id
"""


_SIBLING_PAGE_UPSERT = """
MATCH (parent:PAGE {id: $parent_id})
MATCH (s:SECTION {id: $section_id})
MERGE (child:PAGE {id: $child_id})
  ON CREATE SET
    child.documentId = $document_id,
    child.chapterId = $chapter_id,
    child.sectionId = $section_id,
    child.docPageIndex = $page_index,
    child.mode = 'ocr',
    child.tier = $tier,
    child.role = $role,
    child.parentPageId = $parent_id,
    child.imageUri = $image_uri,
    child.preprocessedImageUri = $image_uri,
    child.preprocessingProvenance = $provenance_json,
    child.preprocessingStatus = 'derived',
    child.preprocessedAt = timestamp(),
    child.createdAt = timestamp()
  ON MATCH SET
    child.imageUri = $image_uri,
    child.preprocessedImageUri = $image_uri,
    child.preprocessingProvenance = $provenance_json,
    child.preprocessingStatus = 'derived',
    child.preprocessedAt = timestamp()
MERGE (s)-[:INCLUDE]->(child)
MERGE (parent)-[r:DERIVED_PAGE {role: $role}]->(child)
  ON CREATE SET r.createdAt = timestamp()
RETURN child.id AS id
"""


# ---------------------------------------------------------------------------
# Dataclasses for reporting.
# ---------------------------------------------------------------------------


@dataclass
class PagePreprocessOutcome:
    """Per-page result returned by :func:`preprocess_page`."""

    page_id: str
    document_id: str
    status: str  # "ok" | "skipped" | "failed"
    final_uri: str | None = None
    provenance: PreprocessProvenance | None = None
    derived_pages: list[str] = field(default_factory=list)
    error: str | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        return d


@dataclass
class PreprocessRunReport:
    """Aggregate report for one :func:`preprocess_pages` invocation."""

    pages_total: int = 0
    pages_processed: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    pages_split: int = 0
    pages_with_marginalia: int = 0
    derived_pages_written: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    by_document: dict[str, dict[str, int]] = field(default_factory=dict)
    sample_outcomes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# MinIO IO helpers.
# ---------------------------------------------------------------------------


def _download_image(minio_client, bucket: str, key: str) -> np.ndarray:
    """Fetch an image from MinIO and decode it into a BGR uint8 array."""

    response = minio_client.get_object(bucket, key)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"failed to decode image at {bucket}/{key} ({len(data)} bytes)")
    return img


def _upload_png(minio_client, bucket: str, key: str, image: np.ndarray) -> int:
    """PNG-encode ``image`` and PUT it to MinIO. Returns byte size."""

    ok, buf = cv2.imencode(".png", ensure_bgr(image), PNG_ENCODE_PARAMS)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {key}")
    payload = buf.tobytes()
    minio_client.put_object(
        bucket_name=bucket,
        object_name=key,
        data=io.BytesIO(payload),
        length=len(payload),
        content_type="image/png",
    )
    return len(payload)


# ---------------------------------------------------------------------------
# Step chain (pure functions over numpy arrays).
# ---------------------------------------------------------------------------


def run_preprocess_chain(
    image: np.ndarray,
    *,
    skip: set[str] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> list[StepResult]:
    """Run the canonical six-step pipeline on a single image.

    Args:
        image: Raw BGR uint8 page image (already downloaded from MinIO).
        skip: Optional set of step names to skip (``"deskew"``, ``"dewarp"``,
            ``"illumination"``, ``"bleed"``, ``"split"``, ``"marginalia"``).
        overrides: Optional ``{step_name: kwargs}`` mapping passed through
            to the step function (e.g. ``{"deskew": {"min_angle": 0.5}}``).

    Returns:
        The list of :class:`StepResult` in execution order. The last
        element's ``image`` is the final body image (post-marginalia
        crop).
    """
    skip = skip or set()
    overrides = overrides or {}
    current = ensure_bgr(image)
    results: list[StepResult] = []

    def _run(name: str, fn, extra: dict[str, Any] | None = None) -> None:
        nonlocal current
        if name in skip:
            results.append(
                StepResult(
                    image=current,
                    step=name,  # type: ignore[arg-type]
                    params={"skipped": True},
                    metrics={"skipped": True},
                )
            )
            return
        kwargs = overrides.get(name, {}).copy()
        if extra:
            kwargs.update(extra)
        result = fn(current, **kwargs)
        current = result.image
        results.append(result)

    _run("deskew", deskew_image)
    _run("dewarp", dewarp_image)
    _run("illumination", correct_illumination)
    _run("bleed", remove_bleed_through)
    _run("split", split_double_page)
    _run("marginalia", separate_marginalia)
    _run("enhance", enhance_contrast)
    return results


# ---------------------------------------------------------------------------
# Public entry point: one page.
# ---------------------------------------------------------------------------


def preprocess_page(
    *,
    minio_client,
    driver: Driver,
    page_row: dict[str, Any],
    bucket: str = "ancient-pages",
    skip: set[str] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    persist_intermediates: bool = True,
    target_dpi: int = 400,
) -> PagePreprocessOutcome:
    """Preprocess one OCR page end-to-end.

    Args:
        minio_client: A :class:`minio.Minio` client (open).
        driver: Open Neo4j driver.
        page_row: A row from :data:`_SELECT_OCR_PAGES` — must carry
            ``page_id``, ``document_id``, ``chapter_id``, ``section_id``,
            ``page_index``, ``tier``, ``image_uri``.
        bucket: MinIO bucket (default ``ancient-pages``).
        skip: Steps to skip.
        overrides: Per-step kwargs.
        persist_intermediates: If ``True`` (default), upload every
            intermediate variant (``raw.png``, ``deskew.png``, …); if
            ``False`` upload only the ``final.png`` (saves ~6× MinIO
            traffic on big runs).
        target_dpi: Recorded in provenance; the orchestrator does not
            currently re-rasterize but the field is used by Phase 3 to
            decide whether to run the lightweight or heavyweight OCR.

    Returns:
        :class:`PagePreprocessOutcome`.
    """
    page_id = page_row["page_id"]
    document_id = page_row["document_id"]
    image_uri = page_row["image_uri"]
    chapter_id = page_row.get("chapter_id")
    section_id = page_row.get("section_id")
    page_index = int(page_row.get("page_index") or 0)
    tier = page_row.get("tier") or "primary"

    started = time.monotonic()
    outcome = PagePreprocessOutcome(
        page_id=page_id, document_id=document_id, status="failed",
    )

    if not image_uri:
        outcome.status = "skipped"
        outcome.error = "page has no imageUri"
        outcome.duration_seconds = round(time.monotonic() - started, 3)
        return outcome

    page_prefix = f"{document_id}/page_{page_index:05d}"

    try:
        raw_image = _download_image(minio_client, bucket, image_uri)
    except Exception as exc:  # noqa: BLE001
        outcome.status = "failed"
        outcome.error = f"download failed ({type(exc).__name__}): {exc}"
        outcome.duration_seconds = round(time.monotonic() - started, 3)
        return outcome

    try:
        results = run_preprocess_chain(raw_image, skip=skip, overrides=overrides)
    except Exception as exc:  # noqa: BLE001
        outcome.status = "failed"
        outcome.error = f"chain failed ({type(exc).__name__}): {exc}"
        outcome.duration_seconds = round(time.monotonic() - started, 3)
        logger.exception("preprocess chain failed for %s", page_id)
        return outcome

    # Build provenance.
    final_image = results[-1].image
    final_key = f"{page_prefix}/final.png"
    provenance = PreprocessProvenance(
        page_id=page_id,
        document_id=document_id,
        source_uri=image_uri,
        final_uri=final_key,
        target_dpi=target_dpi,
        steps=[r.to_summary() for r in results],
    )

    # Upload variants.
    upload_errors: list[str] = []
    if persist_intermediates:
        for r in results:
            key = f"{page_prefix}/{r.step}.png"
            try:
                _upload_png(minio_client, bucket, key, r.image)
                provenance.variant_uris[r.step] = key
            except Exception as exc:  # noqa: BLE001
                upload_errors.append(f"{r.step}: {exc}")

    try:
        _upload_png(minio_client, bucket, final_key, final_image)
    except Exception as exc:  # noqa: BLE001
        outcome.status = "failed"
        outcome.error = f"final upload failed ({type(exc).__name__}): {exc}"
        outcome.duration_seconds = round(time.monotonic() - started, 3)
        return outcome

    # Sibling pages from split + marginalia.
    derived_pages: list[str] = []
    split_result = next((r for r in results if r.step == "split"), None)
    if split_result is not None and split_result.extras.get("right") is not None:
        right_key = f"{page_prefix}/split_right.png"
        try:
            _upload_png(minio_client, bucket, right_key, split_result.extras["right"])
            provenance.variant_uris["split_right"] = right_key
            provenance.split_children.append(right_key)
            child_id = f"{page_id}::right"
            _write_sibling_page(
                driver,
                parent_id=page_id,
                child_id=child_id,
                document_id=document_id,
                chapter_id=chapter_id,
                section_id=section_id,
                page_index=page_index,
                tier=tier,
                role="body",
                image_uri=right_key,
                provenance_json=json.dumps(
                    {"derivedFrom": "split", "parentPageId": page_id},
                    ensure_ascii=False,
                ),
            )
            derived_pages.append(child_id)
        except Exception as exc:  # noqa: BLE001
            upload_errors.append(f"split_right: {exc}")

    marginalia_result = next((r for r in results if r.step == "marginalia"), None)
    if marginalia_result is not None:
        for side, img in marginalia_result.extras.items():
            mkey = f"{page_prefix}/marginalia_{side}.png"
            try:
                _upload_png(minio_client, bucket, mkey, img)
                provenance.variant_uris[f"marginalia_{side}"] = mkey
                provenance.marginalia_children.append(mkey)
                child_id = f"{page_id}::marginalia_{side}"
                _write_sibling_page(
                    driver,
                    parent_id=page_id,
                    child_id=child_id,
                    document_id=document_id,
                    chapter_id=chapter_id,
                    section_id=section_id,
                    page_index=page_index,
                    tier=tier,
                    role="marginalia",
                    image_uri=mkey,
                    provenance_json=json.dumps(
                        {
                            "derivedFrom": "marginalia",
                            "side": side,
                            "parentPageId": page_id,
                        },
                        ensure_ascii=False,
                    ),
                )
                derived_pages.append(child_id)
            except Exception as exc:  # noqa: BLE001
                upload_errors.append(f"marginalia_{side}: {exc}")

    provenance.errors = upload_errors
    provenance.duration_seconds = round(time.monotonic() - started, 3)

    # Write back to Neo4j.
    try:
        with driver.session() as session:
            session.run(
                _PAGE_UPDATE,
                page_id=page_id,
                final_uri=final_key,
                provenance_json=json.dumps(provenance.to_dict(), ensure_ascii=False),
                status="ok" if not upload_errors else "partial",
            ).consume()
    except Exception as exc:  # noqa: BLE001
        outcome.status = "failed"
        outcome.error = f"neo4j write failed ({type(exc).__name__}): {exc}"
        outcome.duration_seconds = provenance.duration_seconds
        outcome.provenance = provenance
        return outcome

    outcome.status = "ok" if not upload_errors else "partial"
    outcome.final_uri = final_key
    outcome.provenance = provenance
    outcome.derived_pages = derived_pages
    outcome.duration_seconds = provenance.duration_seconds
    if upload_errors:
        outcome.error = "; ".join(upload_errors)
    return outcome


def _write_sibling_page(
    driver: Driver,
    *,
    parent_id: str,
    child_id: str,
    document_id: str,
    chapter_id: str | None,
    section_id: str | None,
    page_index: int,
    tier: str,
    role: str,
    image_uri: str,
    provenance_json: str,
) -> None:
    if section_id is None:
        raise ValueError(
            f"sibling page {child_id}: parent {parent_id} has no sectionId — "
            "structure planner must run first"
        )
    with driver.session() as session:
        session.run(
            _SIBLING_PAGE_UPSERT,
            parent_id=parent_id,
            child_id=child_id,
            document_id=document_id,
            chapter_id=chapter_id,
            section_id=section_id,
            page_index=page_index,
            tier=tier,
            role=role,
            image_uri=image_uri,
            provenance_json=provenance_json,
        ).consume()


# ---------------------------------------------------------------------------
# Public entry point: bulk over Neo4j.
# ---------------------------------------------------------------------------


def preprocess_pages(
    *,
    driver: Driver,
    minio_client,
    bucket: str = "ancient-pages",
    document_id: str | None = None,
    max_pages: int | None = None,
    recompute_existing: bool = False,
    skip: set[str] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    persist_intermediates: bool = True,
    target_dpi: int = 400,
    sample_size: int = 5,
) -> PreprocessRunReport:
    """Sweep every OCR page that hasn't been preprocessed and run the chain.

    Args:
        driver: Open Neo4j driver.
        minio_client: Open MinIO client.
        bucket: MinIO bucket (default ``ancient-pages``).
        document_id: Optional filter — only preprocess pages of this DOC.
        max_pages: Optional cap (handy for notebook smoke tests).
        recompute_existing: If ``True``, re-run even on pages that already
            carry ``preprocessedImageUri``.
        skip: Set of step names to skip globally.
        overrides: Per-step kwargs.
        persist_intermediates: Whether to upload every step variant.
        target_dpi: Recorded in provenance.
        sample_size: How many outcomes to retain in ``sample_outcomes``.

    Returns:
        :class:`PreprocessRunReport`.
    """
    report = PreprocessRunReport()
    started = time.monotonic()

    with driver.session() as session:
        rows = list(
            session.run(
                _SELECT_OCR_PAGES,
                recompute=recompute_existing,
                document_id=document_id,
            )
        )
    if max_pages is not None:
        rows = rows[:max_pages]
    report.pages_total = len(rows)

    sample_outcomes: list[PagePreprocessOutcome] = []
    for row in rows:
        page_row = dict(row)
        outcome = preprocess_page(
            minio_client=minio_client,
            driver=driver,
            page_row=page_row,
            bucket=bucket,
            skip=skip,
            overrides=overrides,
            persist_intermediates=persist_intermediates,
            target_dpi=target_dpi,
        )
        doc_bucket = report.by_document.setdefault(
            outcome.document_id, {"ok": 0, "partial": 0, "skipped": 0, "failed": 0}
        )
        if outcome.status == "ok":
            report.pages_processed += 1
            doc_bucket["ok"] += 1
        elif outcome.status == "partial":
            report.pages_processed += 1
            doc_bucket["partial"] += 1
        elif outcome.status == "skipped":
            report.pages_skipped += 1
            doc_bucket["skipped"] += 1
        else:
            report.pages_failed += 1
            doc_bucket["failed"] += 1
            if outcome.error:
                report.errors.append(f"{outcome.page_id}: {outcome.error}")

        if outcome.provenance is not None:
            split_step = next(
                (s for s in outcome.provenance.steps if s["step"] == "split"), None
            )
            if split_step and split_step["metrics"].get("split"):
                report.pages_split += 1
            marg_step = next(
                (s for s in outcome.provenance.steps if s["step"] == "marginalia"), None
            )
            if marg_step and (marg_step["metrics"].get("top_detected") or marg_step["metrics"].get("bottom_detected")):
                report.pages_with_marginalia += 1
            report.derived_pages_written += len(outcome.derived_pages)

        if len(sample_outcomes) < sample_size:
            sample_outcomes.append(outcome)

    report.duration_seconds = round(time.monotonic() - started, 3)
    report.sample_outcomes = [o.to_dict() for o in sample_outcomes]
    return report


def preprocessing_summary(driver: Driver) -> dict[str, Any]:
    """Roll up preprocessing coverage across the corpus (notebook check)."""

    cypher = """
    MATCH (p:PAGE)
    WHERE p.mode = 'ocr'
    WITH coalesce(p.preprocessingStatus, '(unset)') AS status,
         coalesce(p.tier, '(unset)') AS tier,
         coalesce(p.role, 'body') AS role
    RETURN status, tier, role, count(*) AS n
    """
    by_status: dict[str, int] = {}
    by_tier_status: dict[str, dict[str, int]] = {}
    by_role: dict[str, int] = {}
    with driver.session() as session:
        for row in session.run(cypher):
            s = row["status"]
            t = row["tier"]
            r = row["role"]
            n = row["n"]
            by_status[s] = by_status.get(s, 0) + n
            bucket = by_tier_status.setdefault(t, {})
            bucket[s] = bucket.get(s, 0) + n
            by_role[r] = by_role.get(r, 0) + n
    return {
        "by_status": by_status,
        "by_tier_status": by_tier_status,
        "by_role": by_role,
    }
