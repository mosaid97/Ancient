"""PyMuPDF-backed PDF reader: text-vs-image detection per page (plan Stage 1).

Per page:

1. ``page.get_text("text")``; if ``len(stripped) >= NATIVE_TEXT_MIN_CHARS``,
   yield a ``native_text`` :class:`PageRecord`.
2. Otherwise rasterize at ``target_dpi`` (default 400 for 影印本 古籍, 300 for
   modern reprints — caller decides via ``target_dpi`` arg) and yield an
   ``ocr`` :class:`PageRecord` whose ``image_bytes`` the orchestrator will
   upload to MinIO.

Large-PDF guard: pages are processed lazily so callers can iterate millions
of pages without holding image bytes in memory. Plan §6 Stage 1 calls for
100-page batches with ``gc.collect()`` for the 800+ MB primaries (册府元龟,
通典, 唐律疏議箋解); the orchestrator owns batching.
"""

from __future__ import annotations

import gc
import hashlib
import io
import logging
from collections.abc import Iterator
from pathlib import Path

import fitz  # PyMuPDF

from apps.backend.readers.base import (
    NATIVE_TEXT_MIN_CHARS,
    PageRecord,
    resolve_tier,
)

logger = logging.getLogger(__name__)

DEFAULT_DPI: int = 400
"""Default rasterization DPI. 400 is the recommended target for scanned
ancient editions (影印本); the orchestrator may drop this to 300 for modern
reprints based on inferred ``editorial_layers`` (plan §6 Stage 1).
"""

GC_BATCH: int = 100
"""Force a ``gc.collect()`` every N pages so the 872 MB 册府元龟 PDF doesn't
balloon RSS over 8 GB. Aligns with the plan's large-PDF guard.
"""


def _slug(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem.replace(' ', '_')[:60]}__{digest}"


def iter_pages(
    path: str | Path,
    *,
    target_dpi: int = DEFAULT_DPI,
    max_pages: int | None = None,
    document_id: str | None = None,
    repo_root: Path | None = None,
    native_min_chars: int = NATIVE_TEXT_MIN_CHARS,
    force_ocr: bool = False,
) -> Iterator[PageRecord]:
    """Iterate :class:`PageRecord` over a PDF, lazily.

    Args:
        path: Path to a PDF.
        target_dpi: DPI to rasterize OCR-bound pages at (default 400).
        max_pages: If set, stop after this many pages (handy for notebooks).
        document_id: Override document slug; default = derived from filename.
        repo_root: Used to compute a relative ``source_path`` field.
        native_min_chars: Threshold for native-text classification.
        force_ocr: If True, ignore any embedded text layer and always rasterize
            every page for OCR. Use for PDFs with corrupted font CMap encodings
            whose ``get_text()`` returns garbled characters even when character
            count exceeds ``native_min_chars`` (e.g. 爱宕元_唐代的官荫入仕).

    Yields:
        One :class:`PageRecord` per page.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if document_id is None:
        document_id = _slug(p)
    tier = resolve_tier(p)
    src = str(p.relative_to(repo_root)) if repo_root else str(p)

    zoom = target_dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(p)
    try:
        total = doc.page_count
        upper = min(total, max_pages) if max_pages else total
        logger.info(
            "pdf_reader: %s (pages=%d, ingesting=%d, dpi=%d, doc_id=%s)",
            p.name,
            total,
            upper,
            target_dpi,
            document_id,
        )
        for idx in range(upper):
            page = doc.load_page(idx)
            text = page.get_text("text") or ""
            if not force_ocr and len(text.strip()) >= native_min_chars:
                yield PageRecord(
                    document_id=document_id,
                    source_path=src,
                    page_index=idx,
                    mode="native_text",
                    text=text,
                    role="body",
                    tier=tier,
                    metadata={
                        "page_label": page.get_label() or str(idx + 1),
                        "char_count": len(text),
                        "rect": list(page.rect),
                    },
                )
            else:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                buf = io.BytesIO()
                buf.write(pix.tobytes("png"))
                pix = None  # release before next iter
                yield PageRecord(
                    document_id=document_id,
                    source_path=src,
                    page_index=idx,
                    mode="ocr",
                    image_uri=f"{document_id}/page_{idx:05d}.png",
                    image_bytes=buf.getvalue(),
                    role="body",
                    tier=tier,
                    metadata={
                        "page_label": page.get_label() or str(idx + 1),
                        "dpi": target_dpi,
                        "raw_text_chars": len(text.strip()),
                        "rect": list(page.rect),
                    },
                )
            if (idx + 1) % GC_BATCH == 0:
                gc.collect()
    finally:
        doc.close()


def read_toc(path: str | Path) -> list[tuple[int, str, int]]:
    """Return the PDF's table of contents as ``(level, title, page_index)``.

    Uses :py:meth:`fitz.DOCUMENT.get_toc` (``simple=True``) which yields
    ``[level, title, 1-based page]`` rows; we convert the page number to a
    0-based ``page_index`` so it lines up with :class:`PageRecord.page_index`.
    Returns ``[]`` when the PDF has no embedded TOC (very common for
    影印本 / scanned ancient editions) — the structure planner then falls
    back to either Stage 3.5 (LLM) or a synthetic single-CHAPTER plan.

    Plan reference: §6 Stage 1 (TOC-first structure extraction).

    Args:
        path: PDF path.

    Returns:
        List of ``(level, title, page_index)`` tuples in document order.
        ``level`` is 1-based (1 == top-level / CHAPTER candidate);
        ``page_index`` is 0-based.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    doc = fitz.open(p)
    try:
        raw = doc.get_toc(simple=True)
    finally:
        doc.close()
    out: list[tuple[int, str, int]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        level, title, page_no = row[0], row[1], row[2]
        try:
            lvl = int(level)
            idx = max(0, int(page_no) - 1)
        except (TypeError, ValueError):
            continue
        out.append((lvl, str(title).strip(), idx))
    return out


def quick_summary(path: str | Path, sample_pages: int = 20) -> dict:
    """One-shot diagnostic for a PDF: text-vs-image ratio over a head sample.

    Used by the notebook to render the "format-aware routing" decision matrix.

    Args:
        path: PDF path.
        sample_pages: How many leading pages to sample.

    Returns:
        Dict with page count, native vs ocr counts, mean chars per page,
        the path's resolved tier, and the TOC entry count.
    """
    p = Path(path)
    doc = fitz.open(p)
    try:
        total = doc.page_count
        sample = min(total, sample_pages)
        native = 0
        char_total = 0
        for idx in range(sample):
            page = doc.load_page(idx)
            text = page.get_text("text") or ""
            chars = len(text.strip())
            char_total += chars
            if chars >= NATIVE_TEXT_MIN_CHARS:
                native += 1
        toc_entries = len(doc.get_toc(simple=True) or [])
        return {
            "path": str(p),
            "tier": resolve_tier(p),
            "page_count": total,
            "sampled": sample,
            "native_pages_in_sample": native,
            "ocr_pages_in_sample": sample - native,
            "mean_chars_per_sampled_page": char_total / sample if sample else 0,
            "toc_entries": toc_entries,
        }
    finally:
        doc.close()
