"""Format-aware readers for the ingestion pipeline (Phase 1).

Three reader backends, all yielding the unified :class:`PageRecord`:

- :mod:`apps.backend.readers.epub_reader`   — ebooklib spine walker.
- :mod:`apps.backend.readers.pdf_reader`    — PyMuPDF text-vs-image-page detector.
- :mod:`apps.backend.readers.packed_md`     — pre-chunked ``.packed/`` markdown trees.

Each reader also exposes a lightweight ``read_toc(path)`` helper used by
:mod:`apps.backend.pipeline.structure` for TOC-first CHAPTER/SECTION
extraction (plan §6 Stage 1, v2.1 spine).
"""

from pathlib import Path
from typing import Any

from apps.backend.readers.base import (
    NATIVE_TEXT_MIN_CHARS,
    PageMode,
    PageRecord,
    PageRole,
    Tier,
    detect_reader,
    iter_pages,
    resolve_tier,
)


def read_toc(path: str | Path) -> list[tuple[int, str, Any]]:
    """Dispatch to the right reader's TOC extractor.

    Returns a uniform ``[(level, title, locator), ...]`` list where ``locator``
    is a ``int`` (0-based page_index) for PDFs and a ``str`` (manifest href)
    for EPUBs. Packed-md trees don't expose a TOC here — the structure
    extractor reads ``PageRecord.metadata['section_path']`` directly.

    Returns ``[]`` if the backend has no TOC support or the document carries
    no embedded TOC.
    """
    backend = detect_reader(path)
    if backend == "pdf":
        from apps.backend.readers import pdf_reader

        return list(pdf_reader.read_toc(path))
    if backend == "epub":
        from apps.backend.readers import epub_reader

        return list(epub_reader.read_toc(path))
    return []


__all__ = [
    "NATIVE_TEXT_MIN_CHARS",
    "PageMode",
    "PageRecord",
    "PageRole",
    "Tier",
    "detect_reader",
    "iter_pages",
    "read_toc",
    "resolve_tier",
]
