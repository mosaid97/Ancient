"""Single-file readers for website uploads (Track E).

The bulk corpus is PDF / EPUB / packed-md, but the website upload workflow
also accepts a single image (a scanned page / photo) or a plain text file.
These two thin readers wrap such uploads into the same :class:`PageRecord`
contract every downstream stage expects:

- :func:`iter_image_pages` — one OCR :class:`PageRecord` per image file,
  carrying ``image_bytes`` for the orchestrator to upload to MinIO.
- :func:`iter_text_pages` — one native-text :class:`PageRecord` for a
  ``.txt`` / ``.md`` upload (no OCR needed).

A directory of images is also supported (sorted, one page per image).
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path

from apps.backend.readers.base import PageRecord, resolve_tier

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
TEXT_SUFFIXES = {".txt", ".md", ".text"}


def _slug(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem.replace(' ', '_')[:60]}__{digest}"


def _image_files(p: Path) -> list[Path]:
    if p.is_dir():
        return sorted(c for c in p.iterdir() if c.suffix.lower() in IMAGE_SUFFIXES)
    return [p]


def iter_image_pages(
    path: str | Path,
    *,
    document_id: str | None = None,
    repo_root: Path | None = None,
    max_pages: int | None = None,
    **_: object,
) -> Iterator[PageRecord]:
    """Yield one OCR :class:`PageRecord` per uploaded image file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if document_id is None:
        document_id = _slug(p)
    tier = resolve_tier(p)
    src = str(p.relative_to(repo_root)) if repo_root else str(p)

    files = _image_files(p)
    if max_pages is not None:
        files = files[:max_pages]
    for idx, fpath in enumerate(files):
        image_bytes = fpath.read_bytes()
        yield PageRecord(
            document_id=document_id,
            source_path=src,
            page_index=idx,
            mode="ocr",
            image_uri=f"{document_id}/page_{idx:05d}.png",
            image_bytes=image_bytes,
            role="body",
            tier=tier,
            metadata={"original_filename": fpath.name, "upload": True},
        )


def iter_text_pages(
    path: str | Path,
    *,
    document_id: str | None = None,
    repo_root: Path | None = None,
    chars_per_page: int = 4000,
    max_pages: int | None = None,
    **_: object,
) -> Iterator[PageRecord]:
    """Yield native-text :class:`PageRecord`s for an uploaded text file.

    Long files are split into ``chars_per_page``-sized pages so the spine /
    chunker behave the same as for multi-page documents.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if document_id is None:
        document_id = _slug(p)
    tier = resolve_tier(p)
    src = str(p.relative_to(repo_root)) if repo_root else str(p)

    text = p.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return
    pages = [text[i : i + chars_per_page] for i in range(0, len(text), chars_per_page)]
    if max_pages is not None:
        pages = pages[:max_pages]
    for idx, page_text in enumerate(pages):
        yield PageRecord(
            document_id=document_id,
            source_path=src,
            page_index=idx,
            mode="native_text",
            text=page_text,
            role="body",
            tier=tier,
            metadata={"char_count": len(page_text), "upload": True},
        )
