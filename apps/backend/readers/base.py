"""Common types + dispatcher for the format-aware readers (plan §6 Stage 1).

All readers yield a :class:`PageRecord`. Native-text pages carry their text
inline; OCR-bound pages carry an ``image_uri`` (typically a MinIO key) so the
downstream OCR pipeline can fetch them without re-rasterizing.

Tier is resolved deterministically from the source path under ``raw/`` per
plan §2.6 and is propagated to every PAGE derived from the document.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PageMode = Literal["native_text", "ocr"]
PageRole = Literal["body", "marginalia", "cover", "frontmatter"]
Tier = Literal["primary", "secondary"]

NATIVE_TEXT_MIN_CHARS: int = 200
"""Threshold for classifying a PDF page or EPUB spine item as native-text.

Mirrors plan §6 Stage 1: ``len(txt.strip()) >= 200`` -> native; otherwise
the page is rasterized for OCR. Set to 200 because Tang-era 古籍 EPUBs
sometimes have very short heading-only items.
"""


@dataclass
class PageRecord:
    """Uniform record yielded by every reader.

    Attributes:
        document_id: Stable id (typically a slug derived from the filename).
        source_path: Path of the source file relative to the repo root.
        page_index: Zero-based page index (within the document).
        mode: ``native_text`` or ``ocr``.
        text: Extracted text for native-text pages; ``None`` for OCR pages.
        image_uri: Object-store key for the rasterized image (OCR pages),
            ``None`` for native-text pages.
        image_bytes: Optional inline image bytes — provided by some readers
            for the orchestrator to upload to MinIO. Cleared after upload.
        role: Body / marginalia / cover / frontmatter.
        tier: ``primary`` or ``secondary`` (resolved by :func:`resolve_tier`).
        language_hint: Pre-OCR script-class hint (set by Phase 1b for native
            pages or Phase 3a preview-OCR for scanned pages). May be ``None``.
        chapter_id: Owning CHAPTER (plan v2.1 spine, §5). Stamped by
            :mod:`apps.backend.pipeline.structure` after structure extraction
            (TOC / NCX / packed-md tree / synthetic). Will be ``None`` until
            the structure planner has run.
        section_id: Owning SECTION (plan v2.1 spine, §5). Same lifecycle as
            ``chapter_id``. The authoritative graph edge is
            ``(:SECTION)-[:INCLUDE]->(:PAGE)``; this field is a denormalized
            copy for the orchestrator's Cypher write.
        metadata: Free-form per-reader extras (spine href, PDF page label,
            section path, etc.).
    """

    document_id: str
    source_path: str
    page_index: int
    mode: PageMode
    text: str | None = None
    image_uri: str | None = None
    image_bytes: bytes | None = None
    role: PageRole = "body"
    tier: Tier = "primary"
    language_hint: str | None = None
    chapter_id: str | None = None
    section_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text or "")


def resolve_tier(path: str | Path) -> Tier:
    """Determine ``primary`` vs ``secondary`` from the source path.

    Rule (plan §2.6, deterministic): files under ``raw/Primary/`` are
    primary; files under ``raw/Secondary/`` (any depth) are secondary;
    everything else defaults to ``primary`` and emits a debug log.

    Args:
        path: A relative or absolute filesystem path.

    Returns:
        ``"primary"`` or ``"secondary"``.
    """
    p = Path(path)
    parts = [seg.lower() for seg in p.parts]
    for seg in parts:
        if seg == "primary":
            return "primary"
        if seg == "secondary":
            return "secondary"
    return "primary"


def detect_reader(path: str | Path) -> str:
    """Pick a reader name based on the path.

    Returns one of ``epub``, ``pdf``, ``packed_md`` or raises ``ValueError``.
    """
    p = Path(path)
    if p.is_dir() and p.name.endswith(".packed"):
        return "packed_md"
    suffix = p.suffix.lower()
    if suffix == ".epub":
        return "epub"
    if suffix == ".pdf":
        return "pdf"
    if p.is_dir():
        if any(child.suffix == ".md" for child in p.iterdir()):
            return "packed_md"
    raise ValueError(f"no reader for {path}: unknown format")


def iter_pages(path: str | Path, **kwargs: Any) -> Iterator[PageRecord]:
    """Dispatch to the right reader based on file extension / dir shape.

    Args:
        path: Source path.
        **kwargs: Reader-specific overrides (``target_dpi``, ``max_pages``, …).

    Yields:
        :class:`PageRecord` instances, one per page / section.
    """
    backend = detect_reader(path)
    if backend == "epub":
        from apps.backend.readers import epub_reader

        yield from epub_reader.iter_pages(path, **kwargs)
    elif backend == "pdf":
        from apps.backend.readers import pdf_reader

        yield from pdf_reader.iter_pages(path, **kwargs)
    elif backend == "packed_md":
        from apps.backend.readers import packed_md

        yield from packed_md.iter_pages(path, **kwargs)
    else:  # pragma: no cover — guarded by detect_reader
        raise ValueError(f"unsupported reader backend: {backend}")
