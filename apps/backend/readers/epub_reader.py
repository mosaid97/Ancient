"""ebooklib-backed EPUB reader: spine walker + native-vs-scanned detection.

Per spine item:

1. Decode the XHTML body, strip tags, count chars (BeautifulSoup).
2. Count embedded ``<img>`` references.
3. ``len(text) >= NATIVE_TEXT_MIN_CHARS AND img_count <= 1`` -> native_text.
4. Otherwise -> ocr (image hauled out of the manifest and queued for MinIO).

Edge cases handled:

- Cover image (manifest item with id ``cover``) yields a ``role='cover'`` page.
- Empty / nav-only items are skipped silently.
- Non-spine images-only items are skipped (typical 插圖 inside chapters are
  reachable via the manifest but don't need their own PAGE node).
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from collections.abc import Iterator
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import ITEM_DOCUMENT, ITEM_IMAGE, epub

# EPUB content is XHTML; using the lxml HTML parser is fine, just noisy.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from apps.backend.readers.base import (
    NATIVE_TEXT_MIN_CHARS,
    PageRecord,
    resolve_tier,
)

logger = logging.getLogger(__name__)


def _slug(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem.replace(' ', '_')[:60]}__{digest}"


def _decode_body(item: epub.EpubItem) -> tuple[str, int]:
    """Return ``(plain_text, embedded_img_count)`` for an EPUB document item."""
    raw = item.get_content() or b""
    if not raw:
        return "", 0
    try:
        soup = BeautifulSoup(raw, "lxml")
    except Exception:  # noqa: BLE001 — fall back to html parser
        soup = BeautifulSoup(raw, "html.parser")
    img_count = len(soup.find_all("img"))
    for tag in soup(("script", "style")):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return text, img_count


def iter_pages(
    path: str | Path,
    *,
    max_pages: int | None = None,
    document_id: str | None = None,
    repo_root: Path | None = None,
    native_min_chars: int = NATIVE_TEXT_MIN_CHARS,
    include_cover: bool = True,
) -> Iterator[PageRecord]:
    """Yield :class:`PageRecord` per spine item of an EPUB.

    Args:
        path: Path to an ``.epub`` file.
        max_pages: Stop after N spine items.
        document_id: Override slug.
        repo_root: For relative-path bookkeeping.
        native_min_chars: Threshold for native-text classification.
        include_cover: Yield a synthetic cover page for the manifest cover.

    Yields:
        :class:`PageRecord` per spine item.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if document_id is None:
        document_id = _slug(p)
    tier = resolve_tier(p)
    src = str(p.relative_to(repo_root)) if repo_root else str(p)

    book = epub.read_epub(str(p), options={"ignore_ncx": True})
    spine_items = [item_id for item_id, _ in book.spine]
    logger.info(
        "epub_reader: %s (spine=%d, doc_id=%s, tier=%s)",
        p.name,
        len(spine_items),
        document_id,
        tier,
    )

    page_index = 0

    if include_cover:
        cover_item = book.get_item_with_id("cover-image") or book.get_item_with_id("cover")
        if cover_item is None:
            for item in book.get_items_of_type(ITEM_IMAGE):
                name = (item.get_name() or "").lower()
                if "cover" in name:
                    cover_item = item
                    break
        if cover_item is not None and getattr(cover_item, "get_content", None):
            data = cover_item.get_content() or b""
            if data:
                ext = Path(cover_item.get_name() or "cover.jpg").suffix or ".jpg"
                yield PageRecord(
                    document_id=document_id,
                    source_path=src,
                    page_index=page_index,
                    mode="ocr",
                    image_uri=f"{document_id}/cover{ext}",
                    image_bytes=data,
                    role="cover",
                    tier=tier,
                    metadata={"manifest_id": cover_item.get_id()},
                )
                page_index += 1

    upper = max_pages if max_pages is not None else len(spine_items)

    for spine_item_id in spine_items:
        if page_index >= upper:
            break
        item = book.get_item_with_id(spine_item_id)
        if item is None:
            continue
        if item.get_type() != ITEM_DOCUMENT:
            continue
        text, img_count = _decode_body(item)
        chars = len(text.strip())
        # Empty spine items are usually nav stubs / chapter dividers; not
        # ocr-bound, just structural noise. Skip silently.
        if chars == 0 and img_count == 0:
            logger.debug("epub_reader: skipping empty spine item %s", spine_item_id)
            continue
        meta = {
            "spine_id": spine_item_id,
            "href": item.get_name(),
            "img_count": img_count,
            "char_count": chars,
        }
        if chars >= native_min_chars and img_count <= 1:
            yield PageRecord(
                document_id=document_id,
                source_path=src,
                page_index=page_index,
                mode="native_text",
                text=text,
                role="frontmatter" if chars < 600 and page_index <= 5 else "body",
                tier=tier,
                metadata=meta,
            )
            page_index += 1
        else:
            # Image-bundle spine items (common in scanned ancient EPUBs):
            # one HTML wrapper carrying N <img> tags, each a true page.
            # Yield one OCR PageRecord per image so Phase 3 can OCR them
            # individually.
            try:
                soup = BeautifulSoup(item.get_content() or b"", "lxml")
            except Exception:  # noqa: BLE001
                soup = BeautifulSoup(item.get_content() or b"", "html.parser")
            img_tags = soup.find_all("img")
            if not img_tags:
                # Image-less, near-empty wrapper: skip silently.
                logger.debug(
                    "epub_reader: skipping image-less stub %s", spine_item_id
                )
                continue
            base_href = Path(item.get_name())
            for sub_idx, img_tag in enumerate(img_tags):
                src_attr = img_tag.get("src")
                if not src_attr:
                    continue
                manifest_href = base_href.parent.joinpath(src_attr).as_posix()
                manifest_item = book.get_item_with_href(manifest_href)
                if manifest_item is None:
                    continue
                data = manifest_item.get_content()
                if not data:
                    continue
                ext = Path(manifest_item.get_name() or "page.png").suffix or ".png"
                yield PageRecord(
                    document_id=document_id,
                    source_path=src,
                    page_index=page_index,
                    mode="ocr",
                    image_uri=f"{document_id}/page_{page_index:05d}{ext}",
                    image_bytes=data,
                    role="body",
                    tier=tier,
                    metadata={
                        **meta,
                        "manifest_href": manifest_href,
                        "spine_sub_index": sub_idx,
                        "spine_image_count": img_count,
                    },
                )
                page_index += 1


def _flatten_toc(items: list, level: int = 1) -> list[tuple[int, str, str]]:
    """Recursively flatten an ebooklib ``book.toc`` into ``(level, title, href)``.

    ``book.toc`` is a nested list where each entry is either an
    :class:`ebooklib.epub.Link` (leaf) or a tuple
    ``(SECTION, [child_items])`` (branch). The href is stripped of any
    fragment (``#anchor``) so it matches the spine item's ``get_name()``.
    """
    out: list[tuple[int, str, str]] = []
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            section, children = item
            title = getattr(section, "title", None) or ""
            href = getattr(section, "href", None) or ""
            if title:
                href = href.split("#", 1)[0] if href else ""
                out.append((level, str(title).strip(), str(href)))
            out.extend(_flatten_toc(list(children or []), level=level + 1))
        else:
            title = getattr(item, "title", None) or ""
            href = getattr(item, "href", None) or ""
            if not title:
                continue
            href = href.split("#", 1)[0] if href else ""
            out.append((level, str(title).strip(), str(href)))
    return out


def read_toc(path: str | Path) -> list[tuple[int, str, str]]:
    """Return the EPUB's TOC as ``(level, title, href)`` rows.

    Walks ``book.toc`` with NCX enabled (in contrast to
    :func:`iter_pages` which sets ``ignore_ncx=True`` for speed). Hrefs are
    relative paths matching the manifest. Returns ``[]`` when the EPUB has
    no NCX or an empty TOC; the structure planner then falls back to a
    synthetic single-CHAPTER plan.

    Plan reference: §6 Stage 1 (TOC-first structure extraction).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    book = epub.read_epub(str(p), options={"ignore_ncx": False})
    toc = list(book.toc or [])
    return _flatten_toc(toc, level=1)


def read_opf_hints(path: str | Path) -> dict[str, list[str]]:
    """Return Dublin-Core metadata from an EPUB's OPF package for enrichment.

    Args:
        path: Path to an ``.epub`` file.

    Returns:
        Dict with string-list values for the four DC keys that carry editorial
        information::

            {
                "creators":   [...],   # dc:creator  (author + editor strings)
                "publishers": [...],   # dc:publisher
                "titles":     [...],   # dc:title
                "dates":      [...],   # dc:date
            }

        Each list contains only non-empty string values; unknown keys return
        empty lists.  Falls back to all-empty-lists on any read error.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    try:
        book = epub.read_epub(str(p), options={"ignore_ncx": True})
    except Exception:  # noqa: BLE001
        logger.warning("read_opf_hints: could not open %s", p.name)
        return {"creators": [], "publishers": [], "titles": [], "dates": []}

    dc_ns = "http://purl.org/dc/elements/1.1/"
    ns_meta: dict = book.metadata.get(dc_ns, {})

    def _vals(key: str) -> list[str]:
        return [v for v, _ in ns_meta.get(key, []) if isinstance(v, str) and v.strip()]

    return {
        "creators": _vals("creator"),
        "publishers": _vals("publisher"),
        "titles": _vals("title"),
        "dates": _vals("date"),
    }


def quick_summary(path: str | Path, sample_pages: int = 20) -> dict:
    """One-shot diagnostic mirror of :func:`pdf_reader.quick_summary`."""
    p = Path(path)
    book = epub.read_epub(str(p), options={"ignore_ncx": False})
    spine = [item_id for item_id, _ in book.spine]
    total = len(spine)
    sample = min(total, sample_pages)
    native = 0
    char_total = 0
    for spine_id in spine[:sample]:
        item = book.get_item_with_id(spine_id)
        if item is None or item.get_type() != ITEM_DOCUMENT:
            continue
        text, img_count = _decode_body(item)
        chars = len(text.strip())
        char_total += chars
        if chars >= NATIVE_TEXT_MIN_CHARS and img_count <= 1:
            native += 1
    toc_entries = len(_flatten_toc(list(book.toc or []), level=1))
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
