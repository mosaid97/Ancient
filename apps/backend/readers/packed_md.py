"""Reader for ``.packed/`` markdown trees produced by Obsidian's Epub Importer.

The ``raw/Secondary/comprehensive_epub/<book>.packed/`` layout looks like::

    <book>.packed/
        <book>.packed.md          # the spine index (Obsidian-flavored bullets)
        书名页.md
        版权页.md
        引 言.md
        第一部分 两汉时期/
            第一部分 两汉时期.md
            第一章 .../
                ...
        images/
            cover.jpeg
            ...

Each ``.md`` is treated as one native-text page. SECTION hierarchy is
preserved on ``PAGE.metadata['section_path']`` (list of directory segments
under the ``.packed/`` root). The Phase-1 structure extractor
(:mod:`apps.backend.pipeline.structure`) reads that list to build the
``(:CHAPTER)-[:INCLUDE]->(:SECTION)-[:INCLUDE]->(:PAGE)`` spine — top-level
directory → CHAPTER, second-level → SECTION (plan §5, §6 Stage 1).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator
from pathlib import Path

from apps.backend.readers.base import (
    NATIVE_TEXT_MIN_CHARS,
    PageRecord,
    resolve_tier,
)

logger = logging.getLogger(__name__)

YAML_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _slug(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem.replace(' ', '_').replace('.packed', '')[:60]}__{digest}"


def _strip_frontmatter(text: str) -> tuple[str, dict[str, str]]:
    """Return ``(body, frontmatter_dict)``. Best-effort YAML parse."""
    m = YAML_FRONTMATTER_RE.match(text)
    if not m:
        return text, {}
    body = text[m.end():]
    fm_block = text[: m.end()]
    fm: dict[str, str] = {}
    for line in fm_block.splitlines():
        if line.startswith("---") or not line.strip():
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            fm[key.strip()] = val.strip()
    return body, fm


def _section_path(md_file: Path, root: Path) -> list[str]:
    rel = md_file.relative_to(root)
    parts = list(rel.parts)
    return [seg for seg in parts[:-1] if seg]


def iter_pages(
    path: str | Path,
    *,
    max_pages: int | None = None,
    document_id: str | None = None,
    repo_root: Path | None = None,
    native_min_chars: int = NATIVE_TEXT_MIN_CHARS,
    include_index: bool = False,
) -> Iterator[PageRecord]:
    """Walk a packed-markdown tree and yield one :class:`PageRecord` per ``.md``.

    Args:
        path: Path to the ``<book>.packed/`` directory (or any ancestor that
            unambiguously contains exactly one ``.packed/`` child).
        max_pages: Stop after N markdown files.
        document_id: Override slug.
        repo_root: For relative-path bookkeeping.
        native_min_chars: Skip files shorter than this many chars.
        include_index: Include the top-level ``<book>.packed.md`` index file
            (it's just a TOC of wikilinks; usually noisy).

    Yields:
        :class:`PageRecord` per markdown file, in deterministic-sorted order.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if not p.is_dir():
        raise NotADirectoryError(p)

    if document_id is None:
        document_id = _slug(p)
    tier = resolve_tier(p)
    src = str(p.relative_to(repo_root)) if repo_root else str(p)

    md_files = sorted(p.rglob("*.md"))
    if not md_files:
        logger.warning("no .md files under %s", p)
        return

    upper = len(md_files) if max_pages is None else min(len(md_files), max_pages)
    yielded = 0

    for md in md_files:
        if yielded >= upper:
            break
        if not include_index and md.name.endswith(".packed.md"):
            continue
        try:
            raw = md.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to read %s: %s", md, exc)
            continue
        body, fm = _strip_frontmatter(raw)
        body = body.strip()
        if len(body) < native_min_chars and md.name not in {"书名页.md", "版权页.md"}:
            logger.debug("skipping short markdown %s (%d chars)", md, len(body))
            continue
        wikilink_targets = [m.group(1) for m in WIKI_LINK_RE.finditer(body)]
        section_path = _section_path(md, p)
        meta = {
            "section_path": section_path,
            "frontmatter": fm,
            "wikilinks": wikilink_targets[:30],
            "char_count": len(body),
            "rel_path": str(md.relative_to(p)),
        }
        role = (
            "frontmatter"
            if md.name in {"Cover.md", "书名页.md", "版权页.md", "目 录.md"}
            else "body"
        )
        yield PageRecord(
            document_id=document_id,
            source_path=src,
            page_index=yielded,
            mode="native_text",
            text=body,
            role=role,
            tier=tier,
            metadata=meta,
        )
        yielded += 1


def quick_summary(path: str | Path, sample_pages: int = 20) -> dict:
    """Mirror of the EPUB / PDF quick_summary."""
    p = Path(path)
    md_files = sorted(p.rglob("*.md"))
    sample = min(len(md_files), sample_pages)
    char_total = 0
    for md in md_files[:sample]:
        try:
            raw = md.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        body, _ = _strip_frontmatter(raw)
        char_total += len(body.strip())
    return {
        "path": str(p),
        "tier": resolve_tier(p),
        "page_count": len(md_files),
        "sampled": sample,
        "native_pages_in_sample": sample,
        "ocr_pages_in_sample": 0,
        "mean_chars_per_sampled_page": char_total / sample if sample else 0,
    }
