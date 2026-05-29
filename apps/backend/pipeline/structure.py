"""CHAPTER/SECTION structure extraction (plan §6 Stage 1 + Stage 3.5).

The Phase-1 ingest orchestrator (:mod:`apps.backend.pipeline.ingest`) calls
:func:`plan_structure` after materializing the document's :class:`PageRecord`
list. The planner builds a :class:`StructurePlan` containing exactly one
``(:CHAPTER)`` and ``(:SECTION)`` family that covers every page; the
orchestrator then writes it to Neo4j (via :func:`write_plan`) and stamps
each ``PageRecord`` with its owning ``chapter_id`` + ``section_id`` (via
:func:`stamp_pages`) before the PAGE upsert step.

Detection order (deterministic per plan §6 Stage 1):

1. **TOC-first** — PDFs via :func:`pdf_reader.read_toc`
   (``fitz.DOCUMENT.get_toc(simple=True)``), EPUBs via
   :func:`epub_reader.read_toc` (NCX walker), packed-md trees via
   ``PageRecord.metadata['section_path']`` (top-level dir → CHAPTER,
   second-level dir → SECTION).
2. **LLM fallback** — :func:`llm_extract_structure` (plan §6 Stage 3.5).
   Gated on ``tier=='primary'`` and only run *after* Phase-3 fusion has
   produced ``PAGE.text_fused``. **Not invoked from Phase 1** — Phase 1
   marks the document with ``detectionMethod='pending'`` and the post-fusion
   reprocessor (Phase 3.5) calls :func:`llm_extract_structure` to refresh.
3. **Synthetic** — :func:`synthetic_plan`. One CHAPTER (``title='(全書)'``)
   + one SECTION covering the full page range. Always succeeds, so every
   DOCUMENT leaves Phase 1 with a non-empty spine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neo4j import Driver

from apps.backend.readers import PageRecord
from apps.backend.readers.base import detect_reader

logger = logging.getLogger(__name__)


SYNTHETIC_CHAPTER_TITLE = "(全書)"
"""Fallback title used by :func:`synthetic_plan` when no structure is found."""

SYNTHETIC_SECTION_TITLE = "(全篇)"
"""Fallback section title for the synthetic plan."""


# ---------------------------------------------------------------------------
# Dataclasses (plan §5 spine, v2.1).
# ---------------------------------------------------------------------------


@dataclass
class ChapterRecord:
    """Mirrors the ``(:CHAPTER)`` node spec from plan §5."""

    id: str
    document_id: str
    ordinal: int
    title: str
    start_page_index: int
    end_page_index: int
    tier: str
    source_language: str | None = None
    detection_method: str = "synthetic"
    source_heading: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "documentId": self.document_id,
            "ordinal": self.ordinal,
            "title": self.title,
            "startPageIndex": self.start_page_index,
            "endPageIndex": self.end_page_index,
            "tier": self.tier,
            "sourceLanguage": self.source_language,
            "detectionMethod": self.detection_method,
            "sourceHeading": self.source_heading,
        }


@dataclass
class SectionRecord:
    """Mirrors the ``(:SECTION)`` node spec from plan §5."""

    id: str
    chapter_id: str
    document_id: str
    ordinal: int
    title: str
    start_page_index: int
    end_page_index: int
    tier: str
    source_language: str | None = None
    detection_method: str = "synthetic"
    source_heading: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "chapterId": self.chapter_id,
            "documentId": self.document_id,
            "ordinal": self.ordinal,
            "title": self.title,
            "startPageIndex": self.start_page_index,
            "endPageIndex": self.end_page_index,
            "tier": self.tier,
            "sourceLanguage": self.source_language,
            "detectionMethod": self.detection_method,
            "sourceHeading": self.source_heading,
        }


@dataclass
class StructurePlan:
    """Complete CHAPTER+SECTION spine for a single DOCUMENT.

    ``detection_method`` is the *overall* origin (the strongest signal the
    planner used). Per-record ``detection_method`` may differ — e.g. a TOC
    that misses the last chapter would have ``detection_method='toc'`` on
    every TOC-derived CHAPTER and ``'synthetic'`` on the trailing filler.
    """

    document_id: str
    chapters: list[ChapterRecord] = field(default_factory=list)
    sections: list[SectionRecord] = field(default_factory=list)
    detection_method: str = "synthetic"

    @property
    def total_chapters(self) -> int:
        return len(self.chapters)

    @property
    def total_sections(self) -> int:
        return len(self.sections)


# ---------------------------------------------------------------------------
# Synthetic / fallback plan.
# ---------------------------------------------------------------------------


def synthetic_plan(
    *,
    document_id: str,
    tier: str,
    num_pages: int,
    source_language: str | None = None,
) -> StructurePlan:
    """One synthetic CHAPTER + SECTION covering every page.

    Always succeeds; called whenever TOC / NCX / packed-md tree extraction
    yields no structure. ``num_pages`` must be ``>= 1``.
    """
    if num_pages < 1:
        raise ValueError("synthetic_plan requires at least one page")
    chap = ChapterRecord(
        id=f"{document_id}::ch000",
        document_id=document_id,
        ordinal=0,
        title=SYNTHETIC_CHAPTER_TITLE,
        start_page_index=0,
        end_page_index=num_pages - 1,
        tier=tier,
        source_language=source_language,
        detection_method="synthetic",
    )
    sec = SectionRecord(
        id=f"{document_id}::ch000::sec000",
        chapter_id=chap.id,
        document_id=document_id,
        ordinal=0,
        title=SYNTHETIC_SECTION_TITLE,
        start_page_index=0,
        end_page_index=num_pages - 1,
        tier=tier,
        source_language=source_language,
        detection_method="synthetic",
    )
    return StructurePlan(
        document_id=document_id,
        chapters=[chap],
        sections=[sec],
        detection_method="synthetic",
    )


# ---------------------------------------------------------------------------
# PDF: fitz.get_toc(simple=True) → StructurePlan.
# ---------------------------------------------------------------------------


def _plan_from_pdf_toc(
    *,
    document_id: str,
    tier: str,
    num_pages: int,
    toc: list[tuple[int, str, int]],
    source_language: str | None = None,
) -> StructurePlan | None:
    """Turn a PDF TOC into chapters + sections; return None if the TOC is empty."""
    if not toc:
        return None

    chapters: list[ChapterRecord] = []
    sections: list[SectionRecord] = []
    current_chapter: ChapterRecord | None = None
    section_ordinal_in_chapter = 0

    # First pass: emit chapters (level 1) and sections (level >= 2). Pages
    # ranges are filled in the second pass once we know the next-sibling
    # boundary.
    for level, title, page_index in toc:
        page_index = max(0, min(num_pages - 1, page_index))
        if level <= 1 or current_chapter is None:
            ordinal = len(chapters)
            current_chapter = ChapterRecord(
                id=f"{document_id}::ch{ordinal:03d}",
                document_id=document_id,
                ordinal=ordinal,
                title=title or f"卷 {ordinal + 1}",
                start_page_index=page_index,
                end_page_index=page_index,  # patched below
                tier=tier,
                source_language=source_language,
                detection_method="toc",
                source_heading=title,
            )
            chapters.append(current_chapter)
            section_ordinal_in_chapter = 0
        else:
            sec_ordinal = section_ordinal_in_chapter
            section_ordinal_in_chapter += 1
            sec_id = f"{current_chapter.id}::sec{sec_ordinal:03d}"
            sections.append(
                SectionRecord(
                    id=sec_id,
                    chapter_id=current_chapter.id,
                    document_id=document_id,
                    ordinal=sec_ordinal,
                    title=title or f"篇 {sec_ordinal + 1}",
                    start_page_index=page_index,
                    end_page_index=page_index,  # patched below
                    tier=tier,
                    source_language=source_language,
                    detection_method="toc",
                    source_heading=title,
                )
            )

    if not chapters:
        return None

    # Ensure every chapter has at least one section (synthesize when the TOC
    # only has level-1 entries).
    chapters_with_no_sections = {
        c.id for c in chapters
    } - {s.chapter_id for s in sections}
    for chap in chapters:
        if chap.id in chapters_with_no_sections:
            sections.append(
                SectionRecord(
                    id=f"{chap.id}::sec000",
                    chapter_id=chap.id,
                    document_id=document_id,
                    ordinal=0,
                    title=chap.title,
                    start_page_index=chap.start_page_index,
                    end_page_index=chap.start_page_index,
                    tier=tier,
                    source_language=source_language,
                    detection_method="synthetic",
                )
            )

    _patch_end_indices(chapters, num_pages)
    sections_by_chapter: dict[str, list[SectionRecord]] = {}
    for sec in sections:
        sections_by_chapter.setdefault(sec.chapter_id, []).append(sec)
    for chap in chapters:
        chap_sections = sorted(
            sections_by_chapter.get(chap.id, []),
            key=lambda s: s.start_page_index,
        )
        _patch_end_indices_for_sections(chap_sections, chap.end_page_index)

    sections = sorted(
        (s for secs in sections_by_chapter.values() for s in secs),
        key=lambda s: (s.start_page_index, s.ordinal),
    )

    return StructurePlan(
        document_id=document_id,
        chapters=chapters,
        sections=sections,
        detection_method="toc",
    )


def _patch_end_indices(records: list[ChapterRecord], num_pages: int) -> None:
    """Set each record's ``end_page_index`` to ``next.start - 1`` (last = num_pages-1)."""
    for i, rec in enumerate(records):
        if i + 1 < len(records):
            rec.end_page_index = max(
                rec.start_page_index,
                records[i + 1].start_page_index - 1,
            )
        else:
            rec.end_page_index = num_pages - 1


def _patch_end_indices_for_sections(
    records: list[SectionRecord], chapter_end: int
) -> None:
    for i, rec in enumerate(records):
        if i + 1 < len(records):
            rec.end_page_index = max(
                rec.start_page_index,
                records[i + 1].start_page_index - 1,
            )
        else:
            rec.end_page_index = max(rec.start_page_index, chapter_end)


# ---------------------------------------------------------------------------
# EPUB: book.toc + spine href→page_index mapping.
# ---------------------------------------------------------------------------


def _plan_from_epub_toc(
    *,
    document_id: str,
    tier: str,
    page_records: list[PageRecord],
    toc: list[tuple[int, str, str]],
    source_language: str | None = None,
) -> StructurePlan | None:
    """Map ``(level, title, href)`` rows to chapters/sections via spine hrefs."""
    if not toc or not page_records:
        return None

    href_to_page: dict[str, int] = {}
    for rec in page_records:
        href = rec.metadata.get("href") or rec.metadata.get("manifest_href")
        if href and href not in href_to_page:
            href_to_page[str(href)] = rec.page_index

    resolved: list[tuple[int, str, int]] = []
    for level, title, href in toc:
        if not href:
            continue
        page_idx = href_to_page.get(href)
        if page_idx is None:
            base = href.split("#", 1)[0]
            page_idx = href_to_page.get(base)
        if page_idx is None:
            for candidate_href, idx in href_to_page.items():
                if candidate_href.endswith(href) or href.endswith(candidate_href):
                    page_idx = idx
                    break
        if page_idx is None:
            continue
        resolved.append((level, title, page_idx))

    if not resolved:
        return None

    resolved.sort(key=lambda row: row[2])
    return _plan_from_pdf_toc(
        document_id=document_id,
        tier=tier,
        num_pages=len(page_records),
        toc=resolved,
        source_language=source_language,
    )


# ---------------------------------------------------------------------------
# packed-md: directory tree from ``PageRecord.metadata['section_path']``.
# ---------------------------------------------------------------------------


def _plan_from_packed_md(
    *,
    document_id: str,
    tier: str,
    page_records: list[PageRecord],
    source_language: str | None = None,
) -> StructurePlan | None:
    """Top-level dir → CHAPTER, second-level → SECTION."""
    if not page_records:
        return None

    chapters: list[ChapterRecord] = []
    sections: list[SectionRecord] = []
    chap_by_title: dict[str, ChapterRecord] = {}
    sec_by_key: dict[tuple[str, str], SectionRecord] = {}

    has_any_path = False
    for rec in page_records:
        path_segments: list[str] = list(rec.metadata.get("section_path") or [])
        if path_segments:
            has_any_path = True
        chap_title = path_segments[0] if path_segments else SYNTHETIC_CHAPTER_TITLE
        sec_title = (
            path_segments[1]
            if len(path_segments) >= 2
            else chap_title or SYNTHETIC_SECTION_TITLE
        )

        chap = chap_by_title.get(chap_title)
        if chap is None:
            chap_ord = len(chapters)
            chap = ChapterRecord(
                id=f"{document_id}::ch{chap_ord:03d}",
                document_id=document_id,
                ordinal=chap_ord,
                title=chap_title,
                start_page_index=rec.page_index,
                end_page_index=rec.page_index,
                tier=tier,
                source_language=source_language,
                detection_method="packed_md_dir"
                if path_segments
                else "synthetic",
                source_heading=path_segments[0] if path_segments else None,
            )
            chapters.append(chap)
            chap_by_title[chap_title] = chap
        else:
            chap.end_page_index = max(chap.end_page_index, rec.page_index)

        key = (chap_title, sec_title)
        sec = sec_by_key.get(key)
        if sec is None:
            sec_ord = sum(
                1 for k in sec_by_key if k[0] == chap_title
            )
            sec = SectionRecord(
                id=f"{chap.id}::sec{sec_ord:03d}",
                chapter_id=chap.id,
                document_id=document_id,
                ordinal=sec_ord,
                title=sec_title,
                start_page_index=rec.page_index,
                end_page_index=rec.page_index,
                tier=tier,
                source_language=source_language,
                detection_method="packed_md_dir"
                if len(path_segments) >= 2
                else "synthetic",
                source_heading=path_segments[1]
                if len(path_segments) >= 2
                else None,
            )
            sections.append(sec)
            sec_by_key[key] = sec
        else:
            sec.end_page_index = max(sec.end_page_index, rec.page_index)

    if not has_any_path:
        return None

    return StructurePlan(
        document_id=document_id,
        chapters=chapters,
        sections=sections,
        detection_method="packed_md_dir",
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def plan_structure(
    path: str | Path,
    *,
    document_id: str,
    tier: str,
    page_records: list[PageRecord],
    source_language: str | None = None,
) -> StructurePlan:
    """Pick the right extractor for ``path`` and produce a complete plan.

    Always returns a non-empty :class:`StructurePlan` (synthetic fallback
    when TOC / NCX / packed-md detection fail). Caller stamps pages with
    :func:`stamp_pages` and persists via :func:`write_plan`.

    Args:
        path: Source path (used to dispatch the backend).
        document_id: Stable DOCUMENT id.
        tier: ``'primary'`` or ``'secondary'``.
        page_records: All :class:`PageRecord` instances for the DOCUMENT.
            Required even for PDFs because we cap ``end_page_index`` at
            ``len(page_records) - 1``.
        source_language: Hint for ``CHAPTER.sourceLanguage`` /
            ``SECTION.sourceLanguage``. Often unknown at Phase 1 — set to
            ``None`` and let Phase 3 / 6 refine.

    Returns:
        :class:`StructurePlan`.
    """
    num_pages = len(page_records)
    if num_pages < 1:
        return StructurePlan(
            document_id=document_id,
            chapters=[],
            sections=[],
            detection_method="empty",
        )

    backend = detect_reader(path)
    plan: StructurePlan | None = None

    if backend == "pdf":
        from apps.backend.readers import pdf_reader

        try:
            toc_rows = pdf_reader.read_toc(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdf read_toc failed for %s: %s", path, exc)
            toc_rows = []
        plan = _plan_from_pdf_toc(
            document_id=document_id,
            tier=tier,
            num_pages=num_pages,
            toc=toc_rows,
            source_language=source_language,
        )
    elif backend == "epub":
        from apps.backend.readers import epub_reader

        try:
            toc_rows = epub_reader.read_toc(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("epub read_toc failed for %s: %s", path, exc)
            toc_rows = []
        plan = _plan_from_epub_toc(
            document_id=document_id,
            tier=tier,
            page_records=page_records,
            toc=toc_rows,
            source_language=source_language,
        )
    elif backend == "packed_md":
        plan = _plan_from_packed_md(
            document_id=document_id,
            tier=tier,
            page_records=page_records,
            source_language=source_language,
        )

    if plan is None or not plan.chapters or not plan.sections:
        logger.info(
            "structure: falling back to synthetic plan for %s (backend=%s, "
            "num_pages=%d)",
            path,
            backend,
            num_pages,
        )
        return synthetic_plan(
            document_id=document_id,
            tier=tier,
            num_pages=num_pages,
            source_language=source_language,
        )

    return plan


def stamp_pages(page_records: list[PageRecord], plan: StructurePlan) -> None:
    """Mutate each ``PageRecord`` so its ``chapter_id`` + ``section_id`` match the plan.

    Pages whose ``page_index`` falls outside every SECTION's range are
    stamped with the last SECTION (defensive — should not happen given the
    synthetic fallback always covers the full range).
    """
    if not plan.sections:
        return
    sorted_sections = sorted(plan.sections, key=lambda s: s.start_page_index)
    section_by_chapter = {s.chapter_id: s for s in plan.sections}
    last_section = sorted_sections[-1]
    for rec in page_records:
        match: SectionRecord | None = None
        for sec in sorted_sections:
            if sec.start_page_index <= rec.page_index <= sec.end_page_index:
                match = sec
                break
        if match is None:
            match = last_section
        rec.chapter_id = match.chapter_id
        rec.section_id = match.id
        # Defensive: keep the page's section_path metadata informational
        # while making the canonical ids first-class.
        rec.metadata.setdefault("section_title", match.title)
        rec.metadata.setdefault(
            "chapter_title",
            next(
                (c.title for c in plan.chapters if c.id == match.chapter_id),
                None,
            ),
        )
    # Reach the section_by_chapter just to silence a possible unused-var
    # lint when SectionRecord type hints are imported lazily.
    _ = section_by_chapter


# ---------------------------------------------------------------------------
# Neo4j writes (plan §5 v2.1 spine).
# ---------------------------------------------------------------------------


_CHAPTER_UPSERT = """
MATCH (d:DOCUMENT {id: $document_id})
MERGE (c:CHAPTER {id: $chapter_id})
  ON CREATE SET
    c.documentId = $document_id,
    c.ordinal = $ordinal,
    c.title = $title,
    c.startPageIndex = $start_page_index,
    c.endPageIndex = $end_page_index,
    c.tier = $tier,
    c.sourceLanguage = $source_language,
    c.detectionMethod = $detection_method,
    c.sourceHeading = $source_heading,
    c.createdAt = timestamp()
  ON MATCH SET
    c.ordinal = $ordinal,
    c.title = $title,
    c.startPageIndex = $start_page_index,
    c.endPageIndex = $end_page_index,
    c.tier = $tier,
    c.sourceLanguage = coalesce($source_language, c.sourceLanguage),
    c.detectionMethod = $detection_method,
    c.sourceHeading = $source_heading,
    c.updatedAt = timestamp()
MERGE (d)-[:CONSIST_OF]->(c)
RETURN c.id AS id
"""


_SECTION_UPSERT = """
MATCH (c:CHAPTER {id: $chapter_id})
MERGE (s:SECTION {id: $section_id})
  ON CREATE SET
    s.chapterId = $chapter_id,
    s.documentId = $document_id,
    s.ordinal = $ordinal,
    s.title = $title,
    s.startPageIndex = $start_page_index,
    s.endPageIndex = $end_page_index,
    s.tier = $tier,
    s.sourceLanguage = $source_language,
    s.detectionMethod = $detection_method,
    s.sourceHeading = $source_heading,
    s.createdAt = timestamp()
  ON MATCH SET
    s.ordinal = $ordinal,
    s.title = $title,
    s.startPageIndex = $start_page_index,
    s.endPageIndex = $end_page_index,
    s.tier = $tier,
    s.sourceLanguage = coalesce($source_language, s.sourceLanguage),
    s.detectionMethod = $detection_method,
    s.sourceHeading = $source_heading,
    s.updatedAt = timestamp()
MERGE (c)-[:INCLUDE]->(s)
RETURN s.id AS id
"""


def write_plan(driver: Driver, plan: StructurePlan) -> dict[str, int]:
    """Upsert every CHAPTER + SECTION in ``plan`` and wire ``CONSIST_OF`` / ``INCLUDE``.

    Idempotent: re-running on the same plan is safe (`MERGE` + `ON MATCH SET`).
    Caller is responsible for upserting the DOCUMENT first.

    Returns:
        ``{"chapters": <count_written>, "sections": <count_written>}``.
    """
    # NB: ``to_dict`` returns camelCase (matches the Neo4j property convention)
    # but the Cypher placeholders below are snake_case — bind each parameter
    # explicitly so the two naming worlds don't collide.
    #
    # Both upserts open with ``MATCH (parent {...})``; if the parent
    # (DOCUMENT for chapters, CHAPTER for sections) is missing, the MERGE
    # is silently dropped. We count actual `RETURN id` rows so a silent
    # no-op surfaces as a missing-parent error instead of a false success.
    chapters_written = 0
    sections_written = 0
    missing_parents: list[str] = []
    with driver.session() as session:
        for chap in plan.chapters:
            cp = chap.to_dict()
            row = session.run(
                _CHAPTER_UPSERT,
                chapter_id=cp["id"],
                document_id=cp["documentId"],
                ordinal=cp["ordinal"],
                title=cp["title"],
                start_page_index=cp["startPageIndex"],
                end_page_index=cp["endPageIndex"],
                tier=cp["tier"],
                source_language=cp["sourceLanguage"],
                detection_method=cp["detectionMethod"],
                source_heading=cp["sourceHeading"],
            ).single()
            if row is not None:
                chapters_written += 1
            else:
                missing_parents.append(f"CHAPTER {cp['id']}: parent DOCUMENT {cp['documentId']} not in graph")
        for sec in plan.sections:
            sp = sec.to_dict()
            row = session.run(
                _SECTION_UPSERT,
                section_id=sp["id"],
                chapter_id=sp["chapterId"],
                document_id=sp["documentId"],
                ordinal=sp["ordinal"],
                title=sp["title"],
                start_page_index=sp["startPageIndex"],
                end_page_index=sp["endPageIndex"],
                tier=sp["tier"],
                source_language=sp["sourceLanguage"],
                detection_method=sp["detectionMethod"],
                source_heading=sp["sourceHeading"],
            ).single()
            if row is not None:
                sections_written += 1
            else:
                missing_parents.append(f"SECTION {sp['id']}: parent CHAPTER {sp['chapterId']} not in graph")
    if missing_parents:
        raise RuntimeError(
            f"structure write incomplete ({len(missing_parents)} silent no-ops): "
            + "; ".join(missing_parents[:3])
            + (f" (+{len(missing_parents) - 3} more)" if len(missing_parents) > 3 else "")
        )
    return {
        "chapters": chapters_written,
        "sections": sections_written,
    }


# ---------------------------------------------------------------------------
# Stage 3.5 LLM fallback — placeholder.
# ---------------------------------------------------------------------------


def llm_extract_structure(
    *,
    document_id: str,
    tier: str,
    page_records: list[PageRecord],
    source_language: str | None = None,
    silra_client: Any | None = None,
    max_pages_sampled: int = 20,
) -> StructurePlan:
    """Phase-3.5 LLM fallback: deepseek-chat over fused-OCR snippets (plan §6 Stage 3.5).

    **NOT IMPLEMENTED in Phase 1.** The Phase-3 fusion stage (which writes
    ``PAGE.text_fused``) is the prerequisite; this function lives here so
    the contract is visible from the structure module the orchestrator
    already imports.

    Implementation will:

    1. Sample up to ``max_pages_sampled`` pages evenly across the document
       and concatenate their ``PAGE.text_fused`` snippets.
    2. Issue a single :mod:`apps.backend.llm.silra` ``deepseek-chat`` call
       with a classical-Chinese system prompt (per ``AGENTS.md`` §5) asking
       for ``[{title, level, startPageIndex, endPageIndex}, ...]``.
    3. Materialize the parsed JSON via :func:`_plan_from_pdf_toc` (the same
       page-range patching logic), with ``detection_method='llm'``.
    4. Fall back to :func:`synthetic_plan` on parse error or empty output.

    Raises:
        NotImplementedError: until Phase 3 lands.
    """
    raise NotImplementedError(
        "Stage 3.5 LLM structure fallback requires Phase-3 fusion output "
        "(PAGE.text_fused). Will be wired in when Phase 3 lands."
    )
