"""Bulk corpus loader (plan §6 Stage 1 — Phase 1, notebook 01c).

Walks ``raw/Primary/`` + ``raw/Secondary/`` and ingests every PDF / EPUB /
``.packed/`` tree through :func:`apps.backend.pipeline.ingest.ingest_path`,
then (optionally) runs :func:`apps.backend.pipeline.lang_detect.detect_pages`
in a single batched pass after the ingest loop finishes.

The loader is idempotent + restartable:

- Discovery is deterministic (sorted, path-based ``document_id``).
- ``skip_existing=True`` (the default) consults Neo4j and skips any
  ``DOCUMENT.id`` already present, so partial runs can be resumed by
  re-executing the cell.
- ``dry_run=True`` prints the planned manifest without touching MinIO or
  Neo4j — recommended before any expensive bulk write.

LLM enrichment is **on by default** here (per AGENTS.md §11 entry dated
2026-05-17: "Bulk ingest (01c) defaults to LLM-on"); the smoke notebook
01_ingestion keeps it off to control Silra spend.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neo4j import Driver

from apps.backend.pipeline.ingest import (
    IngestReport,
    _document_id_from_path,
    ingest_path,
    upsert_topic,
)
from apps.backend.readers import detect_reader, resolve_tier

logger = logging.getLogger(__name__)


# Suffixes the bulk loader will route to a reader. The set is closed: any
# file we don't recognise gets reported in ``BulkLoadReport.skipped`` with
# the reason so the user can decide whether to add a backend.
_TARGET_FILE_SUFFIXES: frozenset[str] = frozenset({".pdf", ".epub"})

# Suffixes we know about but cannot ingest yet. ``.djvu`` is a real corpus
# format (one paper in ``raw/Secondary/specific_门荫/...``) but lacks a
# reader; ``.md`` is an Obsidian annotation file that sits next to a PDF
# and is *not* a corpus unit.
_KNOWN_NON_INGESTABLE: frozenset[str] = frozenset({".djvu", ".md"})


@dataclass
class CorpusEntry:
    """One ingestable corpus unit (file or ``.packed/`` directory)."""

    path: Path
    relative_path: str
    tier: str  # "primary" | "secondary"
    backend: str  # "pdf" | "epub" | "packed_md"
    document_id: str
    size_bytes: int = 0  # file size or sum of .packed/ tree contents
    page_count_hint: int | None = None  # populated when count_pages=True

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "tier": self.tier,
            "backend": self.backend,
            "document_id": self.document_id,
            "size_bytes": self.size_bytes,
            "page_count_hint": self.page_count_hint,
        }


@dataclass
class SkippedEntry:
    """A file under ``raw/`` that the loader chose not to ingest."""

    path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class DocResult:
    """Outcome of ingesting one :class:`CorpusEntry`."""

    document_id: str
    relative_path: str
    tier: str
    backend: str
    status: str  # "ingested" | "skipped" | "failed" | "would-ingest"
    duration_seconds: float = 0.0
    pages_total: int = 0
    pages_native: int = 0
    pages_ocr: int = 0
    pages_uploaded_to_minio: int = 0
    chapters_written: int = 0
    sections_written: int = 0
    structure_detection_method: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BulkLoadReport:
    """Aggregate run report (suitable for ``json.dump``)."""

    started_at: str = ""
    finished_at: str = ""
    dry_run: bool = False
    skip_existing: bool = True
    run_lang_detect: bool = True
    enrich_metadata_with_llm: bool = True
    discovered: int = 0
    ingested: int = 0
    skipped_existing: int = 0
    failed: int = 0
    would_ingest: int = 0
    pages_total: int = 0
    pages_uploaded_to_minio: int = 0
    lang_detect_pages_processed: int = 0
    lang_detect_pages_written: int = 0
    duration_seconds: float = 0.0
    skipped_files: list[SkippedEntry] = field(default_factory=list)
    documents: list[DocResult] = field(default_factory=list)
    by_tier: dict[str, int] = field(default_factory=dict)
    by_backend: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "skipped_files": [s.to_dict() for s in self.skipped_files],
            "documents": [d.to_dict() for d in self.documents],
        }


# ---------------------------------------------------------------------------
# Discovery.
# ---------------------------------------------------------------------------


def _under_packed_tree(path: Path) -> bool:
    """Return ``True`` iff any parent directory ends with ``.packed``.

    Files inside ``foo.packed/`` are internal corpus shards (markdown +
    image bundles); only the ``foo.packed/`` directory itself is an
    ingestable corpus unit.
    """
    return any(parent.name.endswith(".packed") for parent in path.parents)


def _dir_size(path: Path) -> int:
    """Sum bytes of every regular file under ``path`` (used for .packed dirs)."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue
    except OSError:
        pass
    return total


def _count_pages(path: Path, backend: str) -> int | None:
    """Best-effort page count for the discovery preview.

    Imports the reader lazily to keep the bulk-loader module dependency-
    free at import time. Returns ``None`` on any failure (the discovery
    preview gracefully degrades to "page_count_hint=None").
    """
    try:
        if backend == "pdf":
            import fitz  # type: ignore[import-untyped]

            with fitz.open(path) as doc:
                return doc.page_count
        if backend == "epub":
            from apps.backend.readers import epub_reader

            summary = epub_reader.quick_summary(path, sample_pages=0)
            return int(summary.get("page_count", 0)) or None
        if backend == "packed_md":
            return sum(1 for _ in path.rglob("*.md"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("page count failed for %s: %s", path, exc)
    return None


def discover_corpus(
    root: str | Path,
    *,
    include_primary: bool = True,
    include_secondary: bool = True,
    count_pages: bool = False,
    repo_root: str | Path | None = None,
) -> tuple[list[CorpusEntry], list[SkippedEntry]]:
    """Walk ``raw/`` and return ``(ingestable_entries, skipped)``.

    Args:
        root: The ``raw/`` directory (absolute or relative).
        include_primary: Walk ``raw/Primary/``.
        include_secondary: Walk ``raw/Secondary/``.
        count_pages: If ``True``, run a cheap per-file page-count probe.
            Off by default — for ~70 corpus files the probe adds ~5-10s
            (mostly EPUB unzipping).
        repo_root: Used to compute ``relative_path``; defaults to ``root.parent``.

    Returns:
        Two lists:

        - ``CorpusEntry`` instances sorted by ``(tier, relative_path)``.
        - ``SkippedEntry`` instances with a human-readable reason.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"corpus root not found: {root}")
    repo_root_p = Path(repo_root) if repo_root else root.parent

    entries: list[CorpusEntry] = []
    skipped: list[SkippedEntry] = []
    seen_packed_dirs: set[Path] = set()

    targets: list[tuple[str, Path]] = []
    if include_primary:
        primary_dir = root / "Primary"
        if primary_dir.exists():
            targets.append(("primary", primary_dir))
    if include_secondary:
        secondary_dir = root / "Secondary"
        if secondary_dir.exists():
            targets.append(("secondary", secondary_dir))

    for _tier_hint, base in targets:
        # First pass: ``.packed/`` directories. They show up as files
        # under ``rglob`` so we collect them first and use the set to
        # skip the children.
        for entry in base.rglob("*.packed"):
            if not entry.is_dir():
                continue
            seen_packed_dirs.add(entry.resolve())
            try:
                ent = _make_entry(entry, repo_root_p, count_pages=count_pages)
                entries.append(ent)
            except ValueError as exc:
                skipped.append(SkippedEntry(path=str(entry), reason=str(exc)))

        # Second pass: regular files. Skip anything under a known .packed/.
        for entry in base.rglob("*"):
            if not entry.is_file():
                continue
            resolved_parents = {p.resolve() for p in entry.parents}
            if resolved_parents & seen_packed_dirs:
                continue
            if _under_packed_tree(entry):
                continue
            suffix = entry.suffix.lower()
            if suffix in _TARGET_FILE_SUFFIXES:
                try:
                    ent = _make_entry(entry, repo_root_p, count_pages=count_pages)
                    entries.append(ent)
                except ValueError as exc:
                    skipped.append(SkippedEntry(path=str(entry), reason=str(exc)))
                continue
            if suffix in _KNOWN_NON_INGESTABLE:
                reason = {
                    ".djvu": ".djvu has no reader yet (deferred to Phase 3 OCR)",
                    ".md": "markdown annotation file, not a corpus unit",
                }.get(suffix, f"unsupported suffix: {suffix}")
                skipped.append(SkippedEntry(path=str(entry), reason=reason))
                continue
            # Bookkeeping files (.DS_Store, README.md if we missed it, etc.)
            if entry.name.startswith(".") or entry.name in {"README.md"}:
                continue
            skipped.append(
                SkippedEntry(
                    path=str(entry),
                    reason=f"unknown corpus format: {suffix or '(no suffix)'}",
                )
            )

    entries.sort(key=lambda e: (e.tier, e.relative_path))
    return entries, skipped


def _make_entry(path: Path, repo_root: Path, *, count_pages: bool) -> CorpusEntry:
    """Build a :class:`CorpusEntry`; raises ``ValueError`` if no reader applies."""
    backend = detect_reader(path)  # raises ValueError for unknown formats
    tier = resolve_tier(path)
    try:
        rel = str(path.relative_to(repo_root))
    except ValueError:
        rel = str(path)
    document_id = _document_id_from_path(path)
    size = path.stat().st_size if path.is_file() else _dir_size(path)
    hint = _count_pages(path, backend) if count_pages else None
    return CorpusEntry(
        path=path,
        relative_path=rel,
        tier=tier,
        backend=backend,
        document_id=document_id,
        size_bytes=size,
        page_count_hint=hint,
    )


# ---------------------------------------------------------------------------
# Already-ingested lookup.
# ---------------------------------------------------------------------------


_IS_INGESTED = """
MATCH (d:DOCUMENT {id: $document_id})
OPTIONAL MATCH (d)-[:CONSIST_OF]->(:CHAPTER)-[:INCLUDE]->(:SECTION)-[:INCLUDE]->(p:PAGE)
RETURN d.id AS id, d.title AS title, d.tier AS tier,
       d.sourcePath AS sourcePath, count(DISTINCT p) AS page_count
"""


def is_ingested(driver: Driver, document_id: str) -> dict[str, Any] | None:
    """Return existing DOCUMENT props + page count, or ``None`` if absent.

    The page count walks the v2.1 spine ``DOCUMENT -> CHAPTER -> SECTION
    -> PAGE`` so a partial ingest (DOCUMENT present, no PAGEs yet) is
    distinguishable from a clean one.
    """
    with driver.session() as session:
        rec = session.run(_IS_INGESTED, document_id=document_id).single()
    if rec is None or rec.get("id") is None:
        return None
    return {
        "id": rec["id"],
        "title": rec.get("title"),
        "tier": rec.get("tier"),
        "sourcePath": rec.get("sourcePath"),
        "page_count": rec.get("page_count") or 0,
    }


# ---------------------------------------------------------------------------
# Bulk ingest.
# ---------------------------------------------------------------------------


def bulk_ingest(
    driver: Driver,
    entries: Iterable[CorpusEntry],
    *,
    topic_id: str = "tang-corpus",
    minio_client: Any = None,
    bucket: str = "ancient-pages",
    repo_root: Path | None = None,
    dry_run: bool = False,
    skip_existing: bool = True,
    enrich_metadata_with_llm: bool = True,
    max_pages_per_doc: int | None = None,
    max_docs: int | None = None,
    run_lang_detect: bool = True,
    on_doc_start: Any = None,  # callable(idx, total, entry) -> None
    on_doc_done: Any = None,  # callable(idx, total, doc_result) -> None
) -> BulkLoadReport:
    """Ingest every entry and (optionally) run Phase-1b lang detection.

    Args:
        driver: Open Neo4j driver.
        entries: Iterable of :class:`CorpusEntry` (from :func:`discover_corpus`).
        topic_id: TOPIC node to attach every DOCUMENT to.
        minio_client: Required if any entry has OCR-bound pages (every
            primary EPUB does — they carry scanned image bundles).
        bucket: MinIO bucket for page images.
        repo_root: Forwarded to ``ingest_path`` so ``sourcePath`` is
            relative to the repo.
        dry_run: If ``True``, log the manifest + ``status='would-ingest'``
            per entry; no Neo4j or MinIO writes.
        skip_existing: If ``True``, query Neo4j for each ``document_id``
            and skip when already present. Set to ``False`` to force a
            re-ingest (idempotent because ``ingest_path`` uses MERGE).
        enrich_metadata_with_llm: Forwarded to ``ingest_path``. The plan's
            convention (AGENTS.md §11) is ``True`` for bulk ingest.
        max_pages_per_doc: Optional per-doc page cap (passed through as
            ``ingest_path(max_pages=...)``). ``None`` = full ingest.
        max_docs: Optional cap on the number of entries to process —
            applied AFTER ``skip_existing`` filtering so smoke runs
            advance through the corpus instead of repeatedly hitting
            already-ingested docs.
        run_lang_detect: Run :func:`detect_pages` once at the end (with
            ``recompute_existing=False`` so already-classified pages are
            untouched).
        on_doc_start / on_doc_done: Progress hooks for the notebook to
            stream live status.

    Returns:
        :class:`BulkLoadReport`.
    """
    from datetime import datetime, timezone

    report = BulkLoadReport(
        dry_run=dry_run,
        skip_existing=skip_existing,
        run_lang_detect=run_lang_detect,
        enrich_metadata_with_llm=enrich_metadata_with_llm,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    started = time.monotonic()

    plan: list[CorpusEntry] = list(entries)
    report.discovered = len(plan)
    for e in plan:
        report.by_tier[e.tier] = report.by_tier.get(e.tier, 0) + 1
        report.by_backend[e.backend] = report.by_backend.get(e.backend, 0) + 1

    # Filter out already-ingested docs first so ``max_docs`` cleanly
    # advances through fresh entries on partial runs.
    to_process: list[CorpusEntry] = []
    for entry in plan:
        if skip_existing and not dry_run:
            existing = is_ingested(driver, entry.document_id)
            if existing is not None:
                report.skipped_existing += 1
                report.documents.append(
                    DocResult(
                        document_id=entry.document_id,
                        relative_path=entry.relative_path,
                        tier=entry.tier,
                        backend=entry.backend,
                        status="skipped",
                        pages_total=existing.get("page_count") or 0,
                    )
                )
                continue
        to_process.append(entry)

    if max_docs is not None:
        to_process = to_process[:max_docs]

    # Ensure the TOPIC (and its owning USER) exists once before the
    # ingest loop. ``_DOC_UPSERT`` opens with ``MATCH (t:TOPIC {...})``
    # and silently no-ops the whole DOCUMENT MERGE when the TOPIC is
    # absent — so on a fresh database every ingest_path call would
    # write nothing. ``upsert_topic`` is idempotent (MERGE-based).
    if to_process and not dry_run:
        upsert_topic(driver, topic_id=topic_id)

    total = len(to_process)
    for idx, entry in enumerate(to_process, start=1):
        if on_doc_start is not None:
            try:
                on_doc_start(idx, total, entry)
            except Exception:  # noqa: BLE001
                logger.debug("on_doc_start hook raised; ignoring", exc_info=True)

        if dry_run:
            result = DocResult(
                document_id=entry.document_id,
                relative_path=entry.relative_path,
                tier=entry.tier,
                backend=entry.backend,
                status="would-ingest",
            )
            report.documents.append(result)
            report.would_ingest += 1
            if on_doc_done is not None:
                try:
                    on_doc_done(idx, total, result)
                except Exception:  # noqa: BLE001
                    logger.debug("on_doc_done hook raised; ignoring", exc_info=True)
            continue

        try:
            ir: IngestReport = ingest_path(
                entry.path,
                driver=driver,
                topic_id=topic_id,
                tier=entry.tier,
                minio_client=minio_client,
                bucket=bucket,
                repo_root=repo_root,
                max_pages=max_pages_per_doc,
                enrich_metadata_with_llm=enrich_metadata_with_llm,
            )
            # An IngestReport with non-empty .errors means the v2.1 spine
            # is incomplete (e.g. structure-write failed): pages may exist
            # but chapters/sections do not. Mark as "partial" so the
            # caller can decide whether to retry; don't silently count it
            # as a clean success.
            spine_ok = (
                ir.chapters_written > 0
                and ir.sections_written > 0
                and not ir.errors
            )
            result = DocResult(
                document_id=ir.document_id,
                relative_path=ir.source_path,
                tier=ir.tier,
                backend=ir.backend,
                status="ingested" if spine_ok else "partial",
                duration_seconds=ir.duration_seconds,
                pages_total=ir.pages_total,
                pages_native=ir.pages_native,
                pages_ocr=ir.pages_ocr,
                pages_uploaded_to_minio=ir.pages_uploaded_to_minio,
                chapters_written=ir.chapters_written,
                sections_written=ir.sections_written,
                structure_detection_method=ir.structure_detection_method,
                error=" | ".join(ir.errors) if ir.errors else None,
            )
            if spine_ok:
                report.ingested += 1
            else:
                report.failed += 1
            report.pages_total += ir.pages_total
            report.pages_uploaded_to_minio += ir.pages_uploaded_to_minio
        except Exception as exc:  # noqa: BLE001
            logger.exception("ingest failed for %s", entry.relative_path)
            result = DocResult(
                document_id=entry.document_id,
                relative_path=entry.relative_path,
                tier=entry.tier,
                backend=entry.backend,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            report.failed += 1

        report.documents.append(result)
        if on_doc_done is not None:
            try:
                on_doc_done(idx, total, result)
            except Exception:  # noqa: BLE001
                logger.debug("on_doc_done hook raised; ignoring", exc_info=True)

    # Phase 1b sweep: classify every native page that was created above.
    # Defaults to ``recompute_existing=False`` so reruns are cheap.
    if run_lang_detect and not dry_run and report.ingested > 0:
        try:
            from apps.backend.pipeline.lang_detect import detect_pages

            ld = detect_pages(driver, recompute_existing=False)
            report.lang_detect_pages_processed = ld.pages_processed
            report.lang_detect_pages_written = ld.pages_written
            if ld.errors:
                logger.warning("lang_detect reported errors: %s", ld.errors)
        except Exception as exc:  # noqa: BLE001
            logger.exception("post-ingest lang_detect failed")
            report.documents.append(
                DocResult(
                    document_id="<lang_detect>",
                    relative_path="<lang_detect>",
                    tier="-",
                    backend="-",
                    status="failed",
                    error=f"lang_detect: {type(exc).__name__}: {exc}",
                )
            )

    report.finished_at = datetime.now(timezone.utc).isoformat()
    report.duration_seconds = round(time.monotonic() - started, 3)
    return report
