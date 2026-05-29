"""Phase-1 ingest orchestrator: path -> DOCUMENT/PAGE nodes + MinIO image objects.

Single deterministic pipeline (no Celery yet — Phase 3 wraps it):

1. Resolve tier from path (``raw/Primary`` or ``raw/Secondary``).
2. Extract edition metadata (regex always; LLM optional).
3. Pick the right reader and walk pages.
4. For OCR-bound pages: upload bytes to MinIO under
   ``MINIO_BUCKET_PAGES/<document_id>/page_<n>.png``.
5. Upsert (TOPIC) -> DOCUMENT -> (PAGE) nodes via parameterized Cypher.
6. Return a structured ``IngestReport`` for the notebook / API to render.

The orchestrator is sync + idempotent so re-running it on the same path is
safe (every Cypher write uses ``MERGE`` with set-on-create semantics).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neo4j import Driver

from apps.backend.pipeline.metadata import (
    EditionMetadata,
    enrich_from_opf,
    extract_from_filename,
    llm_enrich,
)
from apps.backend.pipeline.structure import (
    StructurePlan,
    plan_structure,
    stamp_pages,
    write_plan,
)
from apps.backend.readers import (
    PageRecord,
    detect_reader,
    iter_pages,
    resolve_tier,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    document_id: str
    source_path: str
    tier: str
    backend: str
    topic_id: str
    pages_total: int
    pages_native: int
    pages_ocr: int
    pages_uploaded_to_minio: int
    metadata: dict[str, Any]
    duration_seconds: float
    errors: list[str] = field(default_factory=list)
    sample_page_ids: list[str] = field(default_factory=list)
    chapters_written: int = 0
    sections_written: int = 0
    structure_detection_method: str = "synthetic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# TOPIC + DOCUMENT upserts.
# ---------------------------------------------------------------------------


DEFAULT_TOPIC_ID = "tang-corpus"
DEFAULT_TOPIC_NAME = "Tang corpus"
DEFAULT_TOPIC_DESCRIPTION = (
    "Tang-era primary sources and modern scholarly secondaries (v1 corpus)."
)


def upsert_topic(
    driver: Driver,
    *,
    topic_id: str = DEFAULT_TOPIC_ID,
    name: str = DEFAULT_TOPIC_NAME,
    description: str = DEFAULT_TOPIC_DESCRIPTION,
    user_id: str = "system",
    user_name: str = "system",
) -> dict[str, Any]:
    """Idempotently create USER + TOPIC, returning their ids.

    Args:
        driver: Open Neo4j driver.
        topic_id: TOPIC key.
        name, description: Display fields.
        user_id, user_name: Owning user (defaults to ``system`` for bulk ingest).

    Returns:
        Dict ``{"user_id", "topic_id"}``.
    """
    cypher = """
    MERGE (u:USER {id: $user_id})
      ON CREATE SET u.name = $user_name, u.createdAt = timestamp()
    MERGE (t:TOPIC {id: $topic_id})
      ON CREATE SET t.name = $name, t.description = $description, t.createdAt = timestamp()
      ON MATCH  SET t.name = $name, t.description = $description
    MERGE (u)-[:CREATE]->(t)
    RETURN u.id AS user_id, t.id AS topic_id
    """
    with driver.session() as session:
        record = session.run(
            cypher,
            user_id=user_id,
            user_name=user_name,
            topic_id=topic_id,
            name=name,
            description=description,
        ).single()
        return dict(record) if record else {"user_id": user_id, "topic_id": topic_id}


# ---------------------------------------------------------------------------
# DOCUMENT + PAGE writes.
# ---------------------------------------------------------------------------


def _document_id_from_path(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem.replace(' ', '_').replace('.packed', '')[:60]}__{digest}"


_DOC_UPSERT = """
MATCH (t:TOPIC {id: $topic_id})
MERGE (d:DOCUMENT {id: $document_id})
  ON CREATE SET
    d.tier = $tier,
    d.title = $title,
    d.sourcePath = $source_path,
    d.edition = $edition,
    d.publisher = $publisher,
    d.publicationYear = $publication_year,
    d.publicationPeriod = $publication_period,
    d.editorialLayers = $editorial_layers_json,
    d.primaryAuthor = $primary_author,
    d.secondaryAuthor = $secondary_author,
    d.metadataConfidence = $confidence,
    d.metadataExtractedVia = $extracted_via,
    d.backend = $backend,
    d.createdAt = timestamp()
  ON MATCH SET
    d.title = coalesce($title, d.title),
    d.edition = coalesce($edition, d.edition),
    d.publisher = coalesce($publisher, d.publisher),
    d.publicationYear = coalesce($publication_year, d.publicationYear),
    d.publicationPeriod = coalesce($publication_period, d.publicationPeriod),
    d.editorialLayers = coalesce($editorial_layers_json, d.editorialLayers),
    d.primaryAuthor = coalesce($primary_author, d.primaryAuthor),
    d.secondaryAuthor = coalesce($secondary_author, d.secondaryAuthor),
    d.metadataConfidence = $confidence,
    d.metadataExtractedVia = $extracted_via,
    d.backend = $backend,
    d.updatedAt = timestamp()
MERGE (t)-[:CONTAIN]->(d)
RETURN d.id AS id
"""


_PAGE_UPSERT = """
MATCH (s:SECTION {id: $section_id})
MERGE (p:PAGE {id: $page_id})
  ON CREATE SET
    p.documentId = $document_id,
    p.chapterId = $chapter_id,
    p.sectionId = $section_id,
    p.docPageIndex = $page_index,
    p.mode = $mode,
    p.text = $text,
    p.imageUri = $image_uri,
    p.role = $role,
    p.tier = $tier,
    p.language = $language_hint,
    p.charCount = $char_count,
    p.metadataJson = $metadata_json,
    p.createdAt = timestamp()
  ON MATCH SET
    p.chapterId = $chapter_id,
    p.sectionId = $section_id,
    p.mode = $mode,
    p.text = coalesce($text, p.text),
    p.imageUri = coalesce($image_uri, p.imageUri),
    p.role = $role,
    p.tier = $tier,
    p.language = coalesce($language_hint, p.language),
    p.charCount = $char_count,
    p.metadataJson = $metadata_json,
    p.updatedAt = timestamp()
MERGE (s)-[:INCLUDE]->(p)
RETURN p.id AS id
"""


_PAGE_NEXT_LINK = """
MATCH (a:PAGE {id: $prev_id})
MATCH (b:PAGE {id: $curr_id})
MERGE (a)-[:NEXT]->(b)
"""


# ---------------------------------------------------------------------------
# MinIO upload.
# ---------------------------------------------------------------------------


def _upload_image(
    minio_client,
    bucket: str,
    key: str,
    data: bytes,
    *,
    content_type: str = "image/png",
) -> None:
    import io

    from minio.error import S3Error  # noqa: F401  (raised through)

    minio_client.put_object(
        bucket_name=bucket,
        object_name=key,
        data=io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def ingest_path(
    path: str | Path,
    *,
    driver: Driver,
    topic_id: str = DEFAULT_TOPIC_ID,
    tier: str | None = None,
    document_id: str | None = None,
    minio_client=None,
    bucket: str = "ancient-pages",
    repo_root: Path | None = None,
    max_pages: int | None = None,
    enrich_metadata_with_llm: bool = False,
    metadata_snippet_chars: int = 800,
    reader_kwargs: dict[str, Any] | None = None,
) -> IngestReport:
    """Ingest one source (EPUB / PDF / packed-md dir) into Neo4j + MinIO.

    Args:
        path: Source path (must exist under ``raw/`` for tier auto-detection).
        driver: Open Neo4j driver.
        topic_id: TOPIC to attach the document to.
        tier: Override tier (else resolved from path).
        document_id: Override doc slug (else hashed from path).
        minio_client: MinIO client; required if any page is OCR-bound.
            Pass ``None`` for native-only sources.
        bucket: MinIO bucket name.
        repo_root: Optional repo root for relative ``sourcePath``.
        max_pages: Stop after N pages (for notebook smoke tests).
        enrich_metadata_with_llm: If True, run a single deepseek-chat call
            to enrich the regex extraction. Default False.
        metadata_snippet_chars: When enrich_metadata_with_llm=True, take the
            first N chars of the first native-text page as the LLM snippet.
        reader_kwargs: Extra kwargs forwarded to the reader (e.g.
            ``target_dpi`` for PDFs).

    Returns:
        :class:`IngestReport`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if document_id is None:
        document_id = _document_id_from_path(p)
    if tier is None:
        tier = resolve_tier(p)
    backend = detect_reader(p)
    reader_kwargs = reader_kwargs or {}

    started = time.monotonic()

    # 1. Edition metadata (cheap regex first; zero-cost OPF supplement for
    #    EPUBs whose filename was stripped to the bare title by the corpus
    #    renaming convention; optional LLM enrichment after we have the
    #    first-page snippet).
    base_metadata = extract_from_filename(p.name)
    if backend == "epub":
        try:
            from apps.backend.readers.epub_reader import read_opf_hints

            opf_hints = read_opf_hints(p)
            base_metadata = enrich_from_opf(base_metadata, opf_hints)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "OPF hint extraction failed for %s: %s", p.name, exc
            )

    # 2. Walk pages.
    page_records: list[PageRecord] = list(
        iter_pages(
            p,
            max_pages=max_pages,
            document_id=document_id,
            repo_root=repo_root,
            **reader_kwargs,
        )
    )

    # Now we have the first native page; refine with LLM if requested.
    metadata: EditionMetadata = base_metadata
    if enrich_metadata_with_llm:
        snippet = ""
        for rec in page_records:
            if rec.mode == "native_text" and rec.text:
                snippet = rec.text[:metadata_snippet_chars]
                break
        metadata = llm_enrich(p.name, snippet=snippet, base=base_metadata)

    # Coerce metadata for Neo4j (lists of dicts -> JSON string; otherwise the
    # driver can't write nested maps as a property). DOCUMENT.editorialLayers
    # is read back as JSON.
    import json as _json

    editorial_layers_json = _json.dumps(
        [el.to_dict() for el in metadata.editorial_layers],
        ensure_ascii=False,
    )

    errors: list[str] = []
    page_ids_written: list[str] = []
    pages_native = sum(1 for r in page_records if r.mode == "native_text")
    pages_ocr = sum(1 for r in page_records if r.mode == "ocr")
    pages_uploaded = 0

    # 3. Upsert DOCUMENT.
    with driver.session() as session:
        session.run(
            _DOC_UPSERT,
            topic_id=topic_id,
            document_id=document_id,
            tier=tier,
            title=metadata.title,
            source_path=str(p.relative_to(repo_root)) if repo_root else str(p),
            edition=metadata.edition,
            publisher=metadata.publisher,
            publication_year=metadata.publication_year,
            publication_period=metadata.publication_period,
            editorial_layers_json=editorial_layers_json,
            primary_author=metadata.primary_author,
            secondary_author=metadata.secondary_author,
            confidence=metadata.confidence,
            extracted_via=metadata.extracted_via,
            backend=backend,
        ).consume()

    # 3b. Plan + write CHAPTER/SECTION spine (plan §6 Stage 1, v2.1).
    #     Synthetic fallback guarantees every page lands inside a SECTION,
    #     so the PAGE upsert below can MATCH (s:SECTION {id: ...}) safely.
    plan: StructurePlan = plan_structure(
        p,
        document_id=document_id,
        tier=tier,
        page_records=page_records,
    )
    try:
        write_counts = write_plan(driver, plan)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"structure write failed: {exc}")
        write_counts = {"chapters": 0, "sections": 0}
        # CHAPTER / SECTION are part of the v2.1 spine contract; if the
        # write fails the document is half-ingested. Log loud so the
        # bulk loader (which captures IngestReport.errors into a string)
        # doesn't silently mask it.
        logger.error(
            "structure write failed for document_id=%s (planned chapters=%d, sections=%d): %s",
            document_id,
            len(plan.chapters),
            len(plan.sections),
            exc,
        )
    stamp_pages(page_records, plan)

    # 4. Upload OCR images, then upsert Pages.
    prev_page_id: str | None = None
    for rec in page_records:
        page_id = f"{document_id}::p{rec.page_index:05d}"
        if rec.mode == "ocr" and rec.image_bytes:
            if minio_client is None:
                errors.append(f"page {rec.page_index}: minio_client missing for ocr page")
            else:
                key = rec.image_uri or f"{document_id}/page_{rec.page_index:05d}.png"
                try:
                    _upload_image(minio_client, bucket, key, rec.image_bytes)
                    rec.image_uri = key
                    pages_uploaded += 1
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        f"page {rec.page_index}: minio upload failed "
                        f"({type(exc).__name__}): {exc}"
                    )

        if rec.section_id is None or rec.chapter_id is None:
            errors.append(
                f"page {rec.page_index}: structure planner left "
                "chapter_id/section_id unset (skipping page upsert)"
            )
            continue

        with driver.session() as session:
            try:
                session.run(
                    _PAGE_UPSERT,
                    document_id=document_id,
                    chapter_id=rec.chapter_id,
                    section_id=rec.section_id,
                    page_id=page_id,
                    page_index=rec.page_index,
                    mode=rec.mode,
                    text=rec.text,
                    image_uri=rec.image_uri,
                    role=rec.role,
                    tier=rec.tier,
                    language_hint=rec.language_hint,
                    char_count=rec.char_count,
                    metadata_json=_json.dumps(rec.metadata, ensure_ascii=False, default=str),
                ).consume()
                page_ids_written.append(page_id)
                if prev_page_id:
                    session.run(
                        _PAGE_NEXT_LINK,
                        prev_id=prev_page_id,
                        curr_id=page_id,
                    ).consume()
                prev_page_id = page_id
            except Exception as exc:  # noqa: BLE001
                errors.append(f"page {rec.page_index}: cypher write failed: {exc}")

    duration = time.monotonic() - started

    return IngestReport(
        document_id=document_id,
        source_path=str(p.relative_to(repo_root)) if repo_root else str(p),
        tier=tier,
        backend=backend,
        topic_id=topic_id,
        pages_total=len(page_records),
        pages_native=pages_native,
        pages_ocr=pages_ocr,
        pages_uploaded_to_minio=pages_uploaded,
        metadata=metadata.to_dict(),
        duration_seconds=round(duration, 3),
        errors=errors,
        sample_page_ids=page_ids_written[:5],
        chapters_written=write_counts.get("chapters", 0),
        sections_written=write_counts.get("sections", 0),
        structure_detection_method=plan.detection_method,
    )


def list_ingested_documents(driver: Driver, *, topic_id: str | None = None) -> list[dict[str, Any]]:
    """Return a summary of ingested documents (used by notebook checks).

    Walks the v2.1 spine ``DOCUMENT → CHAPTER → SECTION → PAGE`` to count
    pages (plan §5). The page count is collapsed via ``count(DISTINCT p)`` so
    a PAGE reachable through multiple SECTIONs (shouldn't happen in v1, but
    is safe under the spine) is not double-counted.
    """
    cypher = """
    MATCH (d:DOCUMENT)
    OPTIONAL MATCH (d)-[:CONSIST_OF]->(c:CHAPTER)
    OPTIONAL MATCH (c)-[:INCLUDE]->(s:SECTION)
    OPTIONAL MATCH (s)-[:INCLUDE]->(p:PAGE)
    WITH d, count(DISTINCT c) AS chapter_count,
              count(DISTINCT s) AS section_count,
              count(DISTINCT p) AS page_count
    """ + ("WHERE EXISTS { MATCH (:TOPIC {id: $topic_id})-[:CONTAIN]->(d) } " if topic_id else "") + """
    RETURN d.id AS id, d.title AS title, d.tier AS tier, d.backend AS backend,
           d.edition AS edition, d.publisher AS publisher,
           d.publicationYear AS publicationYear,
           d.editorialLayers AS editorialLayers,
           chapter_count, section_count, page_count
    ORDER BY d.title
    """
    with driver.session() as session:
        result = session.run(cypher, topic_id=topic_id) if topic_id else session.run(cypher)
        return [dict(r) for r in result]
