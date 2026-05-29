"""Idempotent Neo4j schema initializer for the Ancient China KG.

Spec source: ``.cursor/plans/ancient-chinese-search-engine_*.plan.md`` §5
("Neo4j schema (refined v2)"). Every Cypher statement uses ``IF NOT EXISTS`` so
this function is safe to call from any notebook or migration script per
AGENTS.md §6 ("For schema changes: write a migration script, not a manual
Cypher session").

Conventions per AGENTS.md §4:
- ``UPPER_SNAKE_CASE`` node labels (single-word labels are simply uppercase).
- ``UPPER_SNAKE_CASE`` relationship types.
- ``camelCase`` property keys.
- All queries parameterized.

Embedding dimension is read from ``EMBEDDING_DIMS`` (default 2048, matching
Silra's ``text-embedding-v4``). The bake-off in Phase 7 may swap this.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from neo4j import Driver

logger = logging.getLogger(__name__)


# Uniqueness constraints per Phase 0 (every node label that has an ``id`` /
# natural key in plan §5). Adding new labels later just means appending here.
CONSTRAINTS: list[tuple[str, str, str]] = [
    # (constraint_name, label, property)
    ("user_id_unique", "USER", "id"),
    ("topic_id_unique", "TOPIC", "id"),
    ("document_id_unique", "DOCUMENT", "id"),
    ("chapter_id_unique", "CHAPTER", "id"),
    ("section_id_unique", "SECTION", "id"),
    ("page_id_unique", "PAGE", "id"),
    ("chunk_id_unique", "CHUNK", "id"),
    ("keyword_name_unique", "KEYWORD", "name"),
    ("dictionary_entry_id_unique", "DICTIONARY_ENTRY", "id"),
    ("norm_id_unique", "NORM", "id"),
    ("problem_class_code_unique", "PROBLEM_CLASS", "code"),
    ("correction_id_unique", "CORRECTION", "id"),
    ("few_shot_example_id_unique", "FEW_SHOT_EXAMPLE", "id"),
    ("verifier_failure_id_unique", "VERIFIER_FAILURE", "id"),
    ("metrics_snapshot_ts_unique", "METRICS_SNAPSHOT", "ts"),
    ("community_id_unique", "COMMUNITY", "id"),
]


# Vector indexes per plan §5 ("Vector indexes"). Dimension comes from
# ``EMBEDDING_DIMS``. ``page_image_embedding_index`` shares this dim by default;
# Phase 7b ADR may pick a separate dim for the vision encoder.
VECTOR_INDEXES: list[tuple[str, str, str]] = [
    # (index_name, label, property)
    # Primary search index — Phase 5 writes CHUNK.embedding (1024-dim text-embedding-v4).
    # Phase 7 similarity search queries this index. Dimension is always 1024 per
    # AGENTS.md §11 (2026-05-16): EMBEDDING_DIMS=1024 is canonical for text-embedding-v4.
    ("chunk_embedding_vector_index", "CHUNK", "embedding"),
    # Legacy / future per-register splits (Phase 7b bakeoff may enable these).
    ("chunk_embedding_classical", "CHUNK", "embeddingClassical"),
    ("chunk_embedding_vernacular", "CHUNK", "embeddingVernacular"),
    ("keyword_embedding_index", "KEYWORD", "embedding"),
    ("page_image_embedding_index", "PAGE", "imageEmbedding"),
    ("community_summary_embedding_index", "COMMUNITY", "embedding"),
]


# Lookup indexes per plan §5 — non-vector indexes that accelerate Cypher
# scans. ``DOCUMENT.tier`` and ``PAGE.tier`` are queried by every search.
#
# v2.1 spine note (plan §5): CHAPTER/SECTION sit between DOCUMENT and PAGE.
# We index the foreign-key-like columns (``CHAPTER.documentId``,
# ``SECTION.chapterId``, ``PAGE.sectionId``, ``CHUNK.sectionId``) because
# every list/aggregation query walks them.
LOOKUP_INDEXES: list[tuple[str, str, str]] = [
    # (index_name, label, property)
    ("document_tier_index", "DOCUMENT", "tier"),
    ("document_source_path_index", "DOCUMENT", "sourcePath"),
    ("chapter_document_index", "CHAPTER", "documentId"),
    ("section_chapter_index", "SECTION", "chapterId"),
    ("page_tier_index", "PAGE", "tier"),
    ("page_language_index", "PAGE", "language"),
    ("page_doc_index_index", "PAGE", "docPageIndex"),
    ("page_section_index", "PAGE", "sectionId"),
    ("page_role_index", "PAGE", "role"),
    ("page_preprocessing_status_index", "PAGE", "preprocessingStatus"),
    ("page_paddle_ocr_status_index", "PAGE", "paddleOcrStatus"),
    ("page_deepseek_ocr_status_index", "PAGE", "deepseekOcrStatus"),
    ("page_qwen_vl_ocr_status_index", "PAGE", "qwenVlOcrStatus"),
    ("page_fusion_status_index", "PAGE", "fusionStatus"),
    ("page_layout_status_index", "PAGE", "layoutStatus"),
    ("page_type_index", "PAGE", "pageType"),
    ("chunk_tier_index", "CHUNK", "tier"),
    ("chunk_language_index", "CHUNK", "language"),
    ("chunk_editorial_layer_index", "CHUNK", "editorialLayerType"),
    ("chunk_section_index", "CHUNK", "sectionId"),
    # Phase 5 — added for chunking + embedding pipeline queries.
    ("chunk_page_id_index", "CHUNK", "pageId"),
    ("chunk_document_id_index", "CHUNK", "documentId"),
    ("chunk_embedding_status_index", "CHUNK", "embeddingStatus"),
    ("chunk_embedding_model_index", "CHUNK", "embeddingModel"),
    ("chunk_chunking_at_index", "CHUNK", "chunkingAt"),
    # Phase 6 — KG construction pipeline gating.
    ("chunk_mention_status_index", "CHUNK", "mentionStatus"),
    # Phase 6 — KEYWORD node indexes.
    ("keyword_type_index", "KEYWORD", "type"),
    ("keyword_frequency_index", "KEYWORD", "frequency"),
    # Phase 4 — Evaluator + Problem Classifier pipeline gating.
    ("page_evaluation_status_index", "PAGE", "evaluationStatus"),
    ("page_evaluation_decision_index", "PAGE", "evaluationDecision"),
    ("page_problem_class_index", "PAGE", "problemClass"),
    ("page_inter_engine_cer_index", "PAGE", "interEngineCer"),
    # Phase 4 — PROBLEM_CLASS node index.
    ("problem_class_language_index", "PROBLEM_CLASS", "language"),
    ("problem_class_routing_index", "PROBLEM_CLASS", "routing"),
    # Phase 5 — CORRECTION and FEW_SHOT_EXAMPLE indexes.
    ("correction_page_id_index", "CORRECTION", "pageId"),
    ("correction_tier_index", "CORRECTION", "tier"),
    ("few_shot_problem_class_index", "FEW_SHOT_EXAMPLE", "problemClass"),
    ("few_shot_language_index", "FEW_SHOT_EXAMPLE", "language"),
    # Phase 5 — active learning priority score.
    ("page_priority_score_index", "PAGE", "priorityScore"),
    # Phase 6 — translation pipeline gating + KB indexes.
    ("chunk_translation_status_index", "CHUNK", "translationStatus"),
    ("dictionary_entry_language_index", "DICTIONARY_ENTRY", "language"),
    ("dictionary_entry_term_index", "DICTIONARY_ENTRY", "term"),
    ("norm_tradition_index", "NORM", "tradition"),
    ("norm_scope_index", "NORM", "scope"),
]


def init_schema(
    driver: Driver,
    embedding_dims: int | None = None,
    similarity: str = "cosine",
) -> dict[str, Any]:
    """Create constraints and vector indexes idempotently.

    Args:
        driver: An open Neo4j driver.
        embedding_dims: Vector dimension for every text/image vector index.
            Defaults to ``EMBEDDING_DIMS`` env var (fallback 2048).
        similarity: ``"cosine"`` (default) or ``"euclidean"``. Cosine is
            consistent with Silra ``text-embedding-v4`` per ADR-08 (pending).

    Returns:
        A summary dict::

            {
                "embedding_dims": int,
                "similarity": str,
                "constraints": list[str],   # all constraint names that exist post-run
                "vector_indexes": list[dict],  # name + label + property + dims + state
                "errors": list[str],
            }
    """
    if embedding_dims is None:
        # AGENTS.md §11 (2026-05-16): EMBEDDING_DIMS=1024 is canonical for text-embedding-v4.
        embedding_dims = int(os.getenv("EMBEDDING_DIMS", "1024"))
    if similarity not in {"cosine", "euclidean"}:
        raise ValueError(f"similarity must be 'cosine' or 'euclidean', got {similarity!r}")

    errors: list[str] = []

    with driver.session() as session:
        for name, label, prop in CONSTRAINTS:
            cypher = (
                f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            )
            try:
                session.run(cypher).consume()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"constraint {name} failed: {exc}")
                logger.exception("Constraint %s failed", name)

        for name, label, prop in LOOKUP_INDEXES:
            cypher = (
                f"CREATE INDEX {name} IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.{prop})"
            )
            try:
                session.run(cypher).consume()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"lookup index {name} failed: {exc}")
                logger.exception("Lookup index %s failed", name)

        for name, label, prop in VECTOR_INDEXES:
            cypher = (
                f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.{prop}) "
                "OPTIONS {indexConfig: {"
                " `vector.dimensions`: $dims,"
                " `vector.similarity_function`: $sim"
                "}}"
            )
            try:
                session.run(cypher, dims=embedding_dims, sim=similarity).consume()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"vector index {name} failed: {exc}")
                logger.exception("Vector index %s failed", name)

        constraint_names: list[str] = [
            r["name"] for r in session.run("SHOW CONSTRAINTS YIELD name").data()
        ]

        index_rows = session.run(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state, "
            "options WHERE type = 'VECTOR' RETURN name, labelsOrTypes, "
            "properties, state, options"
        ).data()

        lookup_rows = session.run(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state "
            "WHERE type = 'RANGE' AND ("
            " name STARTS WITH 'document_' OR"
            " name STARTS WITH 'chapter_' OR"
            " name STARTS WITH 'section_' OR"
            " name STARTS WITH 'page_' OR"
            " name STARTS WITH 'chunk_'"
            ") RETURN name, labelsOrTypes, properties, state"
        ).data()

    vector_indexes_report: list[dict[str, Any]] = []
    for row in index_rows:
        opts = row.get("options") or {}
        cfg = (opts.get("indexConfig") or {}) if isinstance(opts, dict) else {}
        vector_indexes_report.append(
            {
                "name": row["name"],
                "label": (row["labelsOrTypes"] or [""])[0],
                "property": (row["properties"] or [""])[0],
                "dims": cfg.get("vector.dimensions"),
                "similarity": cfg.get("vector.similarity_function"),
                "state": row["state"],
            }
        )

    lookup_indexes_report: list[dict[str, Any]] = []
    for row in lookup_rows:
        labels = row.get("labelsOrTypes") or [""]
        props = row.get("properties") or [""]
        lookup_indexes_report.append(
            {
                "name": row["name"],
                "label": labels[0],
                "property": props[0],
                "state": row.get("state"),
            }
        )

    return {
        "embedding_dims": embedding_dims,
        "similarity": similarity,
        "constraints": sorted(constraint_names),
        "vector_indexes": vector_indexes_report,
        "lookup_indexes": lookup_indexes_report,
        "errors": errors,
    }


# Mapping from the legacy ``PascalCase`` label set (pre-2026-05-18) to the
# current ``UPPER_SNAKE_CASE`` labels. Used by :func:`migrate_labels_to_uppercase`
# to relabel pre-existing nodes after the label-casing rule changed (AGENTS.md
# §11 entry dated 2026-05-18).
LEGACY_LABEL_MAP: dict[str, str] = {
    "User": "USER",
    "Topic": "TOPIC",
    "Document": "DOCUMENT",
    "Chapter": "CHAPTER",
    "Section": "SECTION",
    "Page": "PAGE",
    "Chunk": "CHUNK",
    "Keyword": "KEYWORD",
    "DictionaryEntry": "DICTIONARY_ENTRY",
    "Norm": "NORM",
    "ProblemClass": "PROBLEM_CLASS",
    "Correction": "CORRECTION",
    "FewShotExample": "FEW_SHOT_EXAMPLE",
    "VerifierFailure": "VERIFIER_FAILURE",
    "MetricsSnapshot": "METRICS_SNAPSHOT",
    "Community": "COMMUNITY",
}


def migrate_labels_to_uppercase(
    driver: Driver,
    *,
    drop_legacy_constraints: bool = True,
    relabel_existing_nodes: bool = True,
    delete_legacy_indexes: bool = True,
) -> dict[str, Any]:
    """One-shot migration from ``PascalCase`` to ``UPPER_SNAKE_CASE`` labels.

    Needed when a Neo4j database was previously initialised with the legacy
    ``PascalCase`` label set (the schema prior to AGENTS.md §11 entry dated
    2026-05-18). ``CREATE CONSTRAINT ... IF NOT EXISTS`` raises
    ``EquivalentSchemaRuleAlreadyExistsException`` when the constraint name
    collides with a different label spec, so a fresh ``init_schema()`` call
    will fail without this step.

    The function is **idempotent**: re-running on an already-migrated
    database is a no-op (every step uses ``IF EXISTS`` or ``MATCH`` guards
    that simply find nothing).

    Args:
        driver: Open Neo4j driver.
        drop_legacy_constraints: Drop constraints whose label matches a
            legacy PascalCase key in :data:`LEGACY_LABEL_MAP`.
        relabel_existing_nodes: Relabel every node carrying a legacy label
            (e.g. ``MATCH (n:User) REMOVE n:User SET n:USER``). Properties
            and relationships are preserved.
        delete_legacy_indexes: Drop indexes whose label matches a legacy
            PascalCase key. Vector indexes are dropped + recreated by the
            subsequent :func:`init_schema` call.

    Returns:
        A summary dict::

            {
                "constraints_dropped": list[str],
                "indexes_dropped": list[str],
                "nodes_relabelled": dict[str, int],   # old_label -> count
                "errors": list[str],
            }

    After calling this, call :func:`init_schema` to recreate the uppercase
    constraints + indexes.
    """
    constraints_dropped: list[str] = []
    indexes_dropped: list[str] = []
    nodes_relabelled: dict[str, int] = {}
    errors: list[str] = []

    with driver.session() as session:
        if drop_legacy_constraints:
            rows = session.run(
                "SHOW CONSTRAINTS YIELD name, labelsOrTypes "
                "RETURN name, labelsOrTypes"
            ).data()
            for row in rows:
                label = (row.get("labelsOrTypes") or [""])[0]
                if label in LEGACY_LABEL_MAP:
                    cname = row["name"]
                    try:
                        session.run(f"DROP CONSTRAINT {cname} IF EXISTS").consume()
                        constraints_dropped.append(cname)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"drop constraint {cname} failed: {exc}")
                        logger.exception("Failed to drop legacy constraint %s", cname)

        if delete_legacy_indexes:
            rows = session.run(
                "SHOW INDEXES YIELD name, labelsOrTypes, type "
                "RETURN name, labelsOrTypes, type"
            ).data()
            for row in rows:
                label = (row.get("labelsOrTypes") or [""])[0]
                if label in LEGACY_LABEL_MAP:
                    iname = row["name"]
                    try:
                        session.run(f"DROP INDEX {iname} IF EXISTS").consume()
                        indexes_dropped.append(iname)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"drop index {iname} failed: {exc}")
                        logger.exception("Failed to drop legacy index %s", iname)

        if relabel_existing_nodes:
            for old, new in LEGACY_LABEL_MAP.items():
                try:
                    rec = session.run(
                        f"MATCH (n:`{old}`) "
                        f"WITH n, count(*) AS _ "  # force aggregation in case of duplicates
                        f"REMOVE n:`{old}` SET n:`{new}` "
                        f"RETURN count(n) AS relabelled"
                    ).single()
                    n = rec["relabelled"] if rec else 0
                    if n > 0:
                        nodes_relabelled[old] = n
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"relabel {old} -> {new} failed: {exc}")
                    logger.exception("Failed to relabel %s -> %s", old, new)

    return {
        "constraints_dropped": constraints_dropped,
        "indexes_dropped": indexes_dropped,
        "nodes_relabelled": nodes_relabelled,
        "errors": errors,
    }
