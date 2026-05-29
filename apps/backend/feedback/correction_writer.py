"""Correction writer — Phase 5 (plan §6 Stage 5, Move 2).

Writes span-level ``(:CORRECTION)`` nodes and ``(:FEW_SHOT_EXAMPLE)`` nodes
to Neo4j when a human reviewer confirms or corrects an OCR span.

The CORRECTION schema mirrors plan §5:
  id, pageId, before, after, spanBbox, spanCharRange, problemClass,
  editorId, ts, humanSeconds, language, tier

A FEW_SHOT_EXAMPLE is created from every CORRECTION so Engine B can
pull the top-3 nearest exemplars (by problem class + language) into its
system prompt for the next 24 h of OCR jobs.

Public API
----------
CorrectionInput     — input dataclass
write_correction(driver, inp) -> str   (correction_id)
write_few_shot_example(driver, correction_id, image_uri, gt_text,
                       problem_class, language) -> str
get_corrections_for_page(driver, page_id) -> list[dict]
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class CorrectionInput:
    """Input for a single span-level correction.

    Attributes:
        page_id: The PAGE node id being corrected.
        before: Original (wrong) OCR text in the span.
        after: Human-corrected text.
        span_bbox: Region bounding box ``[x, y, w, h]`` in pixel coordinates
            of the preprocessed image (None if not from OCR page).
        span_char_range: ``[start, end]`` character offsets within
            ``PAGE.textFused`` (inclusive start, exclusive end).
        problem_class: The PROBLEM_CLASS code this correction resolves.
        editor_id: Editor identifier (username or session id).
        human_seconds: Time the reviewer spent on this correction.
        language: ``PAGE.language`` value.
        tier: ``DOCUMENT.tier`` value (``'primary'`` or ``'secondary'``).
    """

    page_id: str
    before: str
    after: str
    span_bbox: list[float] | None = None       # [x, y, w, h]
    span_char_range: list[int] | None = None   # [start, end]
    problem_class: str = "OK"
    editor_id: str = "system"
    human_seconds: float = 0.0
    language: str = "zh-classical"
    tier: str = "primary"


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

_CORRECTION_UPSERT = """
MERGE (c:CORRECTION {id: $id})
ON CREATE SET
    c.pageId          = $page_id,
    c.before          = $before,
    c.after           = $after,
    c.spanBbox        = $span_bbox,
    c.spanCharRange   = $span_char_range,
    c.problemClass    = $problem_class,
    c.editorId        = $editor_id,
    c.ts              = $ts,
    c.humanSeconds    = $human_seconds,
    c.language        = $language,
    c.tier            = $tier,
    c.createdAt       = $ts
ON MATCH SET
    c.after           = $after,
    c.humanSeconds    = c.humanSeconds + $human_seconds
RETURN c.id AS id
"""

_PAGE_CONFIRM_SPAN = """
MATCH (p:PAGE {id: $page_id})
SET p.hasConfirmedSpans = true,
    p.lastCorrectedAt  = $ts
"""

_CORRECTION_PAGE_REL = """
MATCH (c:CORRECTION {id: $correction_id})
MATCH (p:PAGE {id: $page_id})
MERGE (c)-[:CORRECTS]->(p)
"""

_FEW_SHOT_UPSERT = """
MERGE (f:FEW_SHOT_EXAMPLE {id: $id})
ON CREATE SET
    f.problemClass       = $problem_class,
    f.imageUri           = $image_uri,
    f.gtText             = $gt_text,
    f.sourceCorrectionId = $correction_id,
    f.language           = $language,
    f.createdAt          = $ts
RETURN f.id AS id
"""

_FEW_SHOT_CORRECTION_REL = """
MATCH (f:FEW_SHOT_EXAMPLE {id: $fse_id})
MATCH (c:CORRECTION {id: $correction_id})
MERGE (c)-[:PRODUCED]->(f)
"""


# ---------------------------------------------------------------------------
# Writer functions
# ---------------------------------------------------------------------------


def _correction_id(page_id: str, span_char_range: list[int] | None, ts: str) -> str:
    """Deterministic id: hash(page_id + span + ts)."""
    key = f"{page_id}::{span_char_range}::{ts}"
    return "corr_" + hashlib.sha1(key.encode()).hexdigest()[:16]


def write_correction(driver: Driver, inp: CorrectionInput) -> str:
    """Write a span-level CORRECTION node and stamp the PAGE.

    Args:
        driver: Open Neo4j driver.
        inp: :class:`CorrectionInput` with all span details.

    Returns:
        The ``correction_id`` string (``corr_<sha1[:16]>``).
    """
    ts = datetime.now(timezone.utc).isoformat()
    cid = _correction_id(inp.page_id, inp.span_char_range, ts)

    with driver.session() as s:
        s.run(
            _CORRECTION_UPSERT,
            id=cid,
            page_id=inp.page_id,
            before=inp.before[:2000],
            after=inp.after[:2000],
            span_bbox=inp.span_bbox,
            span_char_range=inp.span_char_range,
            problem_class=inp.problem_class,
            editor_id=inp.editor_id,
            ts=ts,
            human_seconds=inp.human_seconds,
            language=inp.language,
            tier=inp.tier,
        ).consume()
        s.run(_PAGE_CONFIRM_SPAN, page_id=inp.page_id, ts=ts).consume()
        s.run(_CORRECTION_PAGE_REL, correction_id=cid, page_id=inp.page_id).consume()

    log.info(
        "write_correction: id=%s page=%s class=%s tier=%s",
        cid, inp.page_id[:60], inp.problem_class, inp.tier,
    )
    return cid


def write_few_shot_example(
    driver: Driver,
    correction_id: str,
    image_uri: str,
    gt_text: str,
    problem_class: str,
    language: str,
) -> str:
    """Write a FEW_SHOT_EXAMPLE node linked to a CORRECTION.

    Args:
        driver: Open Neo4j driver.
        correction_id: Id of the parent CORRECTION node.
        image_uri: MinIO URI for the cropped page region image.
        gt_text: Ground-truth corrected text.
        problem_class: PROBLEM_CLASS code.
        language: PAGE language.

    Returns:
        The ``fse_id`` string.
    """
    ts = datetime.now(timezone.utc).isoformat()
    fse_id = "fse_" + hashlib.sha1(f"{correction_id}::{ts}".encode()).hexdigest()[:16]

    with driver.session() as s:
        s.run(
            _FEW_SHOT_UPSERT,
            id=fse_id,
            problem_class=problem_class,
            image_uri=image_uri,
            gt_text=gt_text[:2000],
            correction_id=correction_id,
            language=language,
            ts=ts,
        ).consume()
        s.run(
            _FEW_SHOT_CORRECTION_REL,
            fse_id=fse_id,
            correction_id=correction_id,
        ).consume()

    log.info("write_few_shot_example: id=%s class=%s lang=%s", fse_id, problem_class, language)
    return fse_id


def get_corrections_for_page(driver: Driver, page_id: str) -> list[dict[str, Any]]:
    """Return all CORRECTION nodes for a given PAGE, newest first."""
    with driver.session() as s:
        rows = s.run(
            "MATCH (c:CORRECTION)-[:CORRECTS]->(p:PAGE {id: $page_id}) "
            "RETURN c.id AS id, c.before AS before, c.after AS after, "
            "c.spanCharRange AS range, c.problemClass AS cls, "
            "c.humanSeconds AS secs, c.ts AS ts "
            "ORDER BY c.ts DESC",
            page_id=page_id,
        ).data()
    return rows
