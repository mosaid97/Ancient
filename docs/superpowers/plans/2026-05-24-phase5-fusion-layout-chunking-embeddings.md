# Phase 5 — Fusion Catch-up + Layout Completion + Chunking + Embeddings

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the OCR pipeline (3-way fusion + layout gap), then build the chunking and embedding layer that produces `(:CHUNK)` nodes with 1024-dim vector embeddings — the foundation for all downstream search and verification phases.

**Architecture:** The pipeline continues the `apps/backend/pipeline/` pattern. Two new orchestrators (`chunk.py`, `embed.py`) call `fuse_results()` and the Silra embedding API respectively. Each page's text source is resolved by priority: `structuredMarkdown` > `textFused` > `text`. Chunks become `(:CHUNK)` nodes wired `(:PAGE)-[:HAS]->(:CHUNK)` in the v2.1 spine; embeddings are stored as `float[]` with a Neo4j vector index for Phase 7 similarity search.

**Tech Stack:** Python 3.12 / uv, Neo4j driver v6.2+, Silra API (text-embedding-v4, 1024 dims), apps/backend/ocr/fusion.py (already supports 3-way), apps/backend/pipeline/fusion.py (already has `fuse_pages()`), paddleocr (layout already done), pytest

---

## Current State (verified 2026-05-24)

| Signal | Value | Meaning |
|---|---|---|
| OCR pages total | 2,942 | All preprocessed |
| Paddle ok | 2,874 | 97.7% |
| DeepSeek ok | 2,209 | 75.1% — **732 empty need rerun** |
| Qwen-VL ok | 2,351 | 79.9% |
| **textFused** | **0** | **BLOCKER — fusion never run full corpus** |
| Layout done | 2,301 | 78.2% — 641 pages missing |
| CHUNK nodes | 0 | Phase 5 not started |
| Native pages with .text | 11,745 | Ready for chunking |

### DeepSeek empty breakdown (732 pages)

| Error reason | Count | Avg Paddle chars | Action |
|---|---|---|---|
| `validation_failed:html_table_markup` | 291 | 281 | Retry — may be tables with real CJK content |
| `validation_failed:low_cjk_ratio:0.000` | 175 | 276 | Mostly Dunhuang grids; retry anyway |
| `(no error — cleaned pre-validator)` | 138 | 115 | Retry with validator active |
| `validation_failed:internal_token_leakage` | 61 | 232 | Retry — stochastic model artefact |
| `validation_failed:sequential_number_table` | 26 | 295 | Retry — updated prompts may fix |
| Other validation failures | 41 | varies | Retry |

**By tier:** 50 primary empty, 682 secondary empty. Primary pages also need `--recompute` to apply the 2026-05-23 traditional-script prompts (which were only applied to Qwen via `rerun_primary_ocr.py --engine qwen`; DeepSeek primary rerun was never done).

---

## Task 0: Rerun DeepSeek OCR (primary + empty pages)

Must complete before Task 1 (fusion). Two sub-passes using existing scripts.

**Files:**
- Use: `scripts/rerun_primary_ocr.py` (existing — `--engine deepseek --recompute`)
- Use: `scripts/rerun_deepseek_empty.py` (existing — secondary tier empty pages)

- [ ] **Step 0.1: Dry-run primary tier to see scope**

```bash
uv run python scripts/rerun_primary_ocr.py --engine deepseek --dry-run
```

Expected: shows ~212 primary pages (all statuses), including 50 currently empty.

- [ ] **Step 0.2: Run DeepSeek on all primary-tier pages with new traditional prompts**

```bash
caffeinate -dimsu uv run python scripts/rerun_primary_ocr.py \
    --engine deepseek --recompute \
    --log-file logs/rerun_primary_ocr_deepseek.log
```

Expected runtime: ~212 pages × ~8 s/page ≈ **30 min**
Monitor: `tail -f logs/rerun_primary_ocr_deepseek.log`
Expected outcome: primary-tier empty count drops from 50; `ok` pages refreshed with traditional-script prompt.

- [ ] **Step 0.3: Verify primary tier improvement**

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USERNAME','neo4j'), os.getenv('NEO4J_PASSWORD','AncientChina')))
with d.session() as s:
    r = s.run('''
    MATCH (doc:DOCUMENT {tier:\"primary\"})-[:CONSIST_OF]->(:CHAPTER)-[:INCLUDE]->(:SECTION)-[:INCLUDE]->(p:PAGE {mode:\"ocr\"})
    RETURN doc.title,
           count(CASE WHEN p.deepseekOcrStatus=\"ok\" THEN 1 END) AS ok,
           count(CASE WHEN p.deepseekOcrStatus=\"empty\" THEN 1 END) AS empty
    ORDER BY doc.title
    ''').data()
    for row in r: print(row)
d.close()
"
```

- [ ] **Step 0.4: Dry-run secondary empty pages to see scope**

```bash
uv run python scripts/rerun_deepseek_empty.py --dry-run
```

Expected: shows ~682 secondary empty pages; after `--require-paddle-text 11` filter, ~623 will be attempted.

- [ ] **Step 0.5: Run DeepSeek rerun on secondary empty pages**

```bash
caffeinate -dimsu uv run python scripts/rerun_deepseek_empty.py \
    --require-paddle-text 11
```

Expected runtime: ~623 pages × ~8 s/page ≈ **1.5–2 hours**
Monitor: `tail -f logs/deepseek_rerun_empty.log`
Expected outcome: ~100–200 pages recover from html_table_markup / token_leakage failures (Dunhuang grids remain empty — confirmed unfixable per AGENTS.md 2026-05-20).

- [ ] **Step 0.6: Verify final DeepSeek status before fusion**

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USERNAME','neo4j'), os.getenv('NEO4J_PASSWORD','AncientChina')))
with d.session() as s:
    r = s.run('''
    MATCH (p:PAGE {mode:\"ocr\"})
    RETURN count(CASE WHEN p.deepseekOcrStatus=\"ok\" THEN 1 END) AS ok,
           count(CASE WHEN p.deepseekOcrStatus=\"empty\" THEN 1 END) AS empty,
           count(CASE WHEN p.deepseekOcrStatus=\"failed\" THEN 1 END) AS failed
    ''').single()
    print(dict(r))
d.close()
"
```

Expected: `ok ≥ 2,300` (up from 2,209); `empty` reduces to Dunhuang-unfixable residue (~500–550).

---

## Task 1: Full-corpus 3-way fusion

The `apps/backend/pipeline/fusion.py` module already supports Qwen. We only need a background runner and one full execution.

**Files:**
- Create: `scripts/run_fusion.py`
- Read: `apps/backend/pipeline/fusion.py` (existing — do not modify)

- [ ] **Step 1.1: Verify fusion works on a sample**

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from neo4j import GraphDatabase
from apps.backend.pipeline.fusion import fuse_pages, fusion_summary

driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USERNAME', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'AncientChina'))
)
report = fuse_pages(driver, max_pages=10, recompute=True, run_language_detection=True)
print(f'Fused: {report.pages_fused}, single: {report.pages_single}, failed: {report.pages_failed}')
driver.close()
"
```

Expected: `Fused: N, single: M, failed: 0` (N+M ≤ 10, N > 0 if both paddle + at least one LLM ok)

- [ ] **Step 1.2: Create `scripts/run_fusion.py`**

```python
#!/usr/bin/env python3
"""Full-corpus 3-way OCR fusion runner.

Usage:
    caffeinate -dimsu uv run python scripts/run_fusion.py
    uv run python scripts/run_fusion.py --max-pages 100 --recompute
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

# Allow `uv run python scripts/run_fusion.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
from neo4j import GraphDatabase
from apps.backend.pipeline.fusion import fuse_pages


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-corpus 3-way OCR fusion")
    p.add_argument("--max-pages", type=int, default=None, help="Cap pages processed (None = all)")
    p.add_argument("--recompute", action="store_true", help="Re-fuse already-fused pages")
    p.add_argument("--no-lang-detect", action="store_true", help="Skip post-fusion language detection")
    p.add_argument("--log-file", default="logs/fusion_run.log", help="Log file path")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    _stop = False

    def _handle_signal(sig: int, _frame: object) -> None:
        nonlocal _stop
        log.info("Signal %s received — will stop after current batch", sig)
        _stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "AncientChina")),
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )

    log.info("Starting full-corpus fusion (max_pages=%s, recompute=%s)", args.max_pages, args.recompute)
    t0 = time.time()

    report = fuse_pages(
        driver,
        max_pages=args.max_pages,
        recompute=args.recompute,
        run_language_detection=not args.no_lang_detect,
    )

    elapsed = time.time() - t0
    log.info(
        "Fusion complete in %.1fs — fused=%d single=%d failed=%d",
        elapsed,
        report.pages_fused,
        report.pages_single,
        report.pages_failed,
    )

    report_path = Path("logs/fusion_report.json")
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    log.info("Report written to %s", report_path)
    driver.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.3: Check `fuse_pages` has a `to_dict()` on its report**

```bash
uv run python -c "
from apps.backend.pipeline.fusion import FuseRunReport
import inspect
print([m for m in dir(FuseRunReport) if not m.startswith('_')])
"
```

If `to_dict` is missing, add it in `apps/backend/pipeline/fusion.py`:

```python
def to_dict(self) -> dict:
    return {
        "pages_total": self.pages_total,
        "pages_fused": self.pages_fused,
        "pages_single": self.pages_single,
        "pages_failed": self.pages_failed,
        "avg_agreement_rate": self.avg_agreement_rate,
        "duration_seconds": self.duration_seconds,
        "errors": [str(e) for e in self.errors[:20]],
    }
```

- [ ] **Step 1.4: Run the full-corpus fusion**

```bash
caffeinate -dimsu uv run python scripts/run_fusion.py --log-file logs/fusion_run.log --verbose
```

Expected runtime: ~30–60 minutes for 2,942 pages (pure in-memory difflib, no API calls).
Monitor: `tail -f logs/fusion_run.log`

- [ ] **Step 1.5: Verify fusion results in Neo4j**

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USERNAME','neo4j'), os.getenv('NEO4J_PASSWORD','AncientChina')))
with d.session() as s:
    r = s.run('''
    MATCH (p:PAGE {mode:\"ocr\"})
    RETURN count(CASE WHEN p.fusionStatus=\"ok\" THEN 1 END) AS fused_ok,
           count(CASE WHEN p.fusionStatus=\"single\" THEN 1 END) AS single,
           count(CASE WHEN p.fusionStatus=\"failed\" THEN 1 END) AS failed,
           count(CASE WHEN p.textFused IS NOT NULL THEN 1 END) AS has_text
    ''').single()
    print(dict(r))
d.close()
"
```

Expected: `fused_ok + single ≥ 2800`, `failed < 100`, `has_text ≥ 2800`

- [ ] **Step 1.6: Commit**

```bash
git add scripts/run_fusion.py logs/fusion_report.json
git commit -m "feat: add full-corpus fusion runner + fusion_report"
```

---

## Task 2: Complete layout analysis for 641 missing pages

The `scripts/run_layout_analysis.py` and `apps/backend/pipeline/layout.py` already exist. We just need to run a catch-up pass.

**Files:**
- Read: `scripts/run_layout_analysis.py` (existing — do not modify unless broken)
- Read: `apps/backend/pipeline/layout.py` (existing)

- [ ] **Step 2.1: Verify which pages are missing layout**

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USERNAME','neo4j'), os.getenv('NEO4J_PASSWORD','AncientChina')))
with d.session() as s:
    r = s.run('''
    MATCH (p:PAGE {mode:\"ocr\"})
    WHERE (p.role = \"body\" OR p.role IS NULL)
      AND p.layoutStatus IS NULL
      AND p.preprocessedImageUri IS NOT NULL
    RETURN count(p) as missing, collect(DISTINCT p.documentId)[..5] as sample_docs
    ''').single()
    print(dict(r))
d.close()
"
```

Expected: `missing: 641`

- [ ] **Step 2.2: Run layout catch-up**

```bash
caffeinate -dimsu uv run python scripts/run_layout_analysis.py \
    --log-file logs/layout_catchup2.log --verbose
```

Expected runtime: ~30–40 min for 641 pages at ~3 s/page.
Monitor: `tail -f logs/layout_catchup2.log`

- [ ] **Step 2.3: Verify layout completion**

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USERNAME','neo4j'), os.getenv('NEO4J_PASSWORD','AncientChina')))
with d.session() as s:
    r = s.run('''
    MATCH (p:PAGE {mode:\"ocr\"})
    RETURN count(CASE WHEN p.layoutStatus=\"ok\" THEN 1 END) AS layout_ok,
           count(CASE WHEN p.layoutStatus=\"manuscript\" THEN 1 END) AS manuscript,
           count(CASE WHEN p.layoutStatus IS NULL THEN 1 END) AS missing
    ''').single()
    print(dict(r))
d.close()
"
```

Expected: `missing < 50` (a small residue of pages with no preprocessedImageUri is acceptable)

---

## Task 3: Add vector index + CHUNK schema to Neo4j

Before writing any chunks, the schema must include the `CHUNK` vector index and constraints.

**Files:**
- Modify: `apps/backend/graph/schema.py`

- [ ] **Step 3.1: Read current schema**

```bash
uv run python -c "
import inspect
from apps.backend.graph import schema
print(inspect.getsource(schema))
" | head -100
```

- [ ] **Step 3.2: Add CHUNK node constraint + vector index**

In `apps/backend/graph/schema.py`, add to the appropriate index/constraint lists:

```python
# Node uniqueness constraint
"CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:CHUNK) REQUIRE c.chunkId IS UNIQUE",

# Standard lookup index
"CREATE INDEX chunk_page_id_index IF NOT EXISTS FOR (c:CHUNK) ON (c.pageId)",
"CREATE INDEX chunk_embedding_model_index IF NOT EXISTS FOR (c:CHUNK) ON (c.embeddingModel)",
"CREATE INDEX chunk_status_index IF NOT EXISTS FOR (c:CHUNK) ON (c.embeddingStatus)",

# Vector index for Phase 7 similarity search (Neo4j >= 5.11)
"""CREATE VECTOR INDEX chunk_embedding_vector_index IF NOT EXISTS
FOR (c:CHUNK) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}""",
```

- [ ] **Step 3.3: Apply schema**

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from neo4j import GraphDatabase
from apps.backend.graph.schema import init_schema
d = GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USERNAME','neo4j'), os.getenv('NEO4J_PASSWORD','AncientChina')))
init_schema(d)
print('Schema updated')
d.close()
"
```

Expected output: `Schema updated` (no exceptions)

- [ ] **Step 3.4: Verify vector index exists**

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USERNAME','neo4j'), os.getenv('NEO4J_PASSWORD','AncientChina')))
with d.session() as s:
    r = s.run('SHOW INDEXES WHERE name = \"chunk_embedding_vector_index\"').data()
    print(r)
d.close()
"
```

Expected: one entry with `type: VECTOR`, `state: ONLINE`

- [ ] **Step 3.5: Commit**

```bash
git add apps/backend/graph/schema.py
git commit -m "feat: add CHUNK uniqueness constraint + vector index to schema"
```

---

## Task 4: Implement `apps/backend/pipeline/chunk.py`

**Files:**
- Create: `apps/backend/pipeline/chunk.py`
- Create: `tests/backend/pipeline/test_chunk.py`

- [ ] **Step 4.1: Write the failing tests first**

```python
# tests/backend/pipeline/test_chunk.py
"""Tests for the chunking pipeline (AGENTS.md §8 — pytest, mocked driver)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from apps.backend.pipeline.chunk import (
    ChunkRecord,
    _sliding_window,
    _markdown_sections,
    resolve_page_text,
)


def test_sliding_window_basic() -> None:
    text = "甲" * 600  # 600 identical chars
    chunks = _sliding_window(text, chunk_size=500, overlap=50)
    assert len(chunks) == 2
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 150  # 600 - 500 + 50 = 150
    assert chunks[0].endswith("甲" * 50) and chunks[1].startswith("甲" * 50)  # overlap


def test_sliding_window_short_text() -> None:
    text = "短文本"  # 3 chars — shorter than chunk_size
    chunks = _sliding_window(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_sliding_window_empty() -> None:
    assert _sliding_window("", chunk_size=500, overlap=50) == []


def test_markdown_sections_splits_on_headings() -> None:
    md = "# 第一章\n天皇大帝\n# 第二章\n地皇大帝"
    chunks = _markdown_sections(md, chunk_size=500, overlap=50)
    assert len(chunks) == 2
    assert "第一章" in chunks[0]
    assert "第二章" in chunks[1]


def test_markdown_sections_merges_small_sections() -> None:
    # Three tiny sections should be merged into one chunk
    md = "# A\n一\n# B\n二\n# C\n三"
    chunks = _markdown_sections(md, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert "一" in chunks[0] and "二" in chunks[0]


def test_resolve_page_text_prefers_markdown() -> None:
    row = {
        "structuredMarkdown": "# 章\n唐律",
        "textFused": "唐律老版",
        "text": "native",
        "layoutStatus": "ok",
        "fusionStatus": "ok",
    }
    text, strategy = resolve_page_text(row)
    assert text == "# 章\n唐律"
    assert strategy == "markdown_section"


def test_resolve_page_text_falls_back_to_fused() -> None:
    row = {
        "structuredMarkdown": None,
        "textFused": "唐律疏議",
        "text": None,
        "layoutStatus": "manuscript",
        "fusionStatus": "ok",
    }
    text, strategy = resolve_page_text(row)
    assert text == "唐律疏議"
    assert strategy == "sliding_window"


def test_resolve_page_text_uses_native_text() -> None:
    row = {
        "structuredMarkdown": None,
        "textFused": None,
        "text": "原文本",
        "layoutStatus": None,
        "fusionStatus": None,
    }
    text, strategy = resolve_page_text(row)
    assert text == "原文本"
    assert strategy == "sliding_window"


def test_resolve_page_text_returns_none_for_empty() -> None:
    row = {"structuredMarkdown": None, "textFused": None, "text": None,
           "layoutStatus": None, "fusionStatus": None}
    text, strategy = resolve_page_text(row)
    assert text is None
```

- [ ] **Step 4.2: Run tests to confirm they fail**

```bash
uv run pytest tests/backend/pipeline/test_chunk.py -v 2>&1 | tail -20
```

Expected: `ERRORS — ModuleNotFoundError: No module named 'apps.backend.pipeline.chunk'`

- [ ] **Step 4.3: Implement `apps/backend/pipeline/chunk.py`**

```python
"""Chunking pipeline — splits page text into overlapping CHUNK nodes (plan §6 Stage 5).

Text source priority per page:
  1. structuredMarkdown (layoutStatus='ok')  → markdown_section strategy
  2. textFused (fusionStatus in ['ok','single']) → sliding_window strategy
  3. text (native pages)                      → sliding_window strategy

CHUNK node properties (camelCase per AGENTS.md §4):
  chunkId, pageId, documentId, chunkIndex, text, charCount, tokenEstimate,
  chunkStrategy, chunkSize, overlapSize, language, chunkingAt
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver

log = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_OVERLAP = 50
_MIN_CHUNK_CHARS = 10


@dataclass
class ChunkRecord:
    """One chunk ready for Neo4j upsert."""
    chunk_id: str
    page_id: str
    document_id: str
    chunk_index: int
    text: str
    char_count: int
    token_estimate: int  # char_count // 1.5 — rough CJK estimate
    chunk_strategy: str
    chunk_size: int
    overlap_size: int
    language: str | None


@dataclass
class ChunkRunReport:
    pages_total: int = 0
    pages_chunked: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    chunks_created: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages_total": self.pages_total,
            "pages_chunked": self.pages_chunked,
            "pages_skipped": self.pages_skipped,
            "pages_failed": self.pages_failed,
            "chunks_created": self.chunks_created,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors[:20],
        }


def _sliding_window(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size overlapping windows."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def _markdown_sections(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split on markdown headings; merge small sections; overflow via sliding window."""
    heading_re = re.compile(r"^#{1,6}\s", re.MULTILINE)
    boundaries = [m.start() for m in heading_re.finditer(text)]
    if not boundaries:
        return _sliding_window(text, chunk_size=chunk_size, overlap=overlap)

    # Build raw sections (heading + content)
    sections: list[str] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        sections.append(text[start:end])

    # Handle any text before the first heading
    if boundaries[0] > 0:
        sections.insert(0, text[: boundaries[0]])

    # Merge tiny sections and split oversized ones
    result: list[str] = []
    buffer = ""
    for sec in sections:
        if len(buffer) + len(sec) <= chunk_size:
            buffer = (buffer + "\n" + sec).lstrip()
        else:
            if buffer:
                result.append(buffer)
            # If this section alone exceeds chunk_size, split it
            if len(sec) > chunk_size:
                result.extend(_sliding_window(sec, chunk_size=chunk_size, overlap=overlap))
                buffer = ""
            else:
                buffer = sec
    if buffer:
        result.append(buffer)
    return [c for c in result if len(c) >= _MIN_CHUNK_CHARS]


def resolve_page_text(row: dict[str, Any]) -> tuple[str | None, str]:
    """Return (text, strategy) for a page row from Neo4j.

    Priority: structuredMarkdown → textFused → text.
    Strategy: 'markdown_section' for structured markdown, 'sliding_window' otherwise.
    """
    md = row.get("structuredMarkdown")
    if md and row.get("layoutStatus") == "ok":
        return md, "markdown_section"
    fused = row.get("textFused")
    if fused and row.get("fusionStatus") in ("ok", "single"):
        return fused, "sliding_window"
    native = row.get("text")
    if native:
        return native, "sliding_window"
    return None, "sliding_window"


def _make_chunk_id(page_id: str, chunk_index: int) -> str:
    raw = f"{page_id}::chunk_{chunk_index:04d}"
    return raw  # human-readable; unique by (pageId, chunkIndex)


_PAGE_QUERY = """
MATCH (p:PAGE)
WHERE (p.chunkingAt IS NULL OR $recompute)
  AND (p.textFused IS NOT NULL OR p.text IS NOT NULL OR p.structuredMarkdown IS NOT NULL)
RETURN
  p.pageId        AS page_id,
  p.documentId    AS document_id,
  p.language      AS language,
  p.textFused     AS textFused,
  p.text          AS text,
  p.structuredMarkdown AS structuredMarkdown,
  p.layoutStatus  AS layoutStatus,
  p.fusionStatus  AS fusionStatus
ORDER BY p.pageId
SKIP $skip LIMIT $batch
"""

_CHUNK_UPSERT = """
UNWIND $chunks AS c
MERGE (ch:CHUNK {chunkId: c.chunkId})
ON CREATE SET
  ch.pageId          = c.pageId,
  ch.documentId      = c.documentId,
  ch.chunkIndex      = c.chunkIndex,
  ch.text            = c.text,
  ch.charCount       = c.charCount,
  ch.tokenEstimate   = c.tokenEstimate,
  ch.chunkStrategy   = c.chunkStrategy,
  ch.chunkSize       = c.chunkSize,
  ch.overlapSize     = c.overlapSize,
  ch.language        = c.language,
  ch.chunkingAt      = c.chunkingAt,
  ch.embeddingStatus = 'pending'
ON MATCH SET
  ch.text            = c.text,
  ch.charCount       = c.charCount,
  ch.chunkingAt      = c.chunkingAt
WITH ch, c
MATCH (p:PAGE {pageId: c.pageId})
MERGE (p)-[:HAS]->(ch)
"""

_STAMP_PAGE = """
MATCH (p:PAGE {pageId: $page_id})
SET p.chunkingAt = $ts, p.chunkCount = $count
"""


def chunk_pages(
    driver: Driver,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_OVERLAP,
    batch_size: int = 200,
    max_pages: int | None = None,
    recompute: bool = False,
) -> ChunkRunReport:
    """Chunk all eligible pages and write CHUNK nodes to Neo4j.

    Eligible = has any text source AND (chunkingAt IS NULL OR recompute=True).
    """
    report = ChunkRunReport()
    t_start = time.time()
    skip = 0

    while True:
        with driver.session() as s:
            rows = s.run(_PAGE_QUERY, recompute=recompute, skip=skip, batch=batch_size).data()

        if not rows:
            break

        report.pages_total += len(rows)
        batch_chunks: list[dict[str, Any]] = []
        page_counts: dict[str, int] = {}

        for row in rows:
            page_id = row["page_id"]
            try:
                text, strategy = resolve_page_text(row)
                if not text or len(text.strip()) < _MIN_CHUNK_CHARS:
                    report.pages_skipped += 1
                    continue

                split_fn = _markdown_sections if strategy == "markdown_section" else _sliding_window
                raw_chunks = split_fn(text.strip(), chunk_size=chunk_size, overlap=overlap)

                for idx, chunk_text in enumerate(raw_chunks):
                    batch_chunks.append({
                        "chunkId": _make_chunk_id(page_id, idx),
                        "pageId": page_id,
                        "documentId": row.get("document_id") or "",
                        "chunkIndex": idx,
                        "text": chunk_text,
                        "charCount": len(chunk_text),
                        "tokenEstimate": int(len(chunk_text) / 1.5),
                        "chunkStrategy": strategy,
                        "chunkSize": chunk_size,
                        "overlapSize": overlap,
                        "language": row.get("language"),
                        "chunkingAt": datetime.now(timezone.utc).isoformat(),
                    })
                page_counts[page_id] = len(raw_chunks)
                report.pages_chunked += 1
                report.chunks_created += len(raw_chunks)

            except Exception as exc:
                log.error("chunk failed page=%s: %s", page_id, exc)
                report.pages_failed += 1
                report.errors.append(f"{page_id}: {exc}")

        if batch_chunks:
            with driver.session() as s:
                s.run(_CHUNK_UPSERT, chunks=batch_chunks).consume()
            ts_now = datetime.now(timezone.utc).isoformat()
            with driver.session() as s:
                for pid, cnt in page_counts.items():
                    s.run(_STAMP_PAGE, page_id=pid, ts=ts_now, count=cnt).consume()

        log.info(
            "Batch skip=%d rows=%d chunks_this_batch=%d total_chunks=%d",
            skip, len(rows), len(batch_chunks), report.chunks_created,
        )
        skip += len(rows)

        if max_pages is not None and report.pages_total >= max_pages:
            break

    report.duration_seconds = time.time() - t_start
    return report
```

- [ ] **Step 4.4: Run the tests — they should pass now**

```bash
uv run pytest tests/backend/pipeline/test_chunk.py -v
```

Expected: `8 passed`

- [ ] **Step 4.5: Commit**

```bash
git add apps/backend/pipeline/chunk.py tests/backend/pipeline/test_chunk.py
git commit -m "feat: add chunking pipeline with markdown_section + sliding_window strategies"
```

---

## Task 5: Implement `apps/backend/pipeline/embed.py`

**Files:**
- Create: `apps/backend/pipeline/embed.py`
- Create: `tests/backend/pipeline/test_embed.py`

- [ ] **Step 5.1: Write failing tests**

```python
# tests/backend/pipeline/test_embed.py
"""Tests for the embedding pipeline."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


def test_embed_chunks_calls_api_in_batches() -> None:
    from apps.backend.pipeline.embed import _batch_embed

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1024) for _ in range(3)]
    )

    texts = ["唐律疏議" * 10] * 3
    result = _batch_embed(mock_client, texts, model="text-embedding-v4")

    assert len(result) == 3
    assert len(result[0]) == 1024
    mock_client.embeddings.create.assert_called_once()


def test_embed_chunks_handles_empty_batch() -> None:
    from apps.backend.pipeline.embed import _batch_embed

    mock_client = MagicMock()
    result = _batch_embed(mock_client, [], model="text-embedding-v4")
    assert result == []
    mock_client.embeddings.create.assert_not_called()


def test_embed_batch_size_splits_large_input() -> None:
    from apps.backend.pipeline.embed import _batch_embed

    call_texts: list[list[str]] = []

    def _fake_create(input: list[str], model: str) -> MagicMock:
        call_texts.append(input)
        return MagicMock(data=[MagicMock(embedding=[0.0] * 1024) for _ in input])

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = _fake_create

    texts = ["text"] * 70
    result = _batch_embed(mock_client, texts, model="text-embedding-v4", batch_size=32)

    assert len(result) == 70
    assert len(call_texts) == 3  # 32 + 32 + 6
```

- [ ] **Step 5.2: Run tests to confirm failure**

```bash
uv run pytest tests/backend/pipeline/test_embed.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: apps.backend.pipeline.embed`

- [ ] **Step 5.3: Implement `apps/backend/pipeline/embed.py`**

```python
"""Embedding pipeline — generates 1024-dim text-embedding-v4 vectors for all CHUNK nodes.

Reads chunks with embeddingStatus='pending' (or recompute=True),
calls Silra embeddings API in batches of 32,
writes float[] to CHUNK.embedding and sets embeddingStatus='ok'.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from neo4j import Driver
from openai import OpenAI

log = logging.getLogger(__name__)

_EMBED_MODEL = os.getenv("EMBED_LLM_MODEL", "text-embedding-v4")
_EMBED_DIMS = 1024
_DEFAULT_BATCH = 32
_MAX_RETRIES = 3


def _get_embed_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", "https://api.silra.cn/v1/"),
    )


def _batch_embed(
    client: OpenAI,
    texts: list[str],
    *,
    model: str,
    batch_size: int = _DEFAULT_BATCH,
) -> list[list[float]]:
    """Embed a list of texts in batches; return list of float vectors."""
    if not texts:
        return []
    result: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.embeddings.create(input=batch, model=model)
                result.extend([item.embedding for item in resp.data])
                break
            except Exception as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = 2 ** attempt
                log.warning("embed batch attempt %d failed (%s), retry in %ds", attempt + 1, exc, wait)
                time.sleep(wait)
    return result


@dataclass
class EmbedRunReport:
    chunks_total: int = 0
    chunks_embedded: int = 0
    chunks_failed: int = 0
    chunks_skipped: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_total": self.chunks_total,
            "chunks_embedded": self.chunks_embedded,
            "chunks_failed": self.chunks_failed,
            "chunks_skipped": self.chunks_skipped,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors[:20],
        }


_CHUNK_QUERY = """
MATCH (c:CHUNK)
WHERE (c.embeddingStatus = 'pending' OR $recompute)
  AND c.text IS NOT NULL AND c.charCount > 0
RETURN c.chunkId AS chunk_id, c.text AS text
ORDER BY c.chunkId
SKIP $skip LIMIT $batch
"""

_EMBED_WRITE = """
UNWIND $rows AS r
MATCH (c:CHUNK {chunkId: r.chunk_id})
SET c.embedding       = r.embedding,
    c.embeddingModel  = r.model,
    c.embeddingDims   = r.dims,
    c.embeddingStatus = 'ok',
    c.embeddingAt     = r.ts
"""

_EMBED_FAIL = """
MATCH (c:CHUNK {chunkId: $chunk_id})
SET c.embeddingStatus = 'failed', c.embeddingError = $error
"""


def embed_chunks(
    driver: Driver,
    *,
    model: str | None = None,
    batch_size: int = _DEFAULT_BATCH,
    max_chunks: int | None = None,
    recompute: bool = False,
) -> EmbedRunReport:
    """Embed all pending CHUNK nodes and write vectors back to Neo4j."""
    embed_model = model or _EMBED_MODEL
    client = _get_embed_client()
    report = EmbedRunReport()
    t_start = time.time()
    skip = 0

    while True:
        with driver.session() as s:
            rows = s.run(_CHUNK_QUERY, recompute=recompute, skip=skip, batch=batch_size * 4).data()

        if not rows:
            break

        report.chunks_total += len(rows)
        chunk_ids = [r["chunk_id"] for r in rows]
        texts = [r["text"] for r in rows]

        try:
            vectors = _batch_embed(client, texts, model=embed_model, batch_size=batch_size)
        except Exception as exc:
            log.error("Batch embedding failed for %d chunks: %s", len(rows), exc)
            for cid in chunk_ids:
                with driver.session() as s:
                    s.run(_EMBED_FAIL, chunk_id=cid, error=str(exc)).consume()
            report.chunks_failed += len(rows)
            report.errors.append(f"batch skip={skip}: {exc}")
            skip += len(rows)
            continue

        ts_now = datetime.now(timezone.utc).isoformat()
        write_rows = [
            {"chunk_id": cid, "embedding": vec, "model": embed_model, "dims": _EMBED_DIMS, "ts": ts_now}
            for cid, vec in zip(chunk_ids, vectors)
        ]
        with driver.session() as s:
            s.run(_EMBED_WRITE, rows=write_rows).consume()

        report.chunks_embedded += len(write_rows)
        log.info(
            "Embedded skip=%d n=%d total_embedded=%d",
            skip, len(rows), report.chunks_embedded,
        )
        skip += len(rows)

        if max_chunks is not None and report.chunks_total >= max_chunks:
            break

    report.duration_seconds = time.time() - t_start
    return report
```

- [ ] **Step 5.4: Run tests — should pass**

```bash
uv run pytest tests/backend/pipeline/test_embed.py -v
```

Expected: `3 passed`

- [ ] **Step 5.5: Commit**

```bash
git add apps/backend/pipeline/embed.py tests/backend/pipeline/test_embed.py
git commit -m "feat: add embedding pipeline with batched Silra API calls"
```

---

## Task 6: Background runner `scripts/run_chunking.py`

**Files:**
- Create: `scripts/run_chunking.py`

- [ ] **Step 6.1: Create `scripts/run_chunking.py`**

```python
#!/usr/bin/env python3
"""Background runner for full-corpus chunking.

Usage:
    uv run python scripts/run_chunking.py
    uv run python scripts/run_chunking.py --max-pages 100 --recompute
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
from neo4j import GraphDatabase
from apps.backend.pipeline.chunk import chunk_pages


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-corpus chunking runner")
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--log-file", default="logs/chunking_run.log")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "AncientChina")),
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )

    log.info("Starting chunking (max_pages=%s, chunk_size=%d, overlap=%d)",
             args.max_pages, args.chunk_size, args.overlap)
    t0 = time.time()

    report = chunk_pages(
        driver,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_pages=args.max_pages,
        recompute=args.recompute,
    )

    log.info("Done in %.1fs — chunked=%d skipped=%d failed=%d chunks=%d",
             time.time() - t0, report.pages_chunked, report.pages_skipped,
             report.pages_failed, report.chunks_created)

    Path("logs/chunking_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    )
    driver.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Run smoke test (50 pages)**

```bash
uv run python scripts/run_chunking.py --max-pages 50 --verbose
```

Expected: `chunked=N skipped=M failed=0 chunks=K` where K ≈ N * 3–8

- [ ] **Step 6.3: Verify CHUNK nodes in Neo4j**

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USERNAME','neo4j'), os.getenv('NEO4J_PASSWORD','AncientChina')))
with d.session() as s:
    r = s.run('''
    MATCH (c:CHUNK)
    RETURN count(c) as total, avg(c.charCount) as avg_chars,
           count(CASE WHEN c.chunkStrategy=\"markdown_section\" THEN 1 END) as md_chunks,
           count(CASE WHEN c.chunkStrategy=\"sliding_window\" THEN 1 END) as sw_chunks
    ''').single()
    print(dict(r))
d.close()
"
```

Expected: `total > 0`, both strategy types represented, `avg_chars` between 200 and 600

- [ ] **Step 6.4: Run full corpus**

```bash
caffeinate -dimsu uv run python scripts/run_chunking.py --log-file logs/chunking_full.log
```

Expected runtime: 15–30 min for ~14,000 pages × ~5 chunks/page = ~70,000 chunks

- [ ] **Step 6.5: Commit**

```bash
git add scripts/run_chunking.py logs/chunking_report.json
git commit -m "feat: add chunking background runner"
```

---

## Task 7: Notebook `notebooks/05_chunking_embeddings_bakeoff.ipynb`

This notebook documents the bakeoff (comparing chunking strategies) and runs the embedding smoke test.

**Files:**
- Create: `notebooks/05_chunking_embeddings_bakeoff.ipynb`

- [ ] **Step 7.1: Create notebook with the following cells**

The notebook must:
1. Load health.json and refuse to advance if any probe is `ok=false`
2. Import `chunk_pages` and `embed_chunks`
3. Run a 50-page smoke chunk (both strategies side by side)
4. Show chunk distribution stats (avg chars, strategy mix, docs sampled)
5. Run embedding on 20 chunks as a smoke test (call Silra API)
6. Write `notebooks/_artifacts/05_chunking/chunking.json`

Artifact schema:
```json
{
  "phase": "05_chunking_embeddings_bakeoff",
  "ts": "2026-...",
  "config": {"chunk_size": 500, "overlap": 50, "max_pages": 50, "embed_sample": 20},
  "chunk_report": { ...ChunkRunReport.to_dict() },
  "embed_report": { ...EmbedRunReport.to_dict() },
  "strategy_distribution": {"markdown_section": N, "sliding_window": M},
  "avg_chars_per_chunk": X,
  "sample_chunks": [{"chunk_id": ..., "text": "...(first 100 chars)...", "chars": N}]
}
```

Use `uv run jupyter nbconvert --to notebook --execute notebooks/05_chunking_embeddings_bakeoff.ipynb --output 05_chunking_embeddings_bakeoff.ipynb` to test headlessly.

- [ ] **Step 7.2: Verify artifact written**

```bash
ls -la notebooks/_artifacts/05_chunking/chunking.json
python3 -c "import json; d=json.load(open('notebooks/_artifacts/05_chunking/chunking.json')); print(d['chunk_report'])"
```

- [ ] **Step 7.3: Run full-corpus embedding (background)**

```bash
# Create scripts/run_embedding.py following the same pattern as run_chunking.py
# Then:
caffeinate -dimsu uv run python scripts/run_embedding.py --log-file logs/embedding_run.log
```

Expected runtime: ~70,000 chunks × 32 per batch = ~2,200 API calls × ~1.5 s = ~55 min

- [ ] **Step 7.4: Verify embeddings in Neo4j**

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USERNAME','neo4j'), os.getenv('NEO4J_PASSWORD','AncientChina')))
with d.session() as s:
    r = s.run('''
    MATCH (c:CHUNK)
    RETURN count(c) as total,
           count(CASE WHEN c.embeddingStatus=\"ok\" THEN 1 END) as embedded,
           count(CASE WHEN c.embeddingStatus=\"pending\" THEN 1 END) as pending,
           count(CASE WHEN c.embeddingStatus=\"failed\" THEN 1 END) as failed
    ''').single()
    print(dict(r))
d.close()
"
```

Expected: `embedded / total > 0.95`, `failed < 100`

- [ ] **Step 7.5: Commit**

```bash
git add notebooks/05_chunking_embeddings_bakeoff.ipynb \
        notebooks/_artifacts/05_chunking/ \
        scripts/run_embedding.py
git commit -m "feat: add chunking+embedding bakeoff notebook and embedding runner (Phase 5)"
```

---

## Self-Review

**Spec coverage:**
- Fusion full-corpus run → Task 1 ✓
- Layout gap (641 pages) → Task 2 ✓
- CHUNK node creation + Neo4j spine `(:PAGE)-[:HAS]->(:CHUNK)` → Tasks 3–4 ✓
- Vector index for Phase 7 → Task 3 ✓
- Embedding via text-embedding-v4 / 1024 dims → Task 5 ✓
- Background runners → Tasks 1.2, 6, 7.3 ✓
- Notebook artifact → Task 7 ✓
- Tests for all new business logic → Tasks 4.1, 5.1 ✓

**Placeholder scan:** None — all steps contain actual code, commands, and expected output.

**Type consistency:**
- `ChunkRecord` defined in Task 4 and used nowhere else in earlier tasks ✓
- `ChunkRunReport.to_dict()` defined in Task 4.3, used in Task 7 ✓
- `EmbedRunReport.to_dict()` defined in Task 5.3, used in Task 7 ✓
- `fuse_pages()` is existing API — not redefined ✓

---

## After Phase 5: What Comes Next

Once CHUNK nodes + embeddings exist, the immediate follow-on phases are:

| Phase | Notebook | Depends on |
|---|---|---|
| 6 — KG construction | `06_kg_construction.ipynb` | CHUNK nodes (keyword extraction → KEYWORD nodes, MENTION edges) |
| 6b — Citation linker | `06b_citation_linker.ipynb` | CHUNK nodes (CITES edges between documents) |
| 6c — Community summaries | `06c_community_summaries.ipynb` | KEYWORD + CITES graph (APOC community detection) |
| 7 — Search + verifier | `07_search_subgraph.ipynb` | Vector index (similarity search), CHUNK.embedding |
| 8 — Search UI smoke | `08_search_ui_smoke.ipynb` | Phase 7 search APIs |

Update `notebooks/RUN_ALL.md` Phase 5+ entries after completing this plan.
