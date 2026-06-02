---
goal: Phase 4 — Layout Analysis via PP-StructureV3 (structural Markdown + region classification)
version: "1.0"
date_created: "2026-05-19"
last_updated: "2026-05-19"
owner: Ancient China KG project
status: "Planned"
tags: [feature, architecture, layout, phase-4]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

Phase 3 produced flat, linearised text (`textFused`) for every preprocessed page. Phase 4 layers **structural understanding** on top: it runs PP-StructureV3 (already shipped inside the `paddleocr` package) over the same preprocessed images to detect layout regions (text blocks, headings, tables, figures, footnotes, seals, page numbers), recover correct multi-column reading order, and produce a structured Markdown rendition of each page. The Markdown and layout JSON become the primary inputs for Phase 5 (chunking) and make downstream citation linking more precise.

A secondary outcome is the first **page-type classification**: pages that contain only cursive handwritten manuscript script (both OCR engines returned ≤10 chars) are tagged `pageType='manuscript_cursive'` and receive `layoutStatus='skipped'`, so Phase 5 falls back to `textFused` for them and Phase 9 assigns them reduced evidence trust.

PP-StructureV3 is an **additive layer** — it does not replace Phase 3. The fused `textFused` remains the canonical character-level transcription. `structuredMarkdown` provides the structural wrapper that chunking and citation linking consume.

---

## 1. Requirements & Constraints

- **REQ-001**: Run PP-StructureV3 on every PAGE where `mode='ocr'` AND `preprocessedImageUri IS NOT NULL` AND `fusionStatus IN ['ok','single']`.
- **REQ-002**: Idempotent and resumable — `layoutStatus` gate skips already-processed pages unless `recompute=True`.
- **REQ-003**: Outputs follow `camelCase` property naming per AGENTS.md §4.
- **REQ-004**: Disable `use_doc_orientation_classify` and `use_doc_unwarping` — Phase 2 already handled both. Keep `use_textline_orientation=True`.
- **REQ-005**: Table HTML stored separately in `tableHtmlJson` so Phase 5 can emit table-aware chunks.
- **REQ-006**: Pages with `paddleOcrStatus='empty'` AND `deepseekOcrStatus IN ('empty', None)` AND `fusionCharCount <= 10` are pre-classified as `pageType='manuscript_cursive'` and receive `layoutStatus='manuscript'` with no inference.
- **REQ-007**: Pages with `role='cover'` or `role='front_matter'` still run layout analysis (titles, author names, series info live there).
- **REQ-008**: Background runner follows the `scripts/run_paddle_ocr.py` pattern.
- **REQ-009**: No new Python dependency — `PPStructureV3` ships inside `paddleocr>=2.10.0`.
- **REQ-010**: New Neo4j indexes for `PAGE.layoutStatus` and `PAGE.pageType` added to `schema.py`.
- **CON-001**: Apple-Silicon host; PP-StructureV3 defaults to CPU inference. `PP-DocLayout_plus-L` (126 MB) downloads once from PaddlePaddle CDN.
- **CON-002**: Load the `PPStructureV3` pipeline object **once** outside the page loop.
- **CON-003**: Notebooks must not call `driver.close()` per AGENTS.md §11 (2026-05-18 rule).
- **CON-004**: Manuscript pages cannot be usefully processed by PP-StructureV3; `layoutStatus='manuscript'` must be written explicitly.
- **GUD-001**: All Cypher queries parameterised — no string interpolation per AGENTS.md §4.
- **PAT-001**: Mirror the orchestrator pattern from `apps/backend/pipeline/extract.py`.

---

## 2. Implementation Steps

### Implementation Phase 1 — Schema extension

- GOAL-001: Add layout-related indexes to Neo4j schema before any inference runs.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Add `("page_layout_status_index", "PAGE", "layoutStatus")` to `LOOKUP_INDEXES` in `schema.py`. | | |
| TASK-002 | Add `("page_type_index", "PAGE", "pageType")` to `LOOKUP_INDEXES` in `schema.py`. | | |
| TASK-003 | Call `init_schema(driver)` in notebook cell 1 to apply the new indexes. | | |

**New PAGE properties:**

| Property | Type | Values |
|---|---|---|
| `layoutStatus` | string | `'ok'` / `'empty'` / `'failed'` / `'skipped'` / `'manuscript'` |
| `layoutAt` | datetime | `timestamp()` |
| `layoutModelVersion` | string | `'PP-StructureV3/PP-DocLayout_plus-L'` |
| `layoutDurationSeconds` | float | wall-clock |
| `layoutRegionCount` | int | detected bounding boxes |
| `layoutJson` | string | JSON list of `{label, bbox, score}` dicts |
| `structuredMarkdown` | string | reading-order-recovered markdown |
| `tableHtmlJson` | string | JSON list of `{region_id, html}` — `'[]'` when no tables |
| `pageType` | string | `'typeset'` / `'manuscript_cursive'` / `'image_only'` |

### Implementation Phase 2 — PP-StructureV3 wrapper

- GOAL-002: Create `apps/backend/ocr/structure.py` with `StructureEngine` and `LayoutPageResult`.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-004 | Define `LayoutRegion` dataclass: `label`, `score`, `bbox`, `text`, `table_html`. | | |
| TASK-005 | Define `LayoutPageResult` dataclass mirroring `OCRPageResult`: `page_id`, `regions`, `markdown`, `table_html_list`, `duration_seconds`, `model_version`, `error`. | | |
| TASK-006 | Implement `StructureEngine` with lazy-load `_pipeline`. Disable formula/chart/seal sub-pipelines (reduces RAM ~60%). | | |
| TASK-007 | Implement `analyse_page(image, *, page_id)` using `tempfile.mkdtemp()` for markdown output. | | |
| TASK-008 | Export `StructureEngine` and `LayoutPageResult` from `apps/backend/ocr/__init__.py`. | | |

### Implementation Phase 3 — Orchestrator

- GOAL-003: Create `apps/backend/pipeline/layout.py`.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-009 | Define `_SELECT_LAYOUT_PAGES` Cypher query. | | |
| TASK-010 | Define `_LAYOUT_PAGE_UPDATE` Cypher query. | | |
| TASK-011 | Define `LayoutOutcome` and `LayoutRunReport` dataclasses. | | |
| TASK-012 | Implement `_classify_manuscript(row)` heuristic. | | |
| TASK-013 | Implement `run_layout_pages(driver, minio_client, ...)`. | | |
| TASK-014 | Implement `tag_manuscript_page(driver, page_id)` HITL override helper. | | |

### Implementation Phase 4 — Notebook

- GOAL-004: Create `notebooks/04_layout_analysis.ipynb`.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-015 | Bootstrap cell: `load_dotenv`, `get_driver`, `get_minio_client`, `init_schema`, `RUN_FULL=False`, `MAX_PAGES=20`. | | |
| TASK-016 | Smoke demo cell: single known-good page, display regions + markdown. | | |
| TASK-017 | Full run cell: `if RUN_FULL: run_layout_pages(...)`. | | |
| TASK-018 | Artifact cell: write `notebooks/_artifacts/04_layout_analysis/report.json`. | | |
| TASK-019 | Verification + manuscript summary cells. | | |

### Implementation Phase 5 — Background runner

- GOAL-005: Create `scripts/run_layout_analysis.py`.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-020 | argparse CLI: `--max-pages`, `--document-id`, `--recompute`, `--roles`, `--log-file`, `--report-file`, `--verbose`. | | |
| TASK-021 | SIGINT handler saving partial report before exit. | | |
| TASK-022 | JSON report written to `logs/layout_<timestamp>.json`. | | |

---

## 3. Alternatives

- **ALT-001 — Surya**: Adds a heavy new dependency alongside Paddle; no Classical Chinese vertical-text training; duplicates Phase 3's OCR work. Deferred.
- **ALT-002 — RapidDoc**: PP-StructureV3 as ONNX — no benefit when `paddleocr` is already installed.
- **ALT-003 — Kraken + eScriptorium**: Correct long-term solution for handwritten Dunhuang manuscript pages. Deferred to Phase 8. Phase 4 handles these gracefully via `pageType='manuscript_cursive'`.
- **ALT-004 — Run layout on raw images**: Rejected — would misalign layout bboxes with Phase 3 `OCRLine.bbox` coordinates stored in `paddleOcrLinesJson`.
- **ALT-005 — Enable formula/chart/seal in PP-StructureV3**: Deferred — not present in 古籍 corpus; adds ~1.5 GB model weight and ~40% latency overhead on CPU.

---

## 4. Dependencies

- **DEP-001**: `paddleocr>=2.10.0` — already installed; `PPStructureV3` available via `from paddleocr import PPStructureV3`.
- **DEP-002**: Internet on first run to download `PP-DocLayout_plus-L` (126 MB). Offline thereafter.
- **DEP-003**: MinIO running (`docker compose up -d`). Downloads preprocessed images from `ancient-pages` bucket.
- **DEP-004**: Phase 3d fusion complete — `fusionStatus` and `fusionCharCount` must be populated for the manuscript heuristic.

---

## 5. Files

- **FILE-001**: `apps/backend/ocr/structure.py` — NEW.
- **FILE-002**: `apps/backend/ocr/__init__.py` — EDIT.
- **FILE-003**: `apps/backend/pipeline/layout.py` — NEW.
- **FILE-004**: `apps/backend/graph/schema.py` — EDIT.
- **FILE-005**: `notebooks/04_layout_analysis.ipynb` — NEW.
- **FILE-006**: `scripts/run_layout_analysis.py` — NEW.

---

## 6. Testing

- **TEST-001**: Unit test `StructureEngine.analyse_page` — mock `PPStructureV3.predict()`.
- **TEST-002**: Unit test `_classify_manuscript` — 4 edge cases.
- **TEST-003**: Smoke notebook run — `RUN_FULL=False`, `MAX_PAGES=5`, assert artifact written.
- **TEST-004**: Schema test — `page_layout_status_index` and `page_type_index` both `ONLINE` after `init_schema`.

---

## 7. Risks & Assumptions

- **RISK-001**: CPU inference ~2–5s/page × 2,300 pages → ~2–3 hours for a full run.
- **RISK-002**: `PP-DocLayout_plus-L` trained on typeset documents; performance on vertical-column 古籍 scans is untested — smoke demo validates before full run.
- **RISK-003**: `save_to_markdown()` writes to disk; use `tempfile.mkdtemp()` per call and clean up immediately.
- **RISK-004**: Single-threaded orchestrator to avoid OOM on laptop-class RAM.
- **RISK-005**: `_classify_manuscript` heuristic may over-classify sparse pages; `tag_manuscript_page` provides HITL override.
- **ASSUMPTION-001**: `from paddleocr import PPStructureV3` works with installed version; fall back to `from paddleocr.ppstructure import PPStructure` if needed.
- **ASSUMPTION-002**: `res.markdown.get('markdown_texts', '')` returns the page markdown string.

---

## 8. Related Specifications / Further Reading

- [PaddleOCR PP-StructureV3 Usage Tutorial](http://www.paddleocr.ai/v3.0.2/en/version3.x/pipeline_usage/PP-StructureV3.html)
- AGENTS.md §4 (Neo4j conventions), §6 (Docker), §10 (agent rules)
- `apps/backend/pipeline/extract.py` — reference orchestrator pattern
- `apps/backend/ocr/base.py` — `OCRPageResult` shape that `LayoutPageResult` mirrors
