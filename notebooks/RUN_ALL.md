# Notebook chain — RUN_ALL

Canonical sequencing for the 24-notebook debugging chain
(`.cursor/plans/ancient-chinese-search-engine_*.plan.md` §1.5).
Every notebook imports production modules from `apps/backend/...`
(no copy-paste forks) and writes its output to
`notebooks/_artifacts/<stage_id>/` so the next notebook can `json.load` it.

## Prerequisites

```bash
# 1. Start the local stack (Phase 0 needs only neo4j + redis + minio).
docker compose up -d neo4j redis minio

# 2. Install Python deps + register the kernel (one-time).
uv sync
uv run python -m ipykernel install --user --name ancient-china \
    --display-name "Ancient China (uv)"

# 3. Open Jupyter Lab from the repo root, OR re-run any notebook headlessly via:
uv run jupyter nbconvert --to notebook --execute \
    notebooks/<NN>_<name>.ipynb --output <NN>_<name>.ipynb
```

## Phase 0 — infra + philological foundation

| # | Notebook | Status | Artifact |
|---|---|---|---|
| 00 | `00_setup_smoke_test.ipynb` | ✅ | `_artifacts/00_setup_smoke_test/health.json` |
| 00a | `00a_philological_seeds.ipynb` | ✅ | `_artifacts/00a_philological_seeds/seeds.json` |

`00_setup_smoke_test.ipynb` is the canary: it pings Silra
(`apps.backend.llm.silra.ping`), Neo4j
(`apps.backend.graph.neo4j_client.ping`), and MinIO
(`apps.backend.storage.minio_client.ping`); brings the schema online via
`apps.backend.graph.schema.init_schema`; and writes `health.json`. Subsequent
notebooks should refuse to advance if any probe in `health.json` is `ok=false`.

`00a_philological_seeds.ipynb` loads + checksums the seven seed files in
`data/seeds/` (異體字 / 避諱×4 / 通假字 / 紀年), exercises every step of the
canonical pipeline (`apps.backend.normalize.normalize_canonical`) on
hand-crafted classical-Chinese fixtures, and verifies the 紀年→CE converter
against eight Tang reign-period anchors plus a deliberate 干支 mismatch.

## Phase 1 — format-aware ingestion + tier + edition metadata

| # | Notebook | Status | Artifact |
|---|---|---|---|
| 01 | `01_ingestion.ipynb` | ✅ | `_artifacts/01_ingestion/ingestion.json` |
| 01b | `01b_language_detection_native.ipynb` | ✅ | `_artifacts/01b_language_detection_native/lang_detect.json` |
| 01c | `01c_tier_loader.ipynb` | ✅ | `_artifacts/01c_tier_loader/inventory.json` |

`01_ingestion.ipynb` exercises the deterministic ingest pipeline: the tier
resolver (`apps.backend.readers.resolve_tier`), the format detector
(`detect_reader`), the three readers (`pdf_reader`, `epub_reader`,
`packed_md`), the Phase-1 structure extractor
(`apps.backend.pipeline.structure.plan_structure`), and the orchestrator
(`apps.backend.pipeline.ingest.ingest_path`). It ingests one fixture per
backend (Tang 笔记小说 EPUB, modern academic PDF, Obsidian-Epub-Importer
packed-md tree), upserts the v2.1 spine
`(:TOPIC)-[:CONTAIN]->(:DOCUMENT)-[:CONSIST_OF]->(:CHAPTER)-[:INCLUDE]->(:SECTION)-[:INCLUDE]->(:PAGE)`
(plan §5), streams scanned-image pages from the EPUB to MinIO, runs the
regex edition-metadata extractor (`extract_from_filename`) plus an opt-in
`deepseek-chat` enrichment (`llm_enrich`), and writes `ingestion.json` for
the bulk loader (01c) to consume.

`01b_language_detection_native.ipynb` tags every native-text `PAGE` with a
language verdict before any downstream stage looks at it. It runs the
deterministic script-class detector
(`apps.backend.lang.detector.detect_language`) over Unicode blocks
(hiragana / katakana / kanji / kanbun marks / latin / digits / CJK punct),
applies six explainable rules (`kunten-marks`, `kana-density`,
`latin-mix`, `modern-digit`, `simplified-sniff`, `classical-default`),
and idempotently writes `PAGE.{language, scriptMix, kuntenMarks,
langConfidence, langDetectionRule, langDetectionAt}` via
`apps.backend.pipeline.lang_detect.detect_pages`. OCR pages are
deliberately skipped — Phase 3 re-detects them post-fusion.

`01c_tier_loader.ipynb` is the bulk equivalent of 01: it walks
`raw/Primary/` and `raw/Secondary/` via
`apps.backend.pipeline.bulk_loader.discover_corpus`, then drives the same
`ingest_path` orchestrator over every `*.pdf` / `*.epub` / `*.packed/`
unit (74 entries today — 15 primary + 59 secondary). The loader is
idempotent (`is_ingested` consults Neo4j and skips already-present
documents), restartable (kill / re-run advances through the corpus
deterministically), and runs LLM edition-metadata enrichment by default
(AGENTS.md §11 rule "Bulk ingest (01c) defaults to LLM-on"). A single
batched `detect_pages` sweep runs once at the end so every brand-new
native page inherits Phase-1b classification without re-walking the
graph per document. The notebook ships in two modes: a 3-doc / 20-page
smoke run (default) and an opt-in `RUN_FULL=True` full-corpus pass
(30-60 min wall-clock, dominated by 册府元龟). The artifact
`inventory.json` is the canonical corpus dashboard.

## Phase 2 — OCR preprocessing

| # | Notebook | Status | Artifact |
|---|---|---|---|
| 02 | `02_preprocessing.ipynb` | ✅ | `_artifacts/02_preprocessing/preprocessing.json` |

`02_preprocessing.ipynb` runs the deterministic 6-step image pipeline
(`apps.backend.preprocess.{deskew, dewarp, illumination, bleed,
page_split, marginalia}`) over every scanned `(:PAGE {mode:'ocr'})`
ingested by Phase 1. Each step is a pure function `np.ndarray ->
StepResult`; the orchestrator
`apps.backend.pipeline.preprocess.preprocess_pages(driver, minio_client)`
downloads the raw scan from MinIO (`ancient-pages/<imageUri>`), runs the
chain, persists every intermediate variant
(`<document_id>/page_<n>/<step>.png`) plus the final body crop
(`<document_id>/page_<n>/final.png`), and writes back to Neo4j:
`PAGE.preprocessedImageUri`, `PAGE.preprocessingProvenance` (JSON,
camelCased per AGENTS.md §4), `PAGE.preprocessingStatus`. Page-split and
marginalia siblings become derived `(:PAGE {role: ...})` nodes linked
via `(:PAGE)-[:DERIVED_PAGE {role}]->(:PAGE)` and re-parented under the
same SECTION so the v2.1 spine stays intact. The notebook ships in two
modes: a 30-page smoke run (default; ~60 s wall-clock) and an opt-in
`RUN_FULL=True` full-corpus pass (~60 min for ~2900 OCR pages on
M-series CPU).

## Phase 3 — dual OCR + character-level fusion

| # | Notebook | Status | Artifact |
|---|---|---|---|
| 03 | `03_dual_extraction.ipynb` | ✅ | `_artifacts/03_dual_extraction/extraction.json` |
| 03b | `03b_fusion.ipynb` | ✅ | `_artifacts/03b_fusion/fusion.json` |

`03_dual_extraction.ipynb` runs two **independent OCR engines** over
every preprocessed body page from Phase 2:

- **PaddleOCR PP-OCRv5** (local, CPU) via
  `apps.backend.ocr.paddle.PaddleOCREngine`, routed per page
  (`lang='ch'` / `lang='japan'`). Works **offline** after first model
  pull (~30 MB cached under `~/.paddleocr/`).
- **DeepSeek-OCR via Silra** via
  `apps.backend.ocr.silra_deepseek.deepseek_ocr_page`, OpenAI-compatible
  chat endpoint with a classical-Chinese or kanbun system prompt.
  Requires **internet**.

Each engine writes back under a distinct camelCase property prefix
(`paddleOcr*` / `deepseekOcr*`) so they can be run independently and in
any order. The orchestrator
`apps.backend.pipeline.extract.{run_paddle_pages, run_deepseek_pages}`
is idempotent and resumable: per-page status is checked so a killed run
picks up where it stopped. The notebook supports an `RUN_PADDLE` /
`RUN_DEEPSEEK` toggle (default: paddle-only) plus the usual smoke vs
`RUN_FULL` modes. Per-page wall-clock is measured and projected to
corpus runtime in the artifact.

**Background runner** for the multi-hour offline pass:

```bash
caffeinate -dimsu uv run python scripts/run_paddle_ocr.py \
    --log-file logs/paddle_ocr.log --verbose
```

`03b_fusion.ipynb` merges the two engines into the authoritative
`textFused` via character-level align-and-vote
(`apps.backend.ocr.fusion.fuse_results`, `difflib.SequenceMatcher`
over the two strings). For each `equal/replace/insert/delete` opcode
the orchestrator picks the higher-confidence engine's text and tags
the segment in `fusionSegmentsJson`. `fusionAgreementRate` = chars in
`equal` spans / total fused chars. After fusion, the language detector
re-classifies each PAGE on the fused text and overwrites the
authoritative `language` / `scriptMix` / `kuntenMarks` /
`langConfidence` / `langDetectionRule` (Phase 1b skipped OCR pages —
this is the version every downstream stage uses). Edge cases: one
engine failed → `fusionSingleEngine = 'paddleocr' | 'deepseek_ocr'`;
both failed → `fusionStatus='failed'`.

## Phase 3c — Layout analysis (PP-StructureV3) [plan addition]

| # | Notebook | Status | Artifact |
|---|---|---|---|
| 03c | `03c_layout_analysis.ipynb` | ✅ (2,301/2,942 pages; 641 catch-up needed) | `_artifacts/03c_layout_analysis/report.json` |

This notebook is not in the canonical plan. It was added to support PP-StructureV3 multi-column reading-order detection as an OCR-pipeline extension (between Phase 3 and Phase 4).

## Phase 4 — Evaluator + Problem Classifier

| # | Notebook | Status | Artifact |
|---|---|---|---|
| 04 | `04_evaluator_problem_classifier.ipynb` | 🔲 not yet run | `_artifacts/04_evaluator/evaluator.json` |

`04_evaluator_problem_classifier.ipynb` runs over every fused OCR page to compute
inter-engine CER, CJK validity ratio, and classify disagreements into a
`PROBLEM_CLASS` code (zh: `RARE_GLYPH`, `DEGRADATION`, `LAYOUT_AMBIGUITY`,
`READING_ORDER`, `ANNOTATION`, `POLYSEMY`, `CULTURAL_REFERENCE`,
`EDITORIAL_VS_SOURCE`; JP: `KANBUN_KUNTEN`, `OKURIGANA`, `HENTAIGANA`,
`MIXED_SCRIPT`). Routes each page to `pass` / `needs_review` / `failed`
and writes `(:PAGE)-[:CLASSIFIED_AS]->(:PROBLEM_CLASS)` edges.

Production modules:
- `apps/backend/agents/evaluator.py` — CER computation + routing
- `apps/backend/agents/problem_classifier.py` — deepseek-chat classifier
- `data/seeds/problem_classes.yaml` — taxonomy + few-shot examples

`04_layout_analysis.ipynb` runs PP-StructureV3 (already shipped inside
`paddleocr`) over every preprocessed body page to detect layout regions,
recover multi-column reading order, and produce `structuredMarkdown` for
each page. Pages where both OCR engines returned ≤ 10 chars are classified
as `pageType='manuscript_cursive'` with `layoutStatus='manuscript'` and
skipped from inference. Background runner: `scripts/run_layout_analysis.py`.
641 pages are still missing layout as of 2026-05-24 — run the catch-up pass
before Phase 5.

## Phase 5 — Span-level HITL + Active Learning

| # | Notebook | Status | Artifact |
|---|---|---|---|
| 05 | `05_hitl_simulation_span.ipynb` | ✅ built (not yet run) | `_artifacts/05_hitl/report.json` |
| 05b | `05b_active_learning.ipynb` | ✅ built (not yet run) | `_artifacts/05b_active_learning/report.json` |

`05_hitl_simulation_span.ipynb` demonstrates span-level HITL corrections.
Samples NEEDS_REVIEW pages from Neo4j, writes `(:CORRECTION)` nodes with
`spanBbox` + `spanCharRange`, and seeds `(:FEW_SHOT_EXAMPLE)` nodes for
OCR engine prompt augmentation. In production, the human reviewer drags a
bounding-box on the disputed region and types the corrected text;
`POST /api/ocr/confirm/{page_id}` drives the same `write_correction` +
`write_few_shot_example` path.

Production modules:
- `apps/backend/feedback/correction_writer.py` — CORRECTION + FSE writes
- `apps/backend/feedback/active_learning.py` — canonical priority formula

`05b_active_learning.ipynb` computes the canonical priority score (Move 2):
`score = α·uncertainty + β·downstream_impact_norm + γ·tier_weight`
(defaults: α=0.40, β=0.40, γ=0.20). Includes a weight-sensitivity ablation
across 4 configurations. Persists `PAGE.priorityScore` for the review UI queue.

## Phase 6 — Translation Agent

| # | Notebook | Status | Artifact |
|---|---|---|---|
| 06 | `06_translation_agent.ipynb` | ✅ built (not yet run) | `_artifacts/06_translation/report.json` |

`06_translation_agent.ipynb` runs the tier-aware translation pipeline:
- **Primary tier**: full word → paragraph → review chain — produces
  `CHUNK.textCanonical`, `CHUNK.textVernacular`, `CHUNK.textVernacularJa` (ja/kanbun)
- **Secondary tier**: normalize + concept extraction only (~70% LLM cost saving)
  → produces `CHUNK.textCanonical`, `CHUNK.semanticConcepts`

Includes seeds for the bilingual Dictionary KB (`data/seeds/dictionary_seed.jsonl`,
55 Tang-era entries) and Norms KB (`data/seeds/norms_seed.jsonl`, 30 translation
rules). Tokenization via jieba (zh) / fugashi+unidic-lite (ja).

Production modules:
- `apps/backend/lang/tokenizer.py` — jieba / fugashi router
- `apps/backend/kb/dictionary.py` — DICTIONARY_ENTRY Cypher RAG
- `apps/backend/kb/norms.py` — NORM KB
- `apps/backend/agents/translation/word.py` — tokenize + normalize + dict lookup
- `apps/backend/agents/translation/paragraph.py` — text_vernacular production
- `apps/backend/agents/translation/review.py` — 3-pass coherence/grammar/fidelity review
- `apps/backend/pipeline/translate.py` — batch orchestrator

## Phase 7 — Chunking + Embedding bakeoff [🔲 simplified v1 exists]

| # | Notebook | Status | Artifact |
|---|---|---|---|
| 07 | `07_chunking_embeddings_bakeoff.ipynb` | ⚠️ simplified v1 (single model, no bakeoff, no text_canonical) | `_artifacts/07_chunking/chunking.json` |
| 07b | `07b_image_embeddings.ipynb` | 🔲 not built | `_artifacts/07b_image_embeddings/image_embed.json` |

> **Note**: The canonical Phase 7 requires a 4-model bakeoff (text-emb-v4 / BGE-M3 / SikuBERT / Qwen3-Embedding) and two embeddings per chunk (`chunk_embedding_classical` on `text_canonical` + `chunk_embedding_vernacular` on `text_vernacular`). The current `07_chunking_embeddings_bakeoff.ipynb` is a single-model simplified run that will be upgraded after Phase 6 (Translation Agent) provides `text_canonical` and `text_vernacular`.

**Blockers before Phase 5 can start:**
1. Full-corpus 3-way fusion (`fuse_pages`) has NOT been run — 0/2,942 OCR pages have `textFused`. Run `scripts/run_fusion.py`.
2. 641 pages still missing layout. Run `scripts/run_layout_analysis.py` catch-up.

`05_chunking_embeddings_bakeoff.ipynb` creates `(:CHUNK)` nodes wired
`(:PAGE)-[:HAS]->(:CHUNK)` in the v2.1 spine. Text source priority:
`structuredMarkdown` (→ `markdown_section` strategy) > `textFused`
(→ `sliding_window`) > `text` (native pages). Default parameters:
chunk_size=500, overlap=50. After chunking, runs `embed_chunks`
(`apps.backend.pipeline.embed`) to generate 1024-dim `text-embedding-v4`
vectors via the Silra API and writes them to `CHUNK.embedding`, enabling
the Neo4j vector index (`chunk_embedding_vector_index`) used by Phase 7
similarity search. Background runners: `scripts/run_chunking.py`,
`scripts/run_embedding.py`.

Full implementation plan: `docs/superpowers/plans/2026-05-24-phase5-fusion-layout-chunking-embeddings.md`

## Phase 8 — KG construction [⚠️ keywords-only v1 exists]

| # | Notebook | Status | Artifact |
|---|---|---|---|
| 08 | `08_kg_construction.ipynb` | ⚠️ keywords-only v1 | `_artifacts/08_kg_construction/kg_construction.json` |
| 08b | `08b_citation_linker_entailment.ipynb` | 🔲 not built | — |
| 08c | `08c_community_summaries.ipynb` | 🔲 not built | — |

> **Note**: The canonical Phase 8 also needs an entailment-based Secondary→Primary citation linker (BGE-reranker-v2-gemma cross-encoder) and GraphRAG-style Leiden community summaries. Only keyword extraction is done so far.

## Phases 9–13 [🔲 not built]

| Phase | Notebook(s) | Depends on |
|---|---|---|
| 9 — search subgraph + verifier | `09_search_subgraph.ipynb`, `09b_verifier_full_normalization.ipynb` | CHUNK.embedding + BM25 + reranker |
| 10 — search UI smoke | `10_search_ui_smoke.ipynb` | Phase 9 search APIs |
| 11 — HITL closed loop + metrics | `11_hitl_closed_loop.ipynb`, `11b_dict_growth_metrics.ipynb` | Phase 5 HITL |
| 12 — bench eval | `12_bench_eval.ipynb` | Phase 9 |
| 13 — baselines + ablations | `13_baselines_and_ablations.ipynb` | Phase 12 |

## Reset / re-run from scratch

```bash
docker compose down -v   # wipes Neo4j + MinIO + Redis volumes
rm -rf notebooks/_artifacts
docker compose up -d neo4j redis minio
uv run jupyter nbconvert --to notebook --execute \
    notebooks/00_setup_smoke_test.ipynb --output 00_setup_smoke_test.ipynb
```
