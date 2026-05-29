"""Wider gating-survey for the enhance step.

The 5-page A/B harness confirms the gates *can* distinguish covers,
clean pages, and manuscript photos. Before re-processing the corpus we
want evidence the heuristics generalise across many documents and
content types.

This script:

1. Lists every PAGE that's been preprocessed and reads its current
   ``final.png`` from MinIO (so we evaluate the gating on the same
   image PaddleOCR will see in Phase 3, not the raw scan).
2. Samples up to ``SAMPLE_PER_DOC`` pages per document, stratified to
   include the first page (often a cover), one near 25 %, one near 50 %,
   one near 75 %, and one near the end.
3. Calls :func:`enhance_contrast` in "decision-only" mode (it runs the
   gate-cheap probes but the CLAHE/unsharp work only fires when the
   gate passes, which we then time as a side-benefit).
4. Tallies the decision distribution and prints a per-document
   breakdown plus a global summary.
5. For a handful of "apply" candidates with the highest midtone
   fraction (most likely to benefit), runs PaddleOCR with and without
   enhance and records the Δchars / Δconf to confirm the recall lift
   we measured on the single Dunhuang manuscript holds at scale.

Results are written to
``notebooks/_artifacts/02_preprocessing/enhance_survey.json`` and the
representative OCR strips to
``notebooks/_artifacts/02_preprocessing/ab_enhance/``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.ocr.paddle import PaddleOCREngine
from apps.backend.pipeline.extract import _download_image
from apps.backend.preprocess.enhance import enhance_contrast
from apps.backend.storage.minio_client import get_minio_client

logging.basicConfig(level=logging.WARNING)

OUT_DIR = _REPO_ROOT / "notebooks" / "_artifacts" / "02_preprocessing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_DOCS: int = 10
"""How many distinct documents to sample from."""

SAMPLE_PER_DOC: int = 5
"""How many pages to sample per document (5 quintile-spaced)."""

OCR_SUBSET_SIZE: int = 8
"""How many 'apply' candidates to spot-check with PaddleOCR."""

SEED: int = 7
"""Deterministic RNG so the survey is reproducible."""


def _select_doc_sample(driver) -> list[dict]:
    rows: list[dict] = []
    with driver.session() as session:
        docs = [
            r["document_id"]
            for r in session.run(
                """
                MATCH (p:PAGE)
                WHERE p.preprocessedImageUri IS NOT NULL
                  AND p.preprocessingStatus = 'ok'
                  AND (p.role IS NULL OR p.role = 'body')
                RETURN DISTINCT p.documentId AS document_id
                ORDER BY document_id
                """
            )
        ]
        random.seed(SEED)
        random.shuffle(docs)
        for doc_id in docs[:SAMPLE_DOCS]:
            doc_pages = [
                dict(r)
                for r in session.run(
                    """
                    MATCH (p:PAGE)
                    WHERE p.documentId = $doc
                      AND p.preprocessedImageUri IS NOT NULL
                      AND p.preprocessingStatus = 'ok'
                      AND (p.role IS NULL OR p.role = 'body')
                    RETURN p.id AS page_id,
                           p.preprocessedImageUri AS final_uri,
                           p.docPageIndex AS page_index,
                           p.documentId AS document_id,
                           p.language AS language
                    ORDER BY p.docPageIndex
                    """,
                    doc=doc_id,
                )
            ]
            if not doc_pages:
                continue
            n = len(doc_pages)
            indices = sorted({
                0,
                max(0, n // 4),
                max(0, n // 2),
                max(0, 3 * n // 4),
                n - 1,
            })
            for i in indices[:SAMPLE_PER_DOC]:
                rows.append(doc_pages[i])
    return rows


def main() -> int:
    load_dotenv()
    driver = get_driver()
    mc = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET_PAGES", "ancient-pages")

    samples = _select_doc_sample(driver)
    print(f"[boot] surveying {len(samples)} pages across {SAMPLE_DOCS} documents")

    results: list[dict] = []
    reason_counter: Counter[str] = Counter()
    by_doc: dict[str, Counter] = defaultdict(Counter)

    for i, row in enumerate(samples):
        page_id = row["page_id"]
        doc_id = row["document_id"]
        try:
            img = _download_image(mc, bucket, row["final_uri"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {page_id}: {exc}")
            continue

        t0 = time.monotonic()
        step = enhance_contrast(img)
        dt = time.monotonic() - t0

        applied = bool(step.metrics.get("applied", False))
        reason = step.metrics.get("reason", "applied")
        reason_counter[reason] += 1
        by_doc[doc_id][reason] += 1

        results.append(
            {
                "page_id": page_id,
                "document_id": doc_id,
                "page_index": row["page_index"],
                "applied": applied,
                "reason": reason,
                "midtone_fraction": float(step.metrics.get("midtone_fraction", 0.0)),
                "mean_saturation": float(step.metrics.get("mean_saturation", 0.0)),
                "background_mean": float(
                    step.metrics.get("background_mean_polarity_probe", 0.0)
                ),
                "duration_seconds": round(dt, 3),
                "language": row.get("language"),
                "final_uri": row["final_uri"],
            }
        )

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{len(samples):3d}]  applied={applied}  reason={reason}")

    print("\n=== Decision distribution ===")
    for reason, n in reason_counter.most_common():
        pct = n / max(len(results), 1) * 100
        print(f"  {reason:25s}  {n:4d}   ({pct:5.1f}%)")

    print("\n=== Per-document breakdown ===")
    for doc_id, counter in sorted(by_doc.items()):
        total = sum(counter.values())
        parts = " ".join(f"{r}={n}" for r, n in counter.most_common())
        print(f"  {doc_id[-50:]:50s}  n={total}  |  {parts}")

    # ---- spot-check the 'apply' candidates with PaddleOCR ----
    apply_rows = sorted(
        (r for r in results if r["applied"]),
        key=lambda r: r["midtone_fraction"],
        reverse=True,
    )
    apply_rows = apply_rows[:OCR_SUBSET_SIZE]

    if apply_rows:
        print(f"\n=== OCR spot-check on {len(apply_rows)} 'apply' candidates ===")
        engine = PaddleOCREngine()
        engine.warmup(["ch"])

        ocr_records: list[dict] = []
        for r in apply_rows:
            img = _download_image(mc, bucket, r["final_uri"])
            lang = r["language"] or "zh-classical"

            ocr_off = engine.ocr_page(
                img, page_id=f"{r['page_id']}::off", language_hint=lang
            )
            enhanced = enhance_contrast(img).image
            ocr_on = engine.ocr_page(
                enhanced, page_id=f"{r['page_id']}::on", language_hint=lang
            )
            d_chars = ocr_on.char_count - ocr_off.char_count
            d_conf = ocr_on.confidence - ocr_off.confidence
            ocr_records.append(
                {
                    "page_id": r["page_id"],
                    "document_id": r["document_id"],
                    "midtone_fraction": r["midtone_fraction"],
                    "chars_off": ocr_off.char_count,
                    "chars_on": ocr_on.char_count,
                    "delta_chars": d_chars,
                    "delta_chars_pct": (
                        100.0 * d_chars / ocr_off.char_count
                        if ocr_off.char_count > 0
                        else 0.0
                    ),
                    "conf_off": ocr_off.confidence,
                    "conf_on": ocr_on.confidence,
                    "delta_conf": d_conf,
                }
            )
            print(
                f"  {r['page_id'][-50:]:50s}  "
                f"midtone={r['midtone_fraction']:.3f}  "
                f"chars: {ocr_off.char_count:3d}→{ocr_on.char_count:3d} ({d_chars:+3d}, "
                f"{100.0 * d_chars / max(ocr_off.char_count, 1):+5.1f}%)  "
                f"conf: {ocr_off.confidence:.3f}→{ocr_on.confidence:.3f} ({d_conf:+.3f})"
            )

        d_chars_total = sum(o["delta_chars"] for o in ocr_records)
        d_chars_off_total = sum(o["chars_off"] for o in ocr_records)
        ratio = (
            100.0 * d_chars_total / d_chars_off_total if d_chars_off_total else 0.0
        )
        print(
            f"\n  TOTAL: chars {d_chars_off_total} → {d_chars_off_total + d_chars_total}  "
            f"(Δ {d_chars_total:+d}, {ratio:+.1f}%)"
        )
    else:
        ocr_records = []
        print("\n(no 'apply' candidates in sample — gating skipped every page)")

    out_json = OUT_DIR / "enhance_survey.json"
    out_json.write_text(
        json.dumps(
            {
                "sampled_pages": len(results),
                "sample_docs": SAMPLE_DOCS,
                "sample_per_doc": SAMPLE_PER_DOC,
                "seed": SEED,
                "decisions": dict(reason_counter),
                "by_document": {d: dict(c) for d, c in by_doc.items()},
                "ocr_spot_check": ocr_records,
                "page_metrics": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nfull survey saved to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
