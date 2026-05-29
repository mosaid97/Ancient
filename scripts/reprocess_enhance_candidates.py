"""Targeted re-process of the pages where enhance would fire.

Reads the preflight artifact (``enhance_preflight.json``), and for
every page where the new gating would apply CLAHE+unsharp, re-runs
:func:`apps.backend.pipeline.preprocess.preprocess_page` with the
default (now stricter) gates. The chain is idempotent — every other
step will reproduce identical bytes — so the only visible effect is:

1. The page's ``final.png`` in MinIO is re-written with the
   enhanced version.
2. The page's ``preprocessingProvenance`` JSON in Neo4j gains a 7th
   step entry (``enhance``) recording the gate metrics and ``applied``.
3. ``preprocessedAt`` is bumped to ``now()``.

For each candidate we also run PaddleOCR before and after to record
the actual recall delta, so we have ground-truth that the apply was
worth it. The before/after metrics go into
``notebooks/_artifacts/02_preprocessing/enhance_reprocess_report.json``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.ocr.paddle import PaddleOCREngine
from apps.backend.pipeline.extract import _download_image
from apps.backend.pipeline.preprocess import preprocess_page
from apps.backend.storage.minio_client import get_minio_client

logging.basicConfig(level=logging.WARNING)
sys.stdout.reconfigure(line_buffering=True)

ART = _REPO_ROOT / "notebooks" / "_artifacts" / "02_preprocessing"
PREFLIGHT = ART / "enhance_preflight.json"
REPORT = ART / "enhance_reprocess_report.json"


def main() -> int:
    load_dotenv()
    if not PREFLIGHT.exists():
        print(f"[fatal] preflight artifact missing: {PREFLIGHT}")
        print("        run `uv run python scripts/preflight_enhance.py` first")
        return 1

    data = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    apply_pages = data.get("apply_pages", [])
    if not apply_pages:
        print("[done] no candidates to re-process")
        return 0

    print(f"[boot] targeted re-process of {len(apply_pages)} pages")

    driver = get_driver()
    mc = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET_PAGES", "ancient-pages")

    print("[boot] loading PaddleOCR (one-time)…")
    engine = PaddleOCREngine()
    engine.warmup(["ch"])

    # ---- 1) capture BEFORE OCR + page metadata for the chain ----
    print("\n=== Before pass (OCR on current final.png) ===")
    before: dict[str, dict] = {}
    page_rows: list[dict] = []
    for cand in apply_pages:
        page_id = cand["page_id"]
        with driver.session() as session:
            row = session.run(
                """
                MATCH (p:PAGE {id: $id})
                RETURN p.id AS page_id,
                       p.documentId AS document_id,
                       p.chapterId AS chapter_id,
                       p.sectionId AS section_id,
                       p.docPageIndex AS page_index,
                       p.tier AS tier,
                       p.imageUri AS image_uri,
                       p.preprocessedImageUri AS final_uri,
                       p.language AS language
                """,
                id=page_id,
            ).single()
        if row is None or not row["final_uri"]:
            print(f"  [skip] {page_id} missing")
            continue
        page_rows.append(dict(row))
        try:
            img = _download_image(mc, bucket, row["final_uri"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [download fail] {page_id}: {exc}")
            continue
        lang = row["language"] or "zh-classical"
        ocr = engine.ocr_page(img, page_id=f"{page_id}::before", language_hint=lang)
        before[page_id] = {"chars": ocr.char_count, "conf": ocr.confidence}
        print(
            f"  {page_id[-50:]:50s}  before chars={ocr.char_count:4d}  "
            f"conf={ocr.confidence:.3f}"
        )

    # ---- 2) re-run the full chain on each (overwrites final.png) ----
    print("\n=== Re-process pass ===")
    re_start = time.monotonic()
    statuses: dict[str, str] = {}
    for row in page_rows:
        pid = row["page_id"]
        t0 = time.monotonic()
        out = preprocess_page(
            minio_client=mc,
            driver=driver,
            page_row=row,
            bucket=bucket,
        )
        statuses[pid] = out.status
        print(
            f"  {pid[-50:]:50s}  status={out.status:7s}  "
            f"({time.monotonic()-t0:5.1f}s)"
        )
    print(f"[done] chain re-ran for {len(page_rows)} pages in {time.monotonic()-re_start:.1f}s")

    # ---- 3) capture AFTER OCR ----
    print("\n=== After pass (OCR on new final.png) ===")
    after: dict[str, dict] = {}
    for row in page_rows:
        pid = row["page_id"]
        with driver.session() as session:
            r = session.run(
                "MATCH (p:PAGE {id: $id}) RETURN p.preprocessedImageUri AS uri",
                id=pid,
            ).single()
        new_uri = r["uri"] if r else None
        if not new_uri:
            continue
        img = _download_image(mc, bucket, new_uri)
        lang = row["language"] or "zh-classical"
        ocr = engine.ocr_page(img, page_id=f"{pid}::after", language_hint=lang)
        after[pid] = {"chars": ocr.char_count, "conf": ocr.confidence}
        print(
            f"  {pid[-50:]:50s}  after  chars={ocr.char_count:4d}  "
            f"conf={ocr.confidence:.3f}"
        )

    # ---- 4) summary ----
    records: list[dict] = []
    d_chars = 0
    d_chars_off = 0
    positives = regressions = neutral = 0
    for pid in before:
        if pid not in after:
            continue
        b = before[pid]
        a = after[pid]
        delta = a["chars"] - b["chars"]
        d_chars += delta
        d_chars_off += b["chars"]
        if delta > 0:
            positives += 1
        elif delta < 0:
            regressions += 1
        else:
            neutral += 1
        records.append(
            {
                "page_id": pid,
                "chars_before": b["chars"],
                "chars_after": a["chars"],
                "delta_chars": delta,
                "conf_before": b["conf"],
                "conf_after": a["conf"],
                "status": statuses.get(pid, "?"),
            }
        )

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    pct = 100.0 * d_chars / max(d_chars_off, 1)
    print(f"Pages re-processed: {len(records)}")
    print(f"OCR Δchars total:   {d_chars:+d}  ({pct:+.1f}% over {d_chars_off} baseline chars)")
    print(f"Positives:          {positives}")
    print(f"Regressions:        {regressions}")
    print(f"Neutral:            {neutral}")

    REPORT.write_text(
        json.dumps(
            {
                "pages_total": len(records),
                "delta_chars_total": d_chars,
                "delta_chars_pct": pct,
                "positives": positives,
                "regressions": regressions,
                "neutral": neutral,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
