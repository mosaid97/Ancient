"""Threshold-tuning sweep for the enhance step.

The 43-page survey revealed the default ``bimodal_midtone_fraction=0.05``
is too low: pages in the 0.05–0.30 band carry JPEG halos around clean
printed text, and amplifying them confuses PaddleOCR. The Dunhuang
manuscript that motivated the step has midtone=0.393, which suggests a
much higher cutoff is appropriate.

This script reuses the survey's 'apply' candidates plus the original
manuscript page and sweeps the midtone threshold over
[0.05, 0.15, 0.25, 0.30, 0.40] to find the sweet spot — defined as
the threshold that maximises positive Δchars across the set while
holding regressions to zero.
"""

from __future__ import annotations

import json
import logging
import os
import sys
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

THRESHOLDS = [0.05, 0.15, 0.25, 0.30, 0.40]

PAGES: list[str] = [
    "刘后滨_唐代告身的抄寫舆给付_唐研究第十四卷专号_天聖令及唐宋制度与社会研究__27d6d760eb::p00000",
    "唐耕耦、陆宏基_敦煌社会经济文献真迹释录_第二辑__9c6014dd26::p00394",
    "唐耕耦、陆宏基_敦煌社会经济文献真迹释录_第二辑__9c6014dd26::p00525",
    "唐律疏議箋解__3fbb3392d0::p00019",
    "北京大学中国中古史研究中心_敦煌吐鲁番文献研究论集第3辑__96128943a5::p00632",
]


def main() -> int:
    load_dotenv()
    driver = get_driver()
    mc = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET_PAGES", "ancient-pages")

    pages: list[dict] = []
    with driver.session() as session:
        for pid in PAGES:
            row = session.run(
                """
                MATCH (p:PAGE {id: $id})
                RETURN p.preprocessedImageUri AS uri, p.language AS lang
                """,
                id=pid,
            ).single()
            if row and row["uri"]:
                pages.append(
                    {
                        "page_id": pid,
                        "uri": row["uri"],
                        "language": row["lang"] or "zh-classical",
                    }
                )

    print("[boot] loading PaddleOCR…")
    engine = PaddleOCREngine()
    engine.warmup(["ch"])

    # 1) compute baseline OCR (enhance OFF) once per page.
    baselines: dict[str, dict] = {}
    images: dict[str, "tuple"] = {}
    print("\n=== Baseline (enhance OFF) ===")
    for p in pages:
        img = _download_image(mc, bucket, p["uri"])
        images[p["page_id"]] = img
        probe = enhance_contrast(img)
        ocr = engine.ocr_page(
            img, page_id=f"{p['page_id']}::off", language_hint=p["language"]
        )
        baselines[p["page_id"]] = {
            "midtone": float(probe.metrics["midtone_fraction"]),
            "saturation": float(probe.metrics["mean_saturation"]),
            "chars_off": ocr.char_count,
            "conf_off": ocr.confidence,
        }
        print(
            f"  {p['page_id'][-50:]:50s}  "
            f"midtone={probe.metrics['midtone_fraction']:.3f}  "
            f"chars={ocr.char_count:4d}  conf={ocr.confidence:.3f}"
        )

    # 2) for each threshold, only run enhance where midtone >= threshold,
    #    then OCR the enhanced image and record Δ vs baseline.
    print("\n=== Threshold sweep ===")
    matrix: dict[float, list[dict]] = {t: [] for t in THRESHOLDS}
    for thr in THRESHOLDS:
        print(f"\n--- midtone threshold = {thr:.2f} ---")
        total_d_chars = 0
        total_off = 0
        positives = 0
        regressions = 0
        skipped = 0
        for p in pages:
            base = baselines[p["page_id"]]
            if base["midtone"] < thr:
                matrix[thr].append({**p, **base, "applied": False, "delta": 0})
                skipped += 1
                continue
            step = enhance_contrast(images[p["page_id"]])
            ocr_on = engine.ocr_page(
                step.image,
                page_id=f"{p['page_id']}::on@{thr}",
                language_hint=p["language"],
            )
            d = ocr_on.char_count - base["chars_off"]
            total_d_chars += d
            total_off += base["chars_off"]
            if d > 0:
                positives += 1
            elif d < 0:
                regressions += 1
            matrix[thr].append(
                {
                    "page_id": p["page_id"],
                    "midtone": base["midtone"],
                    "chars_off": base["chars_off"],
                    "chars_on": ocr_on.char_count,
                    "delta": d,
                    "conf_off": base["conf_off"],
                    "conf_on": ocr_on.confidence,
                    "applied": True,
                }
            )
            print(
                f"  {p['page_id'][-50:]:50s}  midtone={base['midtone']:.3f}  "
                f"chars: {base['chars_off']:4d}→{ocr_on.char_count:4d}  "
                f"({d:+4d})  conf {base['conf_off']:.3f}→{ocr_on.confidence:.3f}"
            )
        pct = 100.0 * total_d_chars / max(total_off, 1)
        print(
            f"  SUMMARY thr={thr:.2f}: applied {len(pages)-skipped}/{len(pages)},  "
            f"positives={positives}, regressions={regressions},  "
            f"Δchars={total_d_chars:+d} ({pct:+.1f}%)"
        )

    out = OUT_DIR / "enhance_threshold_sweep.json"
    out.write_text(
        json.dumps(
            {"thresholds": THRESHOLDS, "results": matrix, "baselines": baselines},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
