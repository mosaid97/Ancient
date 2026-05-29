"""A/B harness for the enhance step.

For each of N representative pages this script:

1. Downloads the raw OCR scan from MinIO.
2. Runs the *full* preprocessing chain twice — once with the new
   ``enhance`` step active and once with it skipped — using
   :func:`apps.backend.pipeline.preprocess.run_preprocess_chain`.
3. Runs PaddleOCR on both finals so we can compare line count, char
   count, and confidence.
4. Persists a side-by-side strip per page to
   ``notebooks/_artifacts/02_preprocessing/ab_enhance/`` and prints a
   summary table.

It does NOT write to Neo4j or MinIO — purely an offline evaluation.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
import numpy as np
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.ocr.paddle import PaddleOCREngine
from apps.backend.pipeline.extract import _download_image
from apps.backend.pipeline.preprocess import run_preprocess_chain
from apps.backend.storage.minio_client import get_minio_client

logging.basicConfig(level=logging.WARNING)

OUT_DIR = _REPO_ROOT / "notebooks" / "_artifacts" / "02_preprocessing" / "ab_enhance"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SAMPLES: list[tuple[str, str]] = [
    (
        "刘俊文_敦煌吐鲁番唐代法制文书考释_1989__7d163c80df::p00000",
        "cover (blue, white text)",
    ),
    (
        "刘俊文_敦煌吐鲁番唐代法制文书考释_1989__7d163c80df::p00237",
        "clean body page",
    ),
    (
        "唐律疏議箋解__3fbb3392d0::p00001",
        "clean body page (another book)",
    ),
    (
        "北京大学中国中古史研究中心_敦煌吐鲁番文献研究论集第3辑__96128943a5::p00632",
        "manuscript photo facsimile",
    ),
    (
        "叶炜_南北朝隋唐官吏分途研究__3172d04ff0::p00000",
        "second cover (different colours)",
    ),
]


def _crop_text_region(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    y0, y1 = int(h * 0.30), int(h * 0.55)
    x0, x1 = int(w * 0.30), int(w * 0.70)
    return img[y0:y1, x0:x1]


def main() -> int:
    load_dotenv()
    driver = get_driver()
    mc = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET_PAGES", "ancient-pages")

    print("[boot] loading PaddleOCR (one-time)…")
    engine = PaddleOCREngine()
    engine.warmup(["ch"])

    rows: list[dict] = []
    for page_id, label in SAMPLES:
        with driver.session() as session:
            row = session.run(
                """
                MATCH (p:PAGE {id: $id})
                RETURN p.imageUri AS image_uri,
                       p.language AS language
                """,
                id=page_id,
            ).single()
        if row is None or not row["image_uri"]:
            print(f"[skip] {page_id} not in graph or missing imageUri")
            continue

        print(f"\n=== {label}  —  {page_id[-50:]}")
        raw = _download_image(mc, bucket, row["image_uri"])
        lang_hint = row["language"] or "zh-classical"

        # baseline: run the chain but SKIP enhance
        t0 = time.monotonic()
        chain_off = run_preprocess_chain(raw, skip={"enhance"})
        final_off = chain_off[-1].image
        dt_off = time.monotonic() - t0

        # treatment: run the chain INCLUDING enhance (default)
        t0 = time.monotonic()
        chain_on = run_preprocess_chain(raw)
        final_on = chain_on[-1].image
        dt_on = time.monotonic() - t0

        enhance_metrics = chain_on[-1].metrics
        print(
            f"  chain off={dt_off:.2f}s  on={dt_on:.2f}s  "
            f"applied={enhance_metrics.get('applied')}  "
            f"reason={enhance_metrics.get('reason', '-')}  "
            f"midtone={enhance_metrics.get('midtone_fraction', 0):.3f}  "
            f"sat={enhance_metrics.get('mean_saturation', 0):.1f}  "
            f"bg={enhance_metrics.get('background_mean_polarity_probe', 0):.1f}"
        )

        # PaddleOCR comparison
        t0 = time.monotonic()
        ocr_off = engine.ocr_page(
            final_off, page_id=f"{page_id}::off", language_hint=lang_hint
        )
        dt_ocr_off = time.monotonic() - t0
        t0 = time.monotonic()
        ocr_on = engine.ocr_page(
            final_on, page_id=f"{page_id}::on", language_hint=lang_hint
        )
        dt_ocr_on = time.monotonic() - t0

        print(
            f"  OCR off: lines={len(ocr_off.lines):3d}  chars={ocr_off.char_count:4d}  "
            f"conf={ocr_off.confidence:.3f}  ({dt_ocr_off:.2f}s)"
        )
        print(
            f"  OCR on : lines={len(ocr_on.lines):3d}  chars={ocr_on.char_count:4d}  "
            f"conf={ocr_on.confidence:.3f}  ({dt_ocr_on:.2f}s)"
        )

        rows.append(
            {
                "page_id": page_id,
                "label": label,
                "applied": bool(enhance_metrics.get("applied", False)),
                "reason": enhance_metrics.get("reason", "-"),
                "midtone": float(enhance_metrics.get("midtone_fraction", 0.0)),
                "saturation": float(enhance_metrics.get("mean_saturation", 0.0)),
                "bg": float(enhance_metrics.get("background_mean_polarity_probe", 0.0)),
                "lines_off": len(ocr_off.lines),
                "chars_off": ocr_off.char_count,
                "conf_off": ocr_off.confidence,
                "lines_on": len(ocr_on.lines),
                "chars_on": ocr_on.char_count,
                "conf_on": ocr_on.confidence,
                "text_off": ocr_off.text,
                "text_on": ocr_on.text,
            }
        )

        # strip: raw | final off | final on  (full page)
        fig, ax = plt.subplots(1, 3, figsize=(18, 8))
        ax[0].imshow(raw[..., ::-1])
        ax[0].set_title("raw")
        ax[0].axis("off")
        ax[1].imshow(final_off[..., ::-1])
        ax[1].set_title("final  (enhance OFF)")
        ax[1].axis("off")
        ax[2].imshow(final_on[..., ::-1])
        applied_tag = (
            f"APPLIED" if enhance_metrics.get("applied") else f"skipped: {enhance_metrics.get('reason')}"
        )
        ax[2].set_title(f"final  (enhance ON — {applied_tag})")
        ax[2].axis("off")
        fig.suptitle(f"{label} — {page_id[-50:]}", fontsize=12)
        out = OUT_DIR / f"page_{page_id.split('::')[-1]}_strip.png"
        plt.savefig(out, bbox_inches="tight", dpi=100)
        plt.close(fig)
        print(f"  saved {out}")

        # second strip: cropped text region
        c_raw = _crop_text_region(raw)
        c_off = _crop_text_region(final_off)
        c_on = _crop_text_region(final_on)
        fig, ax = plt.subplots(1, 3, figsize=(18, 8))
        for a, im, t in (
            (ax[0], c_raw, "raw  (crop)"),
            (ax[1], c_off, "final OFF (crop)"),
            (ax[2], c_on, f"final ON (crop)"),
        ):
            a.imshow(im[..., ::-1])
            a.set_title(t)
            a.axis("off")
        out = OUT_DIR / f"page_{page_id.split('::')[-1]}_crop.png"
        plt.savefig(out, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"  saved {out}")

    # ---- summary table ----
    print("\n" + "=" * 120)
    print(
        f"{'label':32s}  {'applied':9s}  {'reason':22s}  {'midtone':>7s}  "
        f"{'sat':>5s}  {'bg':>5s}  {'ch_off':>6s}  {'ch_on':>6s}  {'Δch':>6s}  "
        f"{'co_off':>6s}  {'co_on':>6s}"
    )
    print("=" * 120)
    total_d_chars = 0
    for r in rows:
        dch = r["chars_on"] - r["chars_off"]
        total_d_chars += dch
        print(
            f"{r['label'][:32]:32s}  "
            f"{'yes' if r['applied'] else 'NO':9s}  "
            f"{r['reason'][:22]:22s}  "
            f"{r['midtone']:7.3f}  "
            f"{r['saturation']:5.1f}  "
            f"{r['bg']:5.1f}  "
            f"{r['chars_off']:6d}  "
            f"{r['chars_on']:6d}  "
            f"{dch:+6d}  "
            f"{r['conf_off']:6.3f}  "
            f"{r['conf_on']:6.3f}"
        )
    print("=" * 120)
    print(f"Total Δchars across {len(rows)} sample pages: {total_d_chars:+d}")
    print(f"Strips saved under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
