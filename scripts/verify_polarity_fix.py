"""Re-run the preprocessing chain on a chosen page (default: the cover
that triggered the polarity bug) and dump a side-by-side strip of every
step so the fix can be eyeballed.

Usage::

    PYTHONPATH=. uv run python scripts/verify_polarity_fix.py \\
        --page-id "刘俊文_敦煌吐鲁番唐代法制文书考释_1989__7d163c80df::p00000"

Writes to ``logs/polarity_fix/<page_id_safe>/{strip,each step}.png``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
from dotenv import load_dotenv

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.pipeline.preprocess import _download_image, run_preprocess_chain
from apps.backend.storage.minio_client import get_minio_client


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:80]


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--page-id", required=False,
        default="刘俊文_敦煌吐鲁番唐代法制文书考释_1989__7d163c80df::p00000",
    )
    parser.add_argument("--out-dir", default="logs/polarity_fix")
    args = parser.parse_args()

    driver = get_driver()
    minio_client = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET_PAGES", "ancient-pages")

    with driver.session() as session:
        row = session.run(
            "MATCH (p:PAGE {id: $id}) "
            "RETURN p.imageUri AS image_uri, p.role AS role",
            id=args.page_id,
        ).single()
    if row is None or not row["image_uri"]:
        print(f"Page {args.page_id} not found or has no imageUri.")
        return 1

    out_dir = _REPO_ROOT / args.out_dir / _safe(args.page_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Page    : {args.page_id}")
    print(f"Image   : {row['image_uri']}")
    print(f"Role    : {row['role']}")
    print(f"Out dir : {out_dir}")

    raw = _download_image(minio_client, bucket, row["image_uri"])
    print(f"Raw shape: {raw.shape}")
    cv2.imwrite(str(out_dir / "00_raw.png"), raw)

    results = run_preprocess_chain(raw)

    rows = [("raw", raw)]
    metrics_payload: list[dict] = [{"step": "raw", "params": {}, "metrics": {}}]
    for r in results:
        rows.append((r.step, r.image))
        cv2.imwrite(str(out_dir / f"{r.step}.png"), r.image)
        metrics_payload.append({
            "step": r.step,
            "params": r.params,
            "metrics": r.metrics,
        })

    target_h = 600
    scaled = []
    for label, img in rows:
        h, w = img.shape[:2]
        scale = target_h / max(h, 1)
        rs = cv2.resize(img, (max(1, int(w * scale)), target_h))
        cv2.putText(
            rs, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 2, cv2.LINE_AA,
        )
        scaled.append(rs)
    strip = np.concatenate(scaled, axis=1)
    cv2.imwrite(str(out_dir / "strip.png"), strip)

    print()
    print("Per-step metrics:")
    for entry in metrics_payload:
        if entry["step"] == "raw":
            continue
        m = entry["metrics"]
        applied = m.get("applied", "(n/a)")
        reason = m.get("reason", "")
        extras = {
            k: m[k] for k in (
                "background_std", "paper_pixels_replaced",
                "paper_pixels_preserved", "paper_mean_bgr", "rotated",
                "split", "topDetected", "bottomDetected",
            ) if k in m
        }
        print(f"  - {entry['step']:>12}: applied={applied!r:>5} reason={reason!r:>26} {extras}")

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False)
    )
    print(f"\nWrote {len(rows)} step images + strip.png + metrics.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
