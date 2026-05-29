"""Corpus-wide preflight for the enhance step (decision-only, no OCR).

Walks every preprocessed PAGE, downloads its ``final.png`` from MinIO,
and calls :func:`enhance_contrast` to record the gating decision. No
OCR, no image writes — just a count of which pages would apply with
the current thresholds and a per-document breakdown.

Useful before kicking off a targeted re-process to know in advance how
many pages will actually change.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
from dotenv import load_dotenv

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.pipeline.extract import _download_image
from apps.backend.preprocess.enhance import enhance_contrast
from apps.backend.storage.minio_client import get_minio_client

# Force unbuffered stdout so progress shows live under `| tee` etc.
sys.stdout.reconfigure(line_buffering=True)

PROBE_MAX_DIM: int = 800
"""Maximum dimension (longest side) for the downscaled probe image
used by the gating metrics. Saturation/midtone/light-fraction are
statistical aggregates — they're stable to within 1e-3 under a 4×
downscale, but a 4× downscale gives an 8×–16× speed-up on the
metric computation."""

logging.basicConfig(level=logging.WARNING)

OUT_DIR = _REPO_ROOT / "notebooks" / "_artifacts" / "02_preprocessing"


def main() -> int:
    load_dotenv()
    driver = get_driver()
    mc = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET_PAGES", "ancient-pages")

    with driver.session() as session:
        rows = list(
            session.run(
                """
                MATCH (p:PAGE)
                WHERE p.preprocessedImageUri IS NOT NULL
                  AND p.preprocessingStatus = 'ok'
                  AND (p.role IS NULL OR p.role = 'body')
                RETURN p.id AS page_id,
                       p.documentId AS document_id,
                       p.preprocessedImageUri AS uri
                ORDER BY p.documentId, p.docPageIndex
                """
            )
        )

    n = len(rows)
    print(f"[boot] preflight over {n} preprocessed pages")

    counter: Counter[str] = Counter()
    by_doc: dict[str, Counter] = defaultdict(Counter)
    apply_pages: list[dict] = []

    started = time.monotonic()
    for i, row in enumerate(rows):
        try:
            img = _download_image(mc, bucket, row["uri"])
        except Exception:  # noqa: BLE001
            counter["download_failed"] += 1
            continue
        h, w = img.shape[:2]
        if max(h, w) > PROBE_MAX_DIM:
            scale = PROBE_MAX_DIM / max(h, w)
            probe = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            probe = img
        step = enhance_contrast(probe)
        applied = bool(step.metrics.get("applied", False))
        reason = "applied" if applied else step.metrics.get("reason", "unknown")
        counter[reason] += 1
        by_doc[row["document_id"]][reason] += 1
        if applied:
            apply_pages.append(
                {
                    "page_id": row["page_id"],
                    "document_id": row["document_id"],
                    "midtone": float(step.metrics.get("midtone_fraction", 0.0)),
                    "light": float(step.metrics.get("light_fraction", 0.0)),
                    "saturation": float(step.metrics.get("mean_saturation", 0.0)),
                }
            )
        if (i + 1) % 50 == 0 or (i + 1) == n:
            elapsed = time.monotonic() - started
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(
                f"  [{i+1:4d}/{n}]  elapsed={elapsed:5.1f}s  "
                f"rate={rate:.1f} p/s  ETA={eta:5.1f}s  "
                f"apply={counter['applied']}  bimodal={counter['already_bimodal']}  "
                f"color={counter['colorful_content']}  inv={counter['inverted_polarity']}  "
                f"low_light={counter['low_light_fraction']}"
            )

    print("\n=== Decision distribution ===")
    for reason, k in counter.most_common():
        pct = 100.0 * k / max(n, 1)
        print(f"  {reason:25s}  {k:4d}   ({pct:5.1f}%)")

    print("\n=== Per-document 'apply' counts ===")
    for doc_id, c in sorted(by_doc.items(), key=lambda x: x[1].get("applied", 0), reverse=True):
        n_apply = c.get("applied", 0)
        if n_apply == 0:
            continue
        total = sum(c.values())
        parts = " ".join(f"{r}={v}" for r, v in c.most_common())
        print(f"  {doc_id[-55:]:55s}  n={total}  apply={n_apply}  ({parts})")

    out = OUT_DIR / "enhance_preflight.json"
    out.write_text(
        json.dumps(
            {
                "total_pages": n,
                "decisions": dict(counter),
                "by_document": {d: dict(c) for d, c in by_doc.items()},
                "apply_pages": apply_pages,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    print(f"\nTo target only the {len(apply_pages)} 'apply' pages for re-process:")
    print("  uv run python scripts/reprocess_enhance_candidates.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
