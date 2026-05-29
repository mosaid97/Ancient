"""Re-run Phase-2 preprocessing on pages affected by the polarity / multi-modal
bug (the white-text-on-coloured-background failure mode fixed by the
distance-aware bleed + uniformity-aware illumination changes).

Identifies pages by inspecting the existing ``preprocessingProvenance``
JSON for either:

- ``illumination.metrics.background_std >= 35`` — the old algorithm
  applied illumination to a non-uniform background (cover pages, photo
  pages) and over-corrected, AND/OR
- ``illumination.metrics.background_mean < 110`` — the page is
  effectively inverted polarity, where the old bleed algorithm erased
  the light foreground.

Re-runs :func:`apps.backend.pipeline.preprocess.preprocess_page` over
just those pages (default 30–60 s for the whole batch on the current
corpus). Pages not in the affected set are left untouched.

Usage::

    PYTHONPATH=. uv run python scripts/repreprocess_polarity_affected.py
    PYTHONPATH=. uv run python scripts/repreprocess_polarity_affected.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.pipeline.preprocess import preprocess_page
from apps.backend.storage.minio_client import get_minio_client


_SELECT_ROWS = """
MATCH (p:PAGE)
WHERE p.preprocessingProvenance IS NOT NULL
  AND (p.role IS NULL OR p.role = 'body')
RETURN p.id AS page_id,
       p.documentId AS document_id,
       p.chapterId AS chapter_id,
       p.sectionId AS section_id,
       p.docPageIndex AS page_index,
       p.tier AS tier,
       p.imageUri AS image_uri,
       p.language AS language,
       p.preprocessingProvenance AS prov
"""


def _affected(prov_json: str | None, bg_std_limit: float, bg_mean_limit: float) -> dict | None:
    """Return reason dict if the page would change under the new algorithm."""

    if not prov_json:
        return None
    try:
        prov = json.loads(prov_json)
    except json.JSONDecodeError:
        return None
    steps = prov.get("steps") or []
    illum = next((s for s in steps if s.get("step") == "illumination"), {})
    m = illum.get("metrics", {})
    bg_std = m.get("background_std", m.get("backgroundStd"))
    bg_mean = m.get("background_mean", m.get("backgroundMean"))
    reasons: list[str] = []
    if bg_std is not None and bg_std >= bg_std_limit:
        reasons.append(f"bg_std={bg_std:.1f}>={bg_std_limit}")
    if bg_mean is not None and bg_mean < bg_mean_limit:
        reasons.append(f"bg_mean={bg_mean:.1f}<{bg_mean_limit}")
    if not reasons:
        return None
    return {"bg_std": bg_std, "bg_mean": bg_mean, "reasons": reasons}


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List affected pages but don't re-process")
    parser.add_argument("--bg-std-limit", type=float, default=35.0)
    parser.add_argument("--bg-mean-limit", type=float, default=110.0)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    driver = get_driver()
    minio_client = get_minio_client()
    bucket = args.bucket or os.getenv("MINIO_BUCKET_PAGES", "ancient-pages")

    with driver.session() as session:
        all_rows = list(session.run(_SELECT_ROWS))

    affected: list[tuple[dict, dict]] = []
    for row in all_rows:
        reason = _affected(row["prov"], args.bg_std_limit, args.bg_mean_limit)
        if reason is not None:
            affected.append((dict(row), reason))

    print(f"Inspected {len(all_rows)} already-preprocessed pages.")
    print(f"Affected by polarity / non-uniform-background fix: {len(affected)}")
    print(f"  threshold: bg_std >= {args.bg_std_limit} OR bg_mean < {args.bg_mean_limit}")
    print()
    for row, reason in affected[:20]:
        print(f"  {row['page_id'][:70]:<70}  {' & '.join(reason['reasons'])}")
    if len(affected) > 20:
        print(f"  ... +{len(affected) - 20} more")

    if args.dry_run:
        print("\n--dry-run: not re-processing.")
        return 0
    if args.max_pages:
        affected = affected[: args.max_pages]
    if not affected:
        return 0

    print(f"\nRe-processing {len(affected)} pages...")
    started = time.monotonic()
    ok = failed = 0
    for idx, (row, _) in enumerate(affected, start=1):
        outcome = preprocess_page(
            minio_client=minio_client,
            driver=driver,
            page_row=row,
            bucket=bucket,
        )
        if outcome.status in {"ok", "partial"}:
            ok += 1
            print(f"  [{idx:>3}/{len(affected)}] {outcome.page_id[:60]:<60}  {outcome.status} ({outcome.duration_seconds:.1f}s)")
        else:
            failed += 1
            print(f"  [{idx:>3}/{len(affected)}] {outcome.page_id[:60]:<60}  FAILED: {outcome.error}")

    elapsed = time.monotonic() - started
    print()
    print(f"Done: {ok} ok, {failed} failed in {elapsed:.1f}s ({elapsed / max(len(affected), 1):.2f}s/page avg)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
