"""One-off PaddleOCR single-page benchmark.

Picks 3 random preprocessed body pages from MinIO + Neo4j, runs the
:class:`apps.backend.ocr.paddle.PaddleOCREngine` over each, and prints
per-page wall-clock so we can project corpus-wide runtime.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

# Ensure the project root is importable when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.ocr.paddle import PaddleOCREngine
from apps.backend.pipeline.extract import _download_image
from apps.backend.storage.minio_client import get_minio_client


def main() -> None:
    load_dotenv()
    driver = get_driver()
    minio_client = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET_PAGES", "ancient-pages")

    select_cypher = """
    MATCH (p:PAGE)
    WHERE p.mode = 'ocr' AND p.preprocessedImageUri IS NOT NULL
      AND (p.role IS NULL OR p.role = 'body')
    RETURN p.id AS page_id,
           p.documentId AS document_id,
           p.preprocessedImageUri AS image_uri,
           p.language AS language
    ORDER BY rand()
    LIMIT 3
    """
    with driver.session() as session:
        rows = list(session.run(select_cypher))
    if not rows:
        print("No preprocessed pages in the graph; run notebook 02 first.")
        return

    print(f"Benchmarking PaddleOCR on {len(rows)} pages from {bucket}...")
    print("Loading PaddleOCR engine (this triggers model download on first run)...")
    engine = PaddleOCREngine()
    t0 = time.monotonic()
    engine.warmup(langs=["ch"])
    print(f"Warmup (model load) took {time.monotonic() - t0:.1f}s")

    durations: list[float] = []
    char_counts: list[int] = []
    for row in rows:
        page_id = row["page_id"]
        image_uri = row["image_uri"]
        language = row.get("language") or "zh-classical"
        try:
            image = _download_image(minio_client, bucket, image_uri)
        except Exception as exc:  # noqa: BLE001
            print(f"  - {page_id}: download failed: {exc}")
            continue
        h, w = image.shape[:2]
        t = time.monotonic()
        result = engine.ocr_page(image, page_id=page_id, language_hint=language)
        elapsed = time.monotonic() - t
        durations.append(elapsed)
        char_counts.append(result.char_count)
        print(
            f"  - {page_id} ({w}x{h}px, lang={language}): "
            f"{elapsed:.2f}s, {result.char_count} chars, "
            f"conf={result.confidence:.3f}, lines={len(result.lines)}, "
            f"text[:40]={result.text[:40].replace(chr(10), ' / ')!r}"
        )

    if durations:
        median = statistics.median(durations)
        mean = statistics.mean(durations)
        print()
        print(f"PaddleOCR per-page wall-clock — median {median:.2f}s, mean {mean:.2f}s")
        # Get corpus total
        with driver.session() as session:
            total = session.run(
                "MATCH (p:PAGE) WHERE p.mode='ocr' AND p.preprocessedImageUri IS NOT NULL "
                "AND (p.role IS NULL OR p.role='body') RETURN count(*) AS n"
            ).single()["n"]
        print(f"Corpus has {total} preprocessed OCR body pages.")
        for label, sec in (("median", median), ("mean", mean)):
            total_sec = sec * total
            print(
                f"Projected corpus runtime ({label} {sec:.2f}s/page × {total}) "
                f"= {total_sec/3600:.2f}h "
                f"({total_sec/60:.1f}min)"
            )


if __name__ == "__main__":
    main()
