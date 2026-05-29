"""One-off DeepSeek-OCR single-page benchmark via Silra."""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.llm.silra import get_silra_client
from apps.backend.ocr.silra_deepseek import deepseek_ocr_page
from apps.backend.pipeline.extract import _download_bytes
from apps.backend.storage.minio_client import get_minio_client


def main() -> None:
    load_dotenv()
    n = int(os.getenv("DEEPSEEK_BENCH_N", "2"))
    driver = get_driver()
    minio_client = get_minio_client()
    silra_client = get_silra_client(timeout=180.0)
    bucket = os.getenv("MINIO_BUCKET_PAGES", "ancient-pages")

    with driver.session() as session:
        rows = list(
            session.run(
                """
                MATCH (p:PAGE)
                WHERE p.mode='ocr' AND p.preprocessedImageUri IS NOT NULL
                  AND (p.role IS NULL OR p.role='body')
                RETURN p.id AS page_id,
                       p.preprocessedImageUri AS image_uri,
                       p.language AS language
                ORDER BY rand()
                LIMIT $n
                """,
                n=n,
            )
        )

    if not rows:
        print("No preprocessed body pages in the graph.")
        return

    print(f"Benchmarking DeepSeek-OCR via Silra on {len(rows)} pages...")
    durations: list[float] = []
    char_counts: list[int] = []
    for row in rows:
        page_id = row["page_id"]
        image_uri = row["image_uri"]
        language = row.get("language") or "zh-classical"
        try:
            payload = _download_bytes(minio_client, bucket, image_uri)
        except Exception as exc:  # noqa: BLE001
            print(f"  - {page_id}: download failed: {exc}")
            continue
        t = time.monotonic()
        result = deepseek_ocr_page(
            payload,
            page_id=page_id,
            language_hint=language,
            client=silra_client,
            max_tokens=4096,
            timeout=180.0,
        )
        elapsed = time.monotonic() - t
        durations.append(elapsed)
        char_counts.append(result.char_count)
        snippet = result.text[:60].replace("\n", " / ") if result.text else "(empty)"
        print(
            f"  - {page_id} (lang={language}, {len(payload)//1024} KB): "
            f"{elapsed:.2f}s, {result.char_count} chars, error={result.error}, "
            f"text[:60]={snippet!r}"
        )

    if durations:
        median = statistics.median(durations)
        mean = statistics.mean(durations)
        print()
        print(f"DeepSeek-OCR per-page wall-clock — median {median:.2f}s, mean {mean:.2f}s")
        with driver.session() as session:
            total_pre = session.run(
                "MATCH (p:PAGE) WHERE p.mode='ocr' AND p.preprocessedImageUri IS NOT NULL "
                "AND (p.role IS NULL OR p.role='body') RETURN count(*) AS n"
            ).single()["n"]
            total_all = session.run(
                "MATCH (p:PAGE) WHERE p.mode='ocr' "
                "AND (p.role IS NULL OR p.role='body') RETURN count(*) AS n"
            ).single()["n"]
        print(f"Preprocessed body pages: {total_pre}.  Total OCR body pages: {total_all}.")
        for label, sec in (("median", median), ("mean", mean)):
            for tag, total in (("preprocessed", total_pre), ("all_body", total_all)):
                total_sec = sec * total
                print(
                    f"  Projected ({label} {sec:.2f}s × {total} {tag}) = "
                    f"{total_sec/3600:.2f}h"
                )


if __name__ == "__main__":
    main()
