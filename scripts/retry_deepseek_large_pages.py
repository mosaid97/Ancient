"""Retry DeepSeek-OCR on pages that returned empty due to oversized images.

These are 47 pages from the Dunhuang collections (敦煌社会经济文献真迹释录,
敦煌吐鲁番文献研究论集) whose preprocessed images are ~4000×3500 px (3.8 MB
PNG base64). The DeepSeek-OCR endpoint likely hit an internal content limit and
returned empty text without raising an error.

Fix: downscale to max 2048 px on the longer dimension before encoding, which
is within the model's recommended input range, then mark deepseekOcrStatus
back to NULL so the standard run_deepseek_pages orchestrator picks them up —
OR process them inline here for a one-shot catch-up.

Usage::

    uv run python scripts/retry_deepseek_large_pages.py
    uv run python scripts/retry_deepseek_large_pages.py --dry-run   # preview only
    uv run python scripts/retry_deepseek_large_pages.py --paddle-chars-min 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

from apps.backend.graph.neo4j_client import get_driver
from apps.backend.llm.silra import get_silra_client
from apps.backend.ocr.silra_deepseek import deepseek_ocr_page
from apps.backend.pipeline.extract import _write_deepseek_result  # noqa: PLC2701
from apps.backend.storage.minio_client import get_minio_client

# Override the default system prompt: same as classical_zh but explicitly
# prohibits Markdown tables which DeepSeek sometimes emits on columnar pages.
_SYSTEM_PROMPT_LARGE_PAGE = (
    "你是一位精通古籍 OCR 的助手。给定一张敦煌文献或唐代古籍页面的图像，"
    "请逐字逐句地识别画面中所有汉字内容，按古籍阅读顺序（从右至左，自上而下）输出。\n\n"
    "严格要求：\n"
    "1. 只输出识别出的原文，**不要翻译、不要解释、不要添加任何注释**。\n"
    "2. 保留所有异体字、避讳字、通假字等原貌，不要现代化或规范化。\n"
    "3. 行与行之间用换行 (\\n) 分隔，列与列之间用空行分隔。\n"
    "4. **绝对不要使用 Markdown 表格格式（|...| 格式）**，直接输出纯文本。\n"
    "5. 若画面边缘有版心、天头、地脚的小字注文，按出现顺序附在正文末尾，"
    "并以「【注】」开头。\n"
    "6. 如果某些字符模糊不清，用「□」代替；不要猜测。\n"
    "7. 如果整页无字，回复「（空页）」。"
)


def _extract_text_from_table(text: str) -> str:
    """Extract plain text from DeepSeek's Markdown table format if present.

    Handles: | 考 | 上 | 下 | ... | -> 考上下...
    """
    import re
    if "|" not in text:
        return text
    chars = re.findall(r"\|\s*([^\s|]+)\s*(?=\|)", text)
    # Filter out the separator row dashes
    chars = [c for c in chars if not re.match(r"^-+$", c)]
    return "".join(chars) if chars else text

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FMT,
                    handlers=[logging.StreamHandler(sys.stdout)])
for _noisy in ("urllib3", "botocore", "boto3", "s3transfer", "minio", "httpx"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_QUERY = """
MATCH (p:PAGE)
WHERE p.mode = 'ocr'
  AND p.deepseekOcrStatus = 'empty'
  AND p.paddleOcrStatus = 'ok'
  AND p.paddleOcrCharCount >= $min_chars
  AND p.preprocessedImageUri IS NOT NULL
RETURN p.id AS page_id,
       p.preprocessedImageUri AS image_uri,
       p.language AS language,
       p.paddleOcrCharCount AS paddle_chars
ORDER BY p.documentId, p.docPageIndex
"""


def _downscale(img: np.ndarray, max_px: int = 2048) -> np.ndarray:
    """Downscale image so the longer side is at most ``max_px``."""
    h, w = img.shape[:2]
    longer = max(h, w)
    if longer <= max_px:
        return img
    scale = max_px / longer
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retry DeepSeek-OCR on oversized-image pages that returned empty"
    )
    parser.add_argument("--paddle-chars-min", type=int, default=50,
                        help="Min Paddle char count to consider retrying (default 50)")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Cap number of pages processed")
    parser.add_argument("--max-px", type=int, default=2048,
                        help="Downscale longer side to this many pixels (default 2048)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List pages without calling the API")
    args = parser.parse_args()

    driver = get_driver()
    mc = get_minio_client()
    client = None if args.dry_run else get_silra_client()

    with driver.session() as s:
        rows = s.run(_QUERY, min_chars=args.paddle_chars_min).data()

    if args.max_pages:
        rows = rows[: args.max_pages]

    print(f"\nPages to retry: {len(rows)} (paddle_chars >= {args.paddle_chars_min})")
    if args.dry_run:
        for r in rows:
            print(f"  [{r['paddle_chars']} ch] {r['page_id']}")
        return 0

    ok = empty = failed = 0
    errors: list[str] = []
    run_start = time.monotonic()

    for i, row in enumerate(rows, 1):
        page_id: str = row["page_id"]
        image_uri: str = row["image_uri"]
        language: str | None = row.get("language") or "zh-classical"

        # Download
        try:
            data = mc.get_object("ancient-pages", image_uri).read()
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("cv2.imdecode returned None")
        except Exception as exc:
            logger.warning("[%d/%d] download failed %s: %s", i, len(rows), page_id, exc)
            failed += 1
            errors.append(f"{page_id}: download: {exc}")
            continue

        orig_shape = img.shape[:2]
        img_scaled = _downscale(img, max_px=args.max_px)
        scaled_shape = img_scaled.shape[:2]
        scaled = orig_shape != scaled_shape

        logger.info("[%d/%d] %s | orig=%s scaled=%s (downscaled=%s)",
                    i, len(rows), page_id, orig_shape, scaled_shape, scaled)

        # Call the API directly so we can use the custom prompt and post-process
        import os, base64
        from apps.backend.ocr.base import OCRLine, OCRPageResult
        from apps.backend.llm.silra import _retry
        model_name = os.getenv("OCR_LLM_MODEL", "deepseek-ocr")
        _, buf = cv2.imencode(".png", img_scaled)
        b64 = base64.b64encode(buf.tobytes()).decode()
        data_uri = f"data:image/png;base64,{b64}"
        t0 = time.monotonic()
        try:
            response = _retry(
                client.chat.completions.create,
                model=model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT_LARGE_PAGE},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "请按要求识别本页面所有字符，不要使用表格格式。"},
                    ]},
                ],
                max_retries=3,
                max_tokens=4096,
                temperature=0.0,
            )
            raw_text = (response.choices[0].message.content or "").strip()
            # Post-process: collapse Markdown table if DeepSeek still used it
            text = _extract_text_from_table(raw_text)
            # Treat "（空页）" as empty
            if text in {"（空页）", "（空頁）"}:
                text = ""
            lines = [OCRLine(text=l.strip(), confidence=1.0, order=i)
                     for i, l in enumerate(text.split("\n")) if l.strip()]
            result = OCRPageResult(
                engine="deepseek_ocr",
                model_version=model_name,
                page_id=page_id,
                text=text,
                lines=lines,
                confidence=0.85 if text else 0.0,
                char_count=len(text),
                language_hint=language,
                duration_seconds=round(time.monotonic() - t0, 3),
                metadata={"prompt_kind": "large_page_retry",
                          "finish_reason": response.choices[0].finish_reason,
                          "original_shape": list(orig_shape),
                          "scaled_shape": list(scaled_shape)},
            )
        except Exception as exc:
            result = OCRPageResult(
                engine="deepseek_ocr",
                model_version=os.getenv("OCR_LLM_MODEL", "deepseek-ocr"),
                page_id=page_id,
                text="",
                confidence=0.0,
                char_count=0,
                language_hint=language,
                duration_seconds=round(time.monotonic() - t0, 3),
                error=f"{type(exc).__name__}: {exc}",
            )

        _write_deepseek_result(driver, page_id, result)

        if result.error:
            logger.warning("  failed: %s", result.error)
            failed += 1
            errors.append(f"{page_id}: {result.error}")
        elif not result.text:
            logger.info("  still empty after downscale (Dunhuang handwriting)")
            empty += 1
        else:
            logger.info("  ok: %d chars (was empty before)", result.char_count)
            ok += 1

        elapsed = time.monotonic() - run_start
        avg = elapsed / i
        eta = avg * (len(rows) - i)
        print(f"  [{i}/{len(rows)}] ok={ok} empty={empty} failed={failed} "
              f"| {avg:.1f}s/page | ETA ~{eta/60:.0f} min", flush=True)

    elapsed = time.monotonic() - run_start
    print(f"\nDONE — ok={ok} still_empty={empty} failed={failed} "
          f"in {elapsed/60:.1f} min")

    report = {
        "pages_total": len(rows),
        "pages_ok": ok,
        "pages_still_empty": empty,
        "pages_failed": failed,
        "duration_minutes": round(elapsed / 60, 1),
        "errors": errors,
    }
    report_path = _REPO_ROOT / "logs" / "deepseek_large_retry_report.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
