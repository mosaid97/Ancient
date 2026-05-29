"""Phase-3 dual OCR + character-level fusion (plan §6 Stage 3) and
Phase-4 layout analysis (plan §6 Stage 4).

Three production OCR modules:

- :mod:`apps.backend.ocr.paddle` — PaddleOCR PP-OCRv5 wrapper (CPU-friendly,
  ``lang='ch'`` / ``lang='japan'`` routed per page).
- :mod:`apps.backend.ocr.silra_deepseek` — DeepSeek-OCR via Silra
  (OpenAI-compatible vision endpoint); requires internet.
- :mod:`apps.backend.ocr.fusion` — character-level align-and-vote
  (Needleman-Wunsch style) over the two engines' outputs, producing a
  single ``text_fused`` plus a per-character confidence vector + a
  page-level ``fusion_agreement_rate``.

Phase-4 layout module:

- :mod:`apps.backend.ocr.structure` — PP-StructureV3 wrapper that detects
  layout regions, recovers reading order, and outputs structured Markdown.
  Uses the same ``paddleocr`` package; no additional dependency required.

The orchestrators live under :mod:`apps.backend.pipeline.extract` (per-engine
runners), :mod:`apps.backend.pipeline.fusion` (post-extraction fusion +
authoritative language detection), and :mod:`apps.backend.pipeline.layout`
(Phase-4 layout analysis + page-type classification).
"""

from __future__ import annotations

from apps.backend.ocr.base import (
    OCREngine,
    OCRLine,
    OCRPageResult,
    serialize_lines,
)
from apps.backend.ocr.structure import (
    LayoutPageResult,
    LayoutRegion,
    StructureEngine,
)

__all__ = [
    "OCREngine",
    "OCRLine",
    "OCRPageResult",
    "serialize_lines",
    # Phase-4 layout analysis
    "LayoutPageResult",
    "LayoutRegion",
    "StructureEngine",
]
