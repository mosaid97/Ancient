"""Shared types for Phase-3 OCR engines (plan §6 Stage 3).

Both PaddleOCR and DeepSeek-OCR adapt their wildly-different native outputs
into the same :class:`OCRPageResult` shape so the fusion module
(:mod:`apps.backend.ocr.fusion`) and the orchestrators don't need to know
which engine produced what.

The minimal contract:

- ``text`` (str): full-page text in reading order — the only field the
  Phase-6 translator strictly needs. For classical-Chinese 古籍, reading
  order is top-to-bottom, right-to-left columns; both engines emit this
  by default for ``lang='ch'``.
- ``lines`` (list[OCRLine]): per-line detection + recognition, with
  bboxes when the engine provides them. Used by the fusion module's
  character-level alignment.
- ``confidence`` (float): aggregate per-page confidence, in ``[0, 1]``.
- ``duration_seconds`` (float): wall-clock for this page (so we can
  estimate corpus-wide runtime in the notebooks).

Per-engine extras (engine version, language hint, etc.) live in
``metadata`` so we don't bloat the canonical dataclass.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

OCREngine = Literal["paddleocr", "deepseek_ocr"]


@dataclass
class OCRLine:
    """One detected text line.

    Attributes:
        text: Recognised characters for this line.
        confidence: ``[0, 1]`` per-line confidence; engines without a
            calibrated score should return their own raw score (we
            calibrate later in Phase 4).
        bbox: Optional ``(x, y, w, h)`` integer pixel rectangle (in the
            preprocessed image's coordinate space). DeepSeek-OCR often
            returns text only — ``None`` in that case.
        order: Position of this line in reading order (0-based).
    """

    text: str
    confidence: float = 1.0
    bbox: tuple[int, int, int, int] | None = None
    order: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "order": int(self.order),
        }
        if self.bbox is not None:
            d["bbox"] = [int(v) for v in self.bbox]
        return d


@dataclass
class OCRPageResult:
    """Single-engine OCR output for one preprocessed page.

    Attributes:
        engine: ``'paddleocr'`` or ``'deepseek_ocr'``.
        model_version: Engine model identifier (e.g. ``'PP-OCRv5/ch'`` or
            ``'deepseek-ocr'``).
        page_id: The Neo4j PAGE id this result belongs to.
        text: Full-page text in reading order.
        lines: Per-line breakdown (may be empty when the engine emits
            only text).
        confidence: Aggregate per-page confidence.
        char_count: Length of ``text`` after stripping whitespace.
        language_hint: Engine-side language flag (``'ch'`` / ``'japan'``).
        duration_seconds: Wall-clock for this page (for runtime
            projections).
        metadata: Free-form per-engine extras (raw response shapes,
            preprocessing-image-uri used, etc.).
        error: If the engine raised, the string-ified exception lives
            here and the orchestrator surfaces it on the report.
    """

    engine: OCREngine
    model_version: str
    page_id: str
    text: str = ""
    lines: list[OCRLine] = field(default_factory=list)
    confidence: float = 0.0
    char_count: int = 0
    language_hint: str | None = None
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.text)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lines"] = [line.to_dict() for line in self.lines]
        d["confidence"] = round(float(self.confidence), 4)
        d["duration_seconds"] = round(float(self.duration_seconds), 3)
        return d


def serialize_lines(lines: list[OCRLine]) -> str:
    """Compact JSON for ``PAGE.<engine>OcrLines`` Neo4j property."""

    return json.dumps([line.to_dict() for line in lines], ensure_ascii=False)
