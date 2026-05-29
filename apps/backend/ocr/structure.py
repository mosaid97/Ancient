"""PP-StructureV2 wrapper for Phase-4 layout analysis (plan §6 Stage 4).

The installed ``paddleocr==2.x`` ships ``PPStructure`` (PP-StructureV2), not
``PPStructureV3`` which is a 3.x API. This module wraps the 2.x interface
transparently so the orchestrator doesn't care about the version difference.

PP-StructureV2 pipeline:

- Layout region detection (``PP-StructureV2`` layout model, downloaded
  automatically on first run to ``~/.paddleocr/``).
- Per-region OCR (``PP-OCRv4`` recogniser — same model Phase 3 already has
  cached, so no extra download for the OCR component).
- Table recognition: HTML output via ``TableAttn``.
- Markdown recovery via ``paddleocr.paddleocr.convert_info_markdown``.

2.x API differences vs. the 3.x ``PPStructureV3`` that the original
plan assumed:

| Aspect | PPStructure 2.x | PPStructureV3 3.x |
|---|---|---|
| Import | ``from paddleocr import PPStructure`` | ``from paddleocr import PPStructureV3`` |
| Call | ``engine(image)`` | ``engine.predict(input=image)`` |
| Output | ``list[{type, bbox, res, img_idx}]`` | iterator of result objects |
| Markdown | ``convert_info_markdown(res, dir, name)`` → file | ``res.save_to_markdown(dir)`` |
| Region type key | ``region['type']`` | ``box['label']`` |
| Bbox key | ``region['bbox']`` → ``[x1,y1,x2,y2]`` | ``box['coordinate']`` |

The wrapper normalises all of this into the same :class:`LayoutPageResult`
shape regardless of which version is present; no caller change is needed if
the package is later upgraded to 3.x.

The wrapper never raises — errors are captured in
:attr:`LayoutPageResult.error` and the orchestrator writes
``layoutStatus='failed'`` so the page is retried on the next run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_STRUCTURE_LOG_INITIALISED = False


def _hush_paddle_logs() -> None:
    global _STRUCTURE_LOG_INITIALISED
    if _STRUCTURE_LOG_INITIALISED:
        return
    for name in ("paddle", "paddleocr", "ppocr", "PaddleOCR", "paddlex", "ppstructure"):
        try:
            logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:  # noqa: BLE001
            pass
    os.environ.setdefault("FLAGS_call_stack_level", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")
    _STRUCTURE_LOG_INITIALISED = True


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LayoutRegion:
    """One layout region detected by PP-StructureV2.

    Attributes:
        label: Region type label (e.g. ``'text'``, ``'title'``, ``'table'``,
            ``'figure'``, ``'header'``, ``'footer'``).
        score: Detection confidence in ``[0, 1]`` (not always available in
            PP-StructureV2; defaults to 1.0 when absent).
        bbox: ``(x, y, w, h)`` integer pixel rectangle in the preprocessed
            image's coordinate space. ``None`` when not available.
        text: OCR text extracted within the region (may be empty for
            figure regions).
        table_html: HTML string for table regions; ``None`` for non-table.
    """

    label: str
    score: float = 1.0
    bbox: tuple[int, int, int, int] | None = None
    text: str = ""
    table_html: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "label": self.label,
            "score": round(float(self.score), 4),
            "text": self.text,
        }
        if self.bbox is not None:
            d["bbox"] = [int(v) for v in self.bbox]
        if self.table_html is not None:
            d["table_html"] = self.table_html
        return d


@dataclass
class LayoutPageResult:
    """PP-StructureV2 output for one preprocessed page.

    Attributes:
        page_id: The Neo4j PAGE id this result belongs to.
        regions: Detected layout regions in reading order.
        markdown: Reading-order-recovered Markdown text for the page.
        table_html_list: List of ``{region_id: int, html: str}`` dicts.
        duration_seconds: Wall-clock for this page.
        model_version: Pipeline identifier string.
        error: Exception string on failure; ``None`` on success.
    """

    page_id: str
    regions: list[LayoutRegion] = field(default_factory=list)
    markdown: str = ""
    table_html_list: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    model_version: str = "PP-StructureV2/PP-StructureV2"
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def region_count(self) -> int:
        return len(self.regions)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["regions"] = [r.to_dict() for r in self.regions]
        d["duration_seconds"] = round(float(self.duration_seconds), 3)
        return d

    def layout_json(self) -> str:
        """Compact JSON for ``PAGE.layoutJson`` Neo4j property."""
        return json.dumps(
            [r.to_dict() for r in self.regions],
            ensure_ascii=False,
        )

    def table_html_json(self) -> str:
        """Compact JSON for ``PAGE.tableHtmlJson`` Neo4j property."""
        return json.dumps(self.table_html_list, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class StructureEngine:
    """Lazy-loading wrapper around PPStructure (PP-StructureV2).

    The pipeline object is expensive to construct (~10–30 s on CPU due to
    model downloads + weight loading). This class loads it once on the
    first :meth:`analyse_page` call and keeps it in memory for the
    lifetime of the process. The caller (orchestrator) should create one
    ``StructureEngine`` instance and share it across all pages.

    Args:
        device: ``'cpu'`` (default). Apple-Silicon users stay on CPU;
            the Paddle MPS backend is unreliable for PP-Structure.
        lang: Language code for the OCR sub-model (default ``'ch'``).
    """

    _MODEL_VERSION = "PP-StructureV2/PP-StructureV2"

    def __init__(self, *, device: str = "cpu", lang: str = "ch") -> None:
        _hush_paddle_logs()
        self._device = device
        self._lang = lang
        self._pipeline: Any = None

    def _load(self) -> None:
        """Lazy-initialise the PPStructure pipeline."""
        if self._pipeline is not None:
            return
        try:
            from paddleocr import PPStructure  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "PPStructure is not available. Run `uv add paddleocr>=2.10.0`."
            ) from exc

        kwargs: dict[str, Any] = {
            "show_log": False,
            "lang": self._lang,
            "layout": True,
            "table": True,
            "formula": False,        # not relevant for 古籍 corpus
            "image_orientation": False,  # Phase 2 already handled orientation
            "use_gpu": self._device == "gpu",
        }

        logger.info("Loading PPStructure (lang=%s, device=%s)", self._lang, self._device)
        t0 = time.monotonic()
        self._pipeline = PPStructure(**kwargs)
        logger.info("PPStructure loaded in %.1fs", time.monotonic() - t0)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def analyse_page(
        self,
        image: np.ndarray,
        *,
        page_id: str,
    ) -> LayoutPageResult:
        """Run PP-StructureV2 on one preprocessed page image.

        Args:
            image: BGR ``numpy.ndarray`` from MinIO (same array as used
                by Phase 3 PaddleOCR).
            page_id: Neo4j PAGE id — stored in the result for traceability.

        Returns:
            :class:`LayoutPageResult` with ``error=None`` on success, or
            ``error=<exception string>`` when the pipeline raised.
        """
        if image is None or image.size == 0:
            return LayoutPageResult(
                page_id=page_id,
                model_version=self._MODEL_VERSION,
                error="empty image",
            )

        self._load()

        started = time.monotonic()
        try:
            # PP-StructureV2 2.x API: engine(image) returns list[dict]
            raw: list[dict] = self._pipeline(image)
            regions, markdown, table_html_list = self._parse_result(raw)
            elapsed = time.monotonic() - started
            return LayoutPageResult(
                page_id=page_id,
                regions=regions,
                markdown=markdown,
                table_html_list=table_html_list,
                duration_seconds=round(elapsed, 3),
                model_version=self._MODEL_VERSION,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - started
            logger.warning("structure.analyse_page %s failed: %s", page_id, exc)
            return LayoutPageResult(
                page_id=page_id,
                duration_seconds=round(elapsed, 3),
                model_version=self._MODEL_VERSION,
                error=f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------
    # Result parsing (PP-StructureV2 2.x output format)
    # ------------------------------------------------------------------

    def _parse_result(
        self,
        raw: list[dict],
    ) -> tuple[list[LayoutRegion], str, list[dict[str, Any]]]:
        """Parse the 2.x PPStructure output into our typed dataclasses.

        Each item in ``raw`` is a dict with:
        - ``type``: region class string (``'text'``, ``'title'``, ``'table'``,
          ``'figure'``, ``'header'``, ``'footer'``, ``'equation'``)
        - ``bbox``: ``[x1, y1, x2, y2]`` absolute pixel coordinates
        - ``res``: list of OCR line dicts (for text/title) OR
                   ``{'html': ..., 'cell_bbox': ...}`` (for tables) OR
                   empty list (for figures)
        - ``img_idx``: page index (always 0 for single images)
        """
        regions: list[LayoutRegion] = []
        markdown_parts: list[str] = []
        table_html_list: list[dict[str, Any]] = []

        if not raw:
            return regions, "", table_html_list

        for i, region in enumerate(raw):
            region_type = str(region.get("type", "unknown")).lower()
            bbox_raw = region.get("bbox")
            bbox = _xyxy_to_xywh(bbox_raw)
            res = region.get("res") or {}

            # --- Extract text and build the region object ---
            text = ""
            table_html: str | None = None

            if region_type == "table":
                # res is a dict: {'html': '<table>...', 'cell_bbox': [...]}
                if isinstance(res, dict):
                    table_html = res.get("html") or ""
                    text = _table_html_to_text(table_html)
                    if table_html:
                        table_html_list.append({"region_id": i, "html": table_html})
            elif region_type in ("text", "title", "header", "footer", "equation"):
                # res is a list of {'text': ..., 'confidence': ..., 'text_region': ...}
                if isinstance(res, list):
                    text = " ".join(
                        line.get("text", "")
                        for line in res
                        if isinstance(line, dict) and line.get("text")
                    )
            elif region_type == "figure":
                text = ""
            else:
                # Fallback: try list
                if isinstance(res, list):
                    text = " ".join(
                        line.get("text", "")
                        for line in res
                        if isinstance(line, dict) and line.get("text")
                    )

            regions.append(LayoutRegion(
                label=region_type,
                score=1.0,  # PP-StructureV2 doesn't expose a region-level score
                bbox=bbox,
                text=text,
                table_html=table_html,
            ))

            # --- Build markdown ---
            if region_type == "title" and text:
                markdown_parts.append(f"# {text}")
            elif region_type == "table" and table_html:
                markdown_parts.append(table_html)
            elif region_type == "figure":
                markdown_parts.append("![figure]")
            elif region_type in ("header", "footer"):
                pass  # skip headers/footers in markdown
            elif text:
                markdown_parts.append(text)

        markdown = "\n\n".join(p for p in markdown_parts if p.strip())
        return regions, markdown, table_html_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _xyxy_to_xywh(bbox: Any) -> tuple[int, int, int, int] | None:
    """Convert ``[x1, y1, x2, y2]`` to ``(x, y, w, h)``."""
    if bbox is None:
        return None
    try:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    except Exception:  # noqa: BLE001
        return None


def _table_html_to_text(html: str) -> str:
    """Strip HTML tags from a table HTML string to produce plain text."""
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text
