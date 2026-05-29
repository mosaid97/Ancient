"""PaddleOCR PP-OCRv5 wrapper (plan §6 Stage 3a + 3b).

Two engines per language are kept loaded ("ch" and "japan"); the
wrapper exposes a single :class:`PaddleOCREngine` that routes to the
right one based on a per-page language hint.

PP-OCRv5 ships in two flavours: ``server`` (higher accuracy, heavier
CPU/GPU footprint) and ``mobile`` (4x smaller, ~2x faster). Default
here is ``mobile`` because the Phase-2 preprocessed images are large
(typical 古籍 page is 1500×2200 px @ 200 DPI) and the user is targeting
laptop-class hardware. Override via ``model_size='server'`` when GPU is
available.

PaddleOCR 3.x renamed the entry point from ``PaddleOCR.ocr(...)`` to
``PaddleOCR.predict(input=...)`` and replaced ``use_angle_cls`` with
``use_textline_orientation``. This wrapper introspects the installed
version and uses the correct call shape automatically, so the same
notebook code works against both 2.x and 3.x installs.

Bbox handling: PaddleOCR returns four-point polygons; we convert to the
axis-aligned ``(x, y, w, h)`` rectangle expected by
:class:`apps.backend.ocr.base.OCRLine`. Reading-order sort is
already done internally by PaddleOCR for ``lang='ch'``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterable
from typing import Any

import numpy as np

from apps.backend.ocr.base import OCRLine, OCRPageResult

logger = logging.getLogger(__name__)

_PADDLE_LOCK = threading.Lock()
_PADDLE_LOG_INITIALISED = False


def _hush_paddle_logs() -> None:
    """PaddleOCR is wildly chatty; quiet INFO/DEBUG without losing errors."""

    global _PADDLE_LOG_INITIALISED
    if _PADDLE_LOG_INITIALISED:
        return
    for name in (
        "paddle",
        "paddleocr",
        "ppocr",
        "PaddleOCR",
        "paddlex",
        "ppstructure",
    ):
        try:
            logging.getLogger(name).setLevel(logging.WARNING)
        except Exception:  # noqa: BLE001
            pass
    os.environ.setdefault("FLAGS_call_stack_level", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")
    _PADDLE_LOG_INITIALISED = True


_LANG_ROUTE = {
    "zh-classical": "ch",
    "zh-modern": "ch",
    "ch": "ch",
    "chinese": "ch",
    "ja": "japan",
    "jpn": "japan",
    "japan": "japan",
    "kanbun": "japan",
    # mixed/unknown → default to ch (古籍 corpus is Chinese-dominant)
    "mixed": "ch",
    "unknown": "ch",
    None: "ch",
}


def _poly_to_bbox(poly: Any) -> tuple[int, int, int, int] | None:
    """Convert a 4-point polygon (PaddleOCR shape) to ``(x, y, w, h)``."""

    try:
        arr = np.asarray(poly, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 3:
            xs, ys = arr[:, 0], arr[:, 1]
            x_min = int(round(float(xs.min())))
            y_min = int(round(float(ys.min())))
            w = int(round(float(xs.max() - xs.min())))
            h = int(round(float(ys.max() - ys.min())))
            return (x_min, y_min, max(w, 0), max(h, 0))
    except Exception:  # noqa: BLE001
        return None
    return None


class PaddleOCREngine:
    """Thin lazy-loading wrapper around PaddleOCR with per-language engines.

    Args:
        langs: Iterable of language codes to preload (defaults to
            ``('ch', 'japan')``). Loading is lazy — the model only
            actually loads on first ``ocr_page`` for that language.
        model_size: ``'mobile'`` (default; fast, smaller) or
            ``'server'`` (higher accuracy, GPU-friendly).
        use_textline_orientation: Run the 0°/180° classifier per text
            line. Default ``True`` (古籍 scans are sometimes upside-down).
        device: ``'cpu'`` (default) or ``'gpu'``. Apple-Silicon users
            stay on ``'cpu'``; the Paddle MPS backend is unreliable.
    """

    def __init__(
        self,
        langs: Iterable[str] = ("ch", "japan"),
        *,
        model_size: str = "mobile",
        use_textline_orientation: bool = True,
        device: str = "cpu",
    ) -> None:
        _hush_paddle_logs()
        self._langs = tuple(langs)
        self._model_size = model_size
        self._use_textline_orientation = use_textline_orientation
        self._device = device
        self._engines: dict[str, Any] = {}
        # Detected on first use.
        self._uses_predict_api: bool | None = None
        self._paddleocr_version: str | None = None

    # ------------------------------------------------------------------
    # Engine acquisition
    # ------------------------------------------------------------------

    def _detect_api(self) -> None:
        if self._uses_predict_api is not None:
            return
        try:
            import paddleocr as _po  # noqa: F401
            from paddleocr import PaddleOCR

            self._paddleocr_version = getattr(_po, "__version__", "unknown")
            # 3.x replaced .ocr() with .predict()
            self._uses_predict_api = hasattr(PaddleOCR, "predict") and not hasattr(
                PaddleOCR, "ocr_one_image"
            )
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Run `uv add paddleocr paddlepaddle` "
                "(see notebooks/03_dual_extraction.ipynb for the GPU variant)."
            ) from exc

    def _get_engine(self, lang: str) -> Any:
        """Get-or-load a PaddleOCR engine for ``lang`` (``'ch'`` or ``'japan'``)."""

        if lang in self._engines:
            return self._engines[lang]

        self._detect_api()
        from paddleocr import PaddleOCR  # type: ignore[import]

        with _PADDLE_LOCK:
            if lang in self._engines:
                return self._engines[lang]

            kwargs: dict[str, Any] = {"lang": lang}
            if self._uses_predict_api:
                # PaddleOCR 3.x: keep document-level pre/post off (we already
                # did Phase 2 preprocessing), keep the textline orientation
                # classifier on, keep doc-orientation off (古籍 pages are not
                # rotated 90°/270° at this stage).
                kwargs.update(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=self._use_textline_orientation,
                )
                if self._device != "cpu":
                    kwargs["device"] = self._device
            else:
                # PaddleOCR 2.x legacy.
                kwargs.update(
                    use_angle_cls=self._use_textline_orientation,
                    show_log=False,
                    use_gpu=(self._device == "gpu"),
                )
            logger.info(
                "Loading PaddleOCR engine lang=%s version=%s kwargs=%s",
                lang,
                self._paddleocr_version,
                {k: v for k, v in kwargs.items() if k != "lang"},
            )
            engine = PaddleOCR(**kwargs)
            self._engines[lang] = engine
            return engine

    def warmup(self, langs: Iterable[str] | None = None) -> None:
        """Force-load engines (e.g. before a benchmark)."""

        for lang in langs or self._langs:
            self._get_engine(lang)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def ocr_page(
        self,
        image: np.ndarray,
        *,
        page_id: str,
        language_hint: str | None = None,
    ) -> OCRPageResult:
        """OCR one image; choose the engine via ``language_hint``."""

        lang = _LANG_ROUTE.get(language_hint, "ch")
        if lang not in {"ch", "japan"}:
            lang = "ch"
        engine = self._get_engine(lang)

        if image is None or image.size == 0:
            return OCRPageResult(
                engine="paddleocr",
                model_version=f"PP-OCRv5-{self._model_size}/{lang}",
                page_id=page_id,
                text="",
                confidence=0.0,
                language_hint=language_hint,
                duration_seconds=0.0,
                error="empty image",
            )

        started = time.monotonic()
        try:
            if self._uses_predict_api:
                raw = engine.predict(input=image)
            else:
                raw = engine.ocr(image, cls=self._use_textline_orientation)
        except Exception as exc:  # noqa: BLE001 — record & continue
            elapsed = time.monotonic() - started
            logger.warning("paddle.ocr_page %s failed: %s", page_id, exc)
            return OCRPageResult(
                engine="paddleocr",
                model_version=f"PP-OCRv5-{self._model_size}/{lang}",
                page_id=page_id,
                text="",
                confidence=0.0,
                language_hint=language_hint,
                duration_seconds=round(elapsed, 3),
                error=f"{type(exc).__name__}: {exc}",
            )

        lines = _parse_paddle_result(raw, uses_predict=bool(self._uses_predict_api))
        text = "\n".join(line.text for line in lines if line.text.strip())
        conf = (
            float(np.mean([line.confidence for line in lines]))
            if lines else 0.0
        )
        elapsed = time.monotonic() - started

        return OCRPageResult(
            engine="paddleocr",
            model_version=f"PP-OCRv5-{self._model_size}/{lang}",
            page_id=page_id,
            text=text,
            lines=lines,
            confidence=round(conf, 4),
            char_count=len(text.replace("\n", "").strip()),
            language_hint=language_hint,
            duration_seconds=round(elapsed, 3),
            metadata={
                "lang_routed": lang,
                "line_count": len(lines),
                "paddleocr_version": self._paddleocr_version,
            },
        )


def _parse_paddle_result(raw: Any, *, uses_predict: bool) -> list[OCRLine]:
    """Normalise PaddleOCR 2.x and 3.x outputs into a list of OCRLine."""

    lines: list[OCRLine] = []

    # ------- 3.x: PaddleOCR.predict() returns a list of OCRResult-like dicts.
    if uses_predict:
        if not raw:
            return lines
        for page_res in raw:
            res_dict = getattr(page_res, "json", page_res)
            # 3.0+ wraps everything under {'res': {...}}; 3.2+ has the raw dict.
            payload = res_dict.get("res", res_dict) if isinstance(res_dict, dict) else {}
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or []
            polys = payload.get("rec_polys") or payload.get("dt_polys") or []
            n = max(len(texts), len(scores), len(polys))
            for i in range(n):
                txt = texts[i] if i < len(texts) else ""
                if not isinstance(txt, str) or not txt.strip():
                    continue
                score = float(scores[i]) if i < len(scores) else 1.0
                bbox = _poly_to_bbox(polys[i]) if i < len(polys) else None
                lines.append(OCRLine(text=txt, confidence=score, bbox=bbox, order=i))
        return lines

    # ------- 2.x: ocr() returns a list (one per image) of [poly, (text, score)] pairs.
    if not raw:
        return lines
    first = raw[0] if isinstance(raw, list) else raw
    if first is None:
        return lines
    for i, entry in enumerate(first):
        if not entry or len(entry) < 2:
            continue
        poly, rec = entry[0], entry[1]
        try:
            text, score = rec[0], float(rec[1])
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        bbox = _poly_to_bbox(poly)
        lines.append(OCRLine(text=text, confidence=score, bbox=bbox, order=i))
    return lines
