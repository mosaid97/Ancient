"""Double-page split via vertical-projection valley (plan §6 Stage 2).

Many 影印本 scans capture two facing folios on a single image
(the "double-page spread" of a woodblock-printed 古籍). To get clean,
single-page OCR we must split the image at the gutter — the narrow,
mostly-empty vertical strip in the middle.

Strategy (deterministic, ~5 ms per page):

1. Binarise the page (Otsu on grayscale).
2. Sum ink pixels along each column → a 1-D vertical projection
   ``v[x]``.
3. Smooth the projection (rolling mean over a fraction of page width) to
   suppress per-stroke variance.
4. Restrict the search window to the central 40% of the page (the gutter
   never sits in the outer 30% of any scanned spread we've seen).
5. The minimum of the smoothed projection inside the window is the
   gutter — but only call it a split if the minimum is small enough vs
   the mean projection (the page is dense enough to have a real valley).

The result is either:

- A single :class:`StepResult` whose ``extras`` carries ``left`` and
  ``right`` page images (the orchestrator persists each as a sibling
  ``(:PAGE)`` node), or
- A pass-through (``split=False``) when no confident gutter is detected
  (single-folio modern reprint, or a 影印本 where the publisher already
  split the spread).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from apps.backend.preprocess.base import StepResult, ensure_bgr, to_grayscale

DEFAULT_SEARCH_WINDOW: float = 0.40
"""Search the central ``DEFAULT_SEARCH_WINDOW`` fraction of page width for
the gutter (the middle 40% by default). Outside this window we don't
trust a minimum to be the real gutter."""

DEFAULT_VALLEY_RATIO: float = 0.20
"""The smoothed projection valley must be ≤ this fraction of the mean
projection inside the search window for the split to fire. Tighter ⇒
fewer false positives on dense pages; looser ⇒ more aggressive splitting."""

DEFAULT_ASPECT_TRIGGER: float = 1.20
"""Pages narrower than this aspect ratio (width / height) are never split —
even with a strong valley, a near-square page is almost certainly a
single folio."""

DEFAULT_SMOOTH_FRAC: float = 0.01
"""Rolling-mean window for projection smoothing, as a fraction of the
image width. 1% gives ~20 px on a 2000-px-wide page."""


@dataclass
class SplitDecision:
    """Per-page outcome reported by :func:`detect_double_page`."""

    split: bool
    gutter_x: int
    aspect_ratio: float
    valley_value: float
    mean_in_window: float
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "gutterX": self.gutter_x,
            "aspectRatio": round(self.aspect_ratio, 3),
            "valleyValue": round(self.valley_value, 2),
            "meanInWindow": round(self.mean_in_window, 2),
            "reason": self.reason,
        }


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Return the same-shape rolling mean of a 1-D array (edges padded)."""
    if window <= 1:
        return arr.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(arr.astype(np.float32), window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def detect_double_page(
    image: np.ndarray,
    *,
    search_window: float = DEFAULT_SEARCH_WINDOW,
    valley_ratio: float = DEFAULT_VALLEY_RATIO,
    aspect_trigger: float = DEFAULT_ASPECT_TRIGGER,
    smooth_frac: float = DEFAULT_SMOOTH_FRAC,
) -> SplitDecision:
    """Decide whether to split ``image`` and where.

    Returns a :class:`SplitDecision` whose ``split`` flag is ``True`` only
    when (a) the page is wider than ``aspect_trigger`` and (b) the
    smoothed vertical projection has a deep valley in its central
    ``search_window`` band.
    """
    bgr = ensure_bgr(image)
    h, w = bgr.shape[:2]
    aspect = w / max(1, h)

    if aspect < aspect_trigger:
        return SplitDecision(
            split=False,
            gutter_x=w // 2,
            aspect_ratio=aspect,
            valley_value=0.0,
            mean_in_window=0.0,
            reason="aspect_too_narrow",
        )

    gray = to_grayscale(bgr)
    # Otsu: foreground (ink) becomes white, paper black, then sum per column.
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    projection = binary.sum(axis=0).astype(np.float32)
    smooth = _rolling_mean(projection, max(3, int(w * smooth_frac)))

    lo = int(w * (0.5 - search_window / 2))
    hi = int(w * (0.5 + search_window / 2))
    window = smooth[lo:hi]
    if window.size == 0:
        return SplitDecision(
            split=False,
            gutter_x=w // 2,
            aspect_ratio=aspect,
            valley_value=0.0,
            mean_in_window=0.0,
            reason="empty_window",
        )

    mean = float(window.mean())
    if mean <= 1.0:
        # The whole window is essentially empty — there's nothing to project,
        # so the "valley" is meaningless. Treat as no split.
        return SplitDecision(
            split=False,
            gutter_x=lo + int(window.argmin()),
            aspect_ratio=aspect,
            valley_value=float(window.min()),
            mean_in_window=mean,
            reason="window_empty",
        )

    valley_idx = int(window.argmin())
    valley = float(window[valley_idx])
    gutter_x = lo + valley_idx

    if valley > mean * valley_ratio:
        return SplitDecision(
            split=False,
            gutter_x=gutter_x,
            aspect_ratio=aspect,
            valley_value=valley,
            mean_in_window=mean,
            reason="valley_not_deep_enough",
        )

    return SplitDecision(
        split=True,
        gutter_x=gutter_x,
        aspect_ratio=aspect,
        valley_value=valley,
        mean_in_window=mean,
    )


def split_double_page(
    image: np.ndarray,
    *,
    search_window: float = DEFAULT_SEARCH_WINDOW,
    valley_ratio: float = DEFAULT_VALLEY_RATIO,
    aspect_trigger: float = DEFAULT_ASPECT_TRIGGER,
    smooth_frac: float = DEFAULT_SMOOTH_FRAC,
    margin_px: int = 8,
) -> StepResult:
    """Run :func:`detect_double_page` and slice the image at the gutter.

    When ``split=False`` the image is returned unchanged and ``extras`` is
    empty. When ``split=True`` ``extras['left']`` and ``extras['right']``
    carry the two folios (with a small ``margin_px`` keepout around the
    gutter so we don't bleed gutter pixels into either folio); the primary
    ``image`` field carries the *left* folio so the orchestrator's
    "primary final image" is well-defined.
    """
    bgr = ensure_bgr(image)
    decision = detect_double_page(
        bgr,
        search_window=search_window,
        valley_ratio=valley_ratio,
        aspect_trigger=aspect_trigger,
        smooth_frac=smooth_frac,
    )

    params = {
        "search_window": search_window,
        "valley_ratio": valley_ratio,
        "aspect_trigger": aspect_trigger,
        "smooth_frac": smooth_frac,
        "margin_px": margin_px,
    }

    if not decision.split:
        return StepResult(
            image=bgr,
            step="split",
            params=params,
            metrics=decision.to_dict(),
        )

    w = bgr.shape[1]
    cut = decision.gutter_x
    left = bgr[:, : max(1, cut - margin_px)].copy()
    right = bgr[:, min(w, cut + margin_px) :].copy()

    return StepResult(
        image=left,
        step="split",
        params=params,
        metrics=decision.to_dict(),
        extras={"left": left, "right": right},
    )
