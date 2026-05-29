"""Page dewarping (plan §6 Stage 2 "Dewarp via OpenCV; DocUNet optional").

Goal: undo the page curl / perspective that happens when a 影印本 page is
scanned from a bound volume. The text lines start the page straight, then
bow toward the spine, and the marginalia tilts inward.

Two strategies are wired:

1. **Perspective rectification** (default, deterministic, ~10 ms per page).
   Detect the four corners of the page on the scanner background via
   contour finding, then call :func:`cv2.getPerspectiveTransform` to map
   them to the page rectangle. Works well for pages where the bound spine
   isn't dramatically curved.

2. **DocUNet hook** (``method='docunet'``). The plan reserves a slot for a
   learned dewarper (DocUNet / DewarpNet). We don't ship the model in
   Phase 2 (it needs a 200+ MB checkpoint and CPU inference is ~3-5 s per
   page); the hook is a clear extension point for a future GPU phase. The
   call shape is identical so swapping it in later is a one-line change in
   the orchestrator.

If perspective rectification can't confidently find four page corners (e.g.
the page touches all four scan edges — very common for fully-cropped
影印本) the step is a no-op and ``method='noop'`` is reported.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import cv2
import numpy as np

from apps.backend.preprocess.base import StepResult, ensure_bgr, to_grayscale

logger = logging.getLogger(__name__)

DewarpMethod = Literal["perspective", "docunet", "noop"]

DEFAULT_BLUR_KSIZE: int = 5
DEFAULT_BINARIZE_BLOCK: int = 35
DEFAULT_BINARIZE_C: int = 15
DEFAULT_MIN_AREA_FRAC: float = 0.30
"""The detected page contour must cover ≥ this fraction of the image to
count as the page rectangle. Below this we bail out (the contour is
almost certainly a small text region, not the page boundary)."""


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 corner points into top-left, top-right, bottom-right, bottom-left."""

    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left = smallest x+y
    ordered[2] = pts[np.argmax(s)]  # bottom-right = largest x+y
    ordered[1] = pts[np.argmin(d)]  # top-right = smallest y-x
    ordered[3] = pts[np.argmax(d)]  # bottom-left = largest y-x
    return ordered


def _find_page_corners(image: np.ndarray, *, min_area_frac: float) -> np.ndarray | None:
    """Return four (x, y) page corners, or ``None`` if no confident detection."""

    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (DEFAULT_BLUR_KSIZE, DEFAULT_BLUR_KSIZE), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        DEFAULT_BINARIZE_BLOCK,
        DEFAULT_BINARIZE_C,
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    page_area = image.shape[0] * image.shape[1]
    best: tuple[float, np.ndarray] | None = None
    for c in contours:
        area = cv2.contourArea(c)
        if area / page_area < min_area_frac:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and (best is None or area > best[0]):
            best = (area, approx)
    if best is None:
        return None
    return _order_corners(best[1])


def _perspective_warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the quadrilateral defined by ``corners`` into a rectangle."""

    (tl, tr, br, bl) = corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    out_w = int(max(width_top, width_bottom))
    out_h = int(max(height_left, height_right))
    out_w = max(out_w, 100)
    out_h = max(out_h, 100)
    dst = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, M, (out_w, out_h), borderValue=(255, 255, 255))


def dewarp_image(
    image: np.ndarray,
    *,
    method: DewarpMethod = "perspective",
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
) -> StepResult:
    """Apply the chosen dewarping method (default: deterministic perspective).

    Args:
        image: BGR uint8 page image.
        method: ``"perspective"`` (default), ``"docunet"`` (reserved hook —
            raises :class:`NotImplementedError`), or ``"noop"``.
        min_area_frac: Minimum page-contour area fraction.

    Returns:
        :class:`StepResult` with ``step='dewarp'``.
    """
    bgr = ensure_bgr(image)
    metrics: dict[str, Any] = {"method_applied": "noop", "dewarped": False}

    if method == "noop":
        return StepResult(
            image=bgr,
            step="dewarp",
            params={"method": method, "min_area_frac": min_area_frac},
            metrics=metrics,
        )

    if method == "docunet":
        # The DocUNet path is reserved for a future GPU phase. We fail loudly
        # rather than silently degrading to perspective so callers know to
        # plumb the model checkpoint.
        raise NotImplementedError(
            "DocUNet dewarping is reserved for a future GPU phase. "
            "Use method='perspective' (default) for now."
        )

    if method != "perspective":
        raise ValueError(f"unknown dewarp method: {method!r}")

    corners = _find_page_corners(bgr, min_area_frac=min_area_frac)
    if corners is None:
        metrics["reason"] = "no_page_contour"
        return StepResult(
            image=bgr,
            step="dewarp",
            params={"method": method, "min_area_frac": min_area_frac},
            metrics=metrics,
        )

    warped = _perspective_warp(bgr, corners)
    metrics.update(
        {
            "method_applied": "perspective",
            "dewarped": True,
            "input_shape": list(bgr.shape),
            "output_shape": list(warped.shape),
            "corners": corners.tolist(),
        }
    )
    return StepResult(
        image=warped,
        step="dewarp",
        params={"method": method, "min_area_frac": min_area_frac},
        metrics=metrics,
    )
