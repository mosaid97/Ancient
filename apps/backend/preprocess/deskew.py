"""Skew detection + rotation correction (plan §6 Stage 2 "Deskew").

We pick a Hough-line strategy (rather than projection-profile maximisation)
because:

- Most Tang-era 影印本 pages are dominated by vertical text columns; the
  strongest line population is the column rules (column borders) running
  near-vertical. Hough exposes the actual angle distribution.
- The plan explicitly calls out "Hough lines" for this step.

The function is conservative: if the detected angle is too small
(``|angle| < min_angle``) or too large (``|angle| > max_angle``) we return
the original image unchanged and record ``rotated=False`` in the metrics.
Rotating a 400-DPI page is expensive and a wrong rotation is unrecoverable
downstream, so we'd rather under-correct than misrotate.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np

from apps.backend.preprocess.base import StepResult, ensure_bgr, to_grayscale

logger = logging.getLogger(__name__)


DEFAULT_CANNY_LOW: int = 50
DEFAULT_CANNY_HIGH: int = 150
DEFAULT_HOUGH_THRESHOLD: int = 200
DEFAULT_MIN_ANGLE_DEG: float = 0.2
"""Below this absolute angle the page is treated as already-straight."""
DEFAULT_MAX_ANGLE_DEG: float = 15.0
"""Above this we bail out — the detector almost certainly found the wrong
line population (e.g., picture frames on a frontispiece)."""


def detect_skew_angle(
    image: np.ndarray,
    *,
    canny_low: int = DEFAULT_CANNY_LOW,
    canny_high: int = DEFAULT_CANNY_HIGH,
    hough_threshold: int = DEFAULT_HOUGH_THRESHOLD,
) -> float:
    """Estimate the dominant skew angle in degrees, in [-90, +90].

    The angle returned is the *correction* angle: rotating the page by
    ``-angle`` should straighten it. Returns ``0.0`` if no strong line
    population is detected.

    The implementation:

    1. Convert to grayscale and Canny-edge the page.
    2. Run :func:`cv2.HoughLines` (standard Hough, not probabilistic) to get
       lines in (rho, theta) polar form.
    3. Histogram the line angles modulo 90° (so column rules and baselines
       both contribute), pick the bin with the most lines, and take the
       median of that bin as the dominant angle.

    Args:
        image: BGR uint8 image (H × W × 3) or grayscale (H × W).
        canny_low: Lower Canny edge threshold.
        canny_high: Upper Canny edge threshold.
        hough_threshold: Min number of votes for a line to be returned by Hough.

    Returns:
        Skew angle in degrees, positive = counter-clockwise. ``0.0`` when no
        confident estimate is available.
    """
    gray = to_grayscale(image)
    edges = cv2.Canny(gray, canny_low, canny_high)
    if not np.any(edges):
        return 0.0

    lines = cv2.HoughLines(edges, 1, math.pi / 720.0, hough_threshold)
    if lines is None or len(lines) == 0:
        return 0.0

    # ``theta`` in Hough is the angle between the line's normal and the x-axis
    # (in radians, [0, pi]). A vertical line has theta=0 or theta=pi; a
    # horizontal line has theta=pi/2. We want the deviation from the nearest
    # cardinal direction (0° or 90°), which gives the rotation needed.
    angles_deg: list[float] = []
    for entry in lines[:, 0, :]:
        theta = float(entry[1])
        deg = math.degrees(theta)
        # Fold into [-45, +45] so vertical and horizontal lines vote for the
        # same skew direction.
        deg = ((deg + 45.0) % 90.0) - 45.0
        angles_deg.append(deg)

    if not angles_deg:
        return 0.0

    # Histogram with 1° bins; pick the dominant bin then take its median.
    arr = np.asarray(angles_deg, dtype=np.float32)
    hist, edges_ = np.histogram(arr, bins=np.arange(-45.0, 46.0, 1.0))
    if hist.max() == 0:
        return 0.0
    top_bin = int(hist.argmax())
    lo, hi = edges_[top_bin], edges_[top_bin + 1]
    bin_vals = arr[(arr >= lo) & (arr < hi)]
    if bin_vals.size == 0:
        return 0.0
    return float(np.median(bin_vals))


def rotate_image(image: np.ndarray, angle_deg: float, *, border_value: int = 255) -> np.ndarray:
    """Rotate ``image`` by ``angle_deg`` (counter-clockwise) about its centre.

    The output is the same H×W as the input — corners that fall outside the
    rotated frame are filled with ``border_value`` (default white = 255), so
    we never lose pixels we already had but also never grow the image (which
    would blow up downstream OCR memory).
    """
    arr = ensure_bgr(image)
    h, w = arr.shape[:2]
    centre = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    return cv2.warpAffine(
        arr,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value, border_value, border_value),
    )


def deskew_image(
    image: np.ndarray,
    *,
    min_angle: float = DEFAULT_MIN_ANGLE_DEG,
    max_angle: float = DEFAULT_MAX_ANGLE_DEG,
    canny_low: int = DEFAULT_CANNY_LOW,
    canny_high: int = DEFAULT_CANNY_HIGH,
    hough_threshold: int = DEFAULT_HOUGH_THRESHOLD,
) -> StepResult:
    """Detect skew via Hough lines and rotate if it exceeds ``min_angle``.

    Args:
        image: BGR uint8 page image.
        min_angle: Below this absolute angle the page is treated as
            already-straight (no rotation, ``rotated=False``).
        max_angle: Above this absolute angle the detection is discarded as
            spurious (no rotation, ``rotated=False``, ``reason='abs_angle_too_large'``).
        canny_low / canny_high / hough_threshold: Forwarded to :func:`detect_skew_angle`.

    Returns:
        :class:`StepResult` with ``step='deskew'``.
    """
    bgr = ensure_bgr(image)
    angle = detect_skew_angle(
        bgr,
        canny_low=canny_low,
        canny_high=canny_high,
        hough_threshold=hough_threshold,
    )
    rotated = False
    reason: str | None = None
    out = bgr
    if abs(angle) < min_angle:
        reason = "below_min_angle"
    elif abs(angle) > max_angle:
        reason = "above_max_angle"
        logger.debug("deskew: angle %.3f deg exceeds max %.3f, skipping rotation", angle, max_angle)
    else:
        out = rotate_image(bgr, -angle)
        rotated = True

    metrics: dict[str, Any] = {
        "detected_angle_deg": round(angle, 4),
        "rotated": rotated,
    }
    if reason is not None:
        metrics["reason"] = reason

    return StepResult(
        image=out,
        step="deskew",
        params={
            "min_angle": min_angle,
            "max_angle": max_angle,
            "canny_low": canny_low,
            "canny_high": canny_high,
            "hough_threshold": hough_threshold,
        },
        metrics=metrics,
    )
