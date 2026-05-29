"""Marginalia separation: 天頭/地腳/版心 detection (plan §6 Stage 2).

Traditional Chinese block-print pages carry annotations in well-defined
zones around the body text:

- **天頭** (tiāntóu) — the *top* margin; chapter titles, juan headings.
- **地腳** (dìjiǎo) — the *bottom* margin; folio numbers, printer's marks.
- **版心** (bǎnxīn) — the *central* gutter spine; folio numbers,
  printing-block titles. Only relevant before a successful page split —
  after :mod:`apps.backend.preprocess.page_split` the page is single-folio
  and 版心 sits at the *inner* edge of either left or right folio.

We don't try to fully OCR the marginalia in Phase 2 — that's Phase 3's
job. Phase 2 just detects the *zones* (bounding boxes) so the
orchestrator can persist each as a sibling ``(:PAGE {role:'marginalia'})``
node, and so the OCR engines can see a *body-only* primary page (cleaner
text recognition with the small-font marginal annotations cropped away).

Detection (deterministic, ~10 ms per page):

1. Binarise the page (Otsu) so ink pixels become 1.
2. Build horizontal + vertical projections of the binary image.
3. Find the band of consecutive *low-density* rows at the very top and
   very bottom — the 天頭 and 地腳 bands.
4. Body region is what's left between them.
5. (Optional) detect a vertical band of low-density columns near the
   inner edge for 版心 — disabled by default because the inner edge is
   ambiguous before page splitting.

The orchestrator decides whether to actually crop and persist the
marginalia images based on ``min_band_height`` — single-page modern
reprints have no marginalia bands and we want to leave them alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from apps.backend.preprocess.base import StepResult, ensure_bgr, to_grayscale

DEFAULT_MIN_BAND_HEIGHT: float = 0.03
"""A 天頭/地腳 band must be ≥ this fraction of page height to count.
3% on a 4000-px-tall page = ~120 px (~24 mm at 400 DPI), which is the
typical 天頭 height in 中華書局 影印本."""

DEFAULT_MAX_BAND_HEIGHT: float = 0.20
"""And ≤ this fraction (otherwise we mis-detected a sparse body region as
marginalia)."""

DEFAULT_DENSITY_RATIO: float = 0.30
"""A row counts as "low density" (= margin candidate) when its projection
is ≤ this fraction of the row-wise mean. Lower ⇒ stricter (only very
empty bands are marginalia); higher ⇒ more permissive."""


@dataclass
class MarginaliaRegions:
    """Bounding boxes (in ``image`` coordinates) for each detected margin zone.

    Each rectangle is ``(x, y, w, h)`` in pixel units. ``None`` when the
    band wasn't detected.
    """

    top: tuple[int, int, int, int] | None = None
    bottom: tuple[int, int, int, int] | None = None
    body: tuple[int, int, int, int] = (0, 0, 0, 0)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def has_marginalia(self) -> bool:
        return self.top is not None or self.bottom is not None


def _row_density(binary: np.ndarray) -> np.ndarray:
    """Per-row count of ink pixels (binary image with foreground=1)."""
    return binary.sum(axis=1).astype(np.float32)


def _find_low_density_band(
    density: np.ndarray,
    *,
    from_top: bool,
    min_height: int,
    max_height: int,
    cutoff: float,
) -> tuple[int, int] | None:
    """Find a contiguous low-density band at the top (or bottom) edge.

    Args:
        density: Per-row density values.
        from_top: True ⇒ scan downward from row 0; False ⇒ scan upward from
            the last row.
        min_height / max_height: Permissible band sizes (in rows).
        cutoff: Maximum row density to count as "low density".

    Returns:
        ``(start_row, end_row)`` inclusive band, or ``None``.
    """
    n = density.shape[0]
    if from_top:
        idx = 0
        while idx < n and density[idx] <= cutoff:
            idx += 1
        end_band = idx
        if end_band < min_height or end_band > max_height:
            return None
        return (0, end_band - 1)
    # from bottom
    idx = n - 1
    while idx >= 0 and density[idx] <= cutoff:
        idx -= 1
    start_band = idx + 1
    band_h = n - start_band
    if band_h < min_height or band_h > max_height:
        return None
    return (start_band, n - 1)


def detect_marginalia_regions(
    image: np.ndarray,
    *,
    min_band_height: float = DEFAULT_MIN_BAND_HEIGHT,
    max_band_height: float = DEFAULT_MAX_BAND_HEIGHT,
    density_ratio: float = DEFAULT_DENSITY_RATIO,
) -> MarginaliaRegions:
    """Detect 天頭 / 地腳 / body bounding boxes on a (presumed single-folio) page.

    Args:
        image: BGR uint8 page image.
        min_band_height / max_band_height: Permissible 天頭/地腳 band sizes,
            as fractions of page height.
        density_ratio: Row-density cutoff (relative to mean) for "low density".

    Returns:
        :class:`MarginaliaRegions` with the detected boxes.
    """
    bgr = ensure_bgr(image)
    h, w = bgr.shape[:2]
    gray = to_grayscale(bgr)
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    density = _row_density(binary)

    if density.size == 0 or density.max() == 0:
        return MarginaliaRegions(body=(0, 0, w, h), metrics={"reason": "empty_page"})

    mean_density = float(density.mean())
    cutoff = mean_density * density_ratio
    min_h = max(1, int(h * min_band_height))
    max_h = max(min_h + 1, int(h * max_band_height))

    top_band = _find_low_density_band(
        density, from_top=True, min_height=min_h, max_height=max_h, cutoff=cutoff
    )
    bot_band = _find_low_density_band(
        density, from_top=False, min_height=min_h, max_height=max_h, cutoff=cutoff
    )

    top_box: tuple[int, int, int, int] | None = None
    bot_box: tuple[int, int, int, int] | None = None
    body_top = 0
    body_bottom = h - 1

    if top_band is not None:
        ts, te = top_band
        top_box = (0, ts, w, te - ts + 1)
        body_top = te + 1
    if bot_band is not None:
        bs, be = bot_band
        bot_box = (0, bs, w, be - bs + 1)
        body_bottom = bs - 1

    if body_bottom <= body_top:
        # Sanity: the bands collided. Skip marginalia (keep full page as body).
        return MarginaliaRegions(
            body=(0, 0, w, h),
            metrics={
                "reason": "bands_collided",
                "mean_density": mean_density,
                "cutoff": cutoff,
            },
        )

    body_box = (0, body_top, w, body_bottom - body_top + 1)
    metrics = {
        "mean_density": mean_density,
        "cutoff": cutoff,
        "min_band_height_px": min_h,
        "max_band_height_px": max_h,
        "top_detected": top_box is not None,
        "bottom_detected": bot_box is not None,
    }
    return MarginaliaRegions(
        top=top_box,
        bottom=bot_box,
        body=body_box,
        metrics=metrics,
    )


def separate_marginalia(
    image: np.ndarray,
    *,
    min_band_height: float = DEFAULT_MIN_BAND_HEIGHT,
    max_band_height: float = DEFAULT_MAX_BAND_HEIGHT,
    density_ratio: float = DEFAULT_DENSITY_RATIO,
) -> StepResult:
    """Run :func:`detect_marginalia_regions` and crop body + sibling images.

    Returns:
        :class:`StepResult` with ``step='marginalia'``. The primary
        ``image`` is the body crop; ``extras`` carries ``top`` and/or
        ``bottom`` sibling crops when those bands were detected. When
        nothing was detected the input image is returned unchanged.
    """
    bgr = ensure_bgr(image)
    regions = detect_marginalia_regions(
        bgr,
        min_band_height=min_band_height,
        max_band_height=max_band_height,
        density_ratio=density_ratio,
    )
    params = {
        "min_band_height": min_band_height,
        "max_band_height": max_band_height,
        "density_ratio": density_ratio,
    }
    if not regions.has_marginalia:
        return StepResult(
            image=bgr,
            step="marginalia",
            params=params,
            metrics={**regions.metrics, "regions": regions.to_dict()},
        )

    extras: dict[str, np.ndarray] = {}
    if regions.top is not None:
        x, y, w, h = regions.top
        extras["top"] = bgr[y : y + h, x : x + w].copy()
    if regions.bottom is not None:
        x, y, w, h = regions.bottom
        extras["bottom"] = bgr[y : y + h, x : x + w].copy()

    bx, by, bw, bh = regions.body
    body = bgr[by : by + bh, bx : bx + bw].copy()
    return StepResult(
        image=body,
        step="marginalia",
        params=params,
        metrics={**regions.metrics, "regions": regions.to_dict()},
        extras=extras,
    )
