"""Contrast + edge enhancement step (Phase-2 step 7).

Plan §6 Stage 2 leaves room for a final "make-the-glyphs-OCRable" pass.
The bleed/illumination/deskew chain only *cleans* a page — it does not
*enhance* the foreground. On clean printed body pages that is fine
because they are already nearly bimodal (paper ≈ 250, ink ≈ 30). On
photo facsimiles of ancient manuscripts (黑白照片版) the chain leaves a
gray, low-contrast image that PaddleOCR struggles with — empirically we
measured ~2× character recall on a Dunhuang manuscript photo after
applying CLAHE + a gentle unsharp mask.

This module exposes :func:`enhance_contrast`, the seventh and final step
of :func:`apps.backend.pipeline.preprocess.run_preprocess_chain`. It is
intentionally **conservatively gated** so it never fires on pages where
it would do harm (covers with white-on-blue, colour photo plates,
already-pristine body pages):

* **Inverted-polarity** pages → skip (would push the dark background
  even darker and lose the light foreground).
* **Colourful** pages (mean HSV saturation > ``COLOR_SATURATION_LIMIT``)
  → skip (CLAHE on a colour cover oversaturates and can shift hues even
  when run on the L channel because of LAB↔BGR round-trip quantisation).
* **Already-bimodal** pages (midtone fraction below
  ``BIMODAL_MIDTONE_FRACTION``) → skip (nothing for CLAHE to spread).

Where it does fire, the algorithm is:

1. Convert BGR → LAB and split the L channel.
2. Apply OpenCV CLAHE with ``clipLimit`` / ``tileGridSize`` (defaults
   tuned on the corpus: ``clip=3.0``, ``tile=8`` — strong enough to
   reveal faded ink, mild enough not to ring on JPEG edges).
3. Merge L back and convert LAB → BGR.
4. Optional unsharp mask (Gaussian blur + weighted subtraction) with
   ``radius=1.2`` and ``amount=0.5`` (subtle stroke crispening).

The result respects the :class:`StepResult` contract used by every
other step in the pipeline and records its gating decision in
``metrics["reason"]`` so the audit notebook can roll up coverage.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from apps.backend.preprocess.base import (
    StepResult,
    detect_page_polarity,
    ensure_bgr,
    to_grayscale,
)

DEFAULT_CLAHE_CLIP_LIMIT: float = 3.0
"""Tuned on a 5-page A/B (cover, 2 clean body, 1 manuscript photo, 1
plate). ``clip<2`` was visually indistinguishable from the input on
manuscript pages; ``clip>4`` started to ring on JPEG block boundaries
and amplify scanner noise. ``3.0`` is the recall sweet spot for
PaddleOCR (+2× chars on the test manuscript)."""

DEFAULT_CLAHE_TILE_SIZE: int = 8
"""Grid size for the CLAHE histogram tiles. Smaller tiles (8×8)
produce more local contrast — useful on photo plates where ink density
varies across the page. Larger tiles (16×16) behave closer to a global
histogram stretch and miss faded-ink regions."""

DEFAULT_UNSHARP_RADIUS: float = 1.2
"""Standard deviation (pixels) of the Gaussian blur subtracted from
the original for the unsharp-mask pass. ``1.2`` targets character
strokes at 400 DPI scans (stroke width ≈ 2–4 px) without ringing on
larger block boundaries."""

DEFAULT_UNSHARP_AMOUNT: float = 0.5
"""Weight on the high-frequency residual added back. ``0.5`` is a
gentle bump (a value of ``1.0`` is the classic "unsharp mask" used in
photography; that ringed PaddleOCR character recognition in our
tests)."""

COLOR_SATURATION_LIMIT: float = 12.0
"""Mean HSV saturation (0–255 scale) above which a page is treated as
"colourful" and skipped. Tuned on a 43-page survey: pure B/W body
pages score 0-2, marbled gray covers (which superficially look
monochrome but have a slight green/brown cast) score ~14, full
colour covers score 40-120. ``12`` is the safe cutoff that catches
the marbled cover class without over-rejecting body pages — the
earlier threshold of 25 leaked one false-apply (a marbled cover
where enhance lost -29 chars on PaddleOCR)."""

BIMODAL_MIDTONE_FRACTION: float = 0.30
"""Fraction of pixels in the [50, 200] midtone band BELOW which the
page is treated as already-bimodal (nothing for CLAHE to spread).
Tuned on the 43-page survey: clean printed body pages score 0.00-0.02,
body pages with JPEG halos around glyphs score 0.05-0.10 (enhance
*hurt* these by -3% to -15% on PaddleOCR), photo facsimiles of
manuscripts score 0.35-0.55 (enhance helped by +90% on PaddleOCR).
``0.30`` is the cliff between "halos" and "real low-contrast text"."""

LIGHT_FRACTION_FLOOR: float = 0.30
"""Fraction of pixels brighter than 225 BELOW which the page is
treated as "not really a text-on-paper page" — covers, full-page
photo plates, and decorative pages have light_fraction < 0.10.
Manuscripts photographed with white-paper margins score 0.30-0.70.
The combined gate (midtone ≥ 0.30 **AND** light_fraction ≥ 0.30)
isolates the "low-contrast text with white margins" case where
enhance genuinely helps."""

MIDTONE_LOW: int = 50
MIDTONE_HIGH: int = 200
LIGHT_THRESHOLD: int = 225


def _mean_saturation(image: np.ndarray) -> float:
    """Mean HSV-saturation of ``image`` on the 0–255 scale.

    Cheap proxy for "is this a colour page?". Pure greys give 0; vivid
    primaries give > 200. We average over the whole frame; large
    monochrome margins around a small colour stamp will still trip the
    gate, which is the conservative behaviour we want.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


def _midtone_fraction(image: np.ndarray) -> float:
    """Fraction of grayscale pixels in the ``[MIDTONE_LOW, MIDTONE_HIGH]`` band."""

    gray = to_grayscale(image)
    mask = (gray >= MIDTONE_LOW) & (gray <= MIDTONE_HIGH)
    return float(mask.mean())


def _light_fraction(image: np.ndarray) -> float:
    """Fraction of grayscale pixels brighter than ``LIGHT_THRESHOLD``.

    Used by the page-character gate: a true "text on paper" page has
    plenty of bright margin pixels; covers and full-page photo plates
    do not.
    """

    gray = to_grayscale(image)
    return float((gray > LIGHT_THRESHOLD).mean())


def _clahe_lab(image: np.ndarray, *, clip: float, tile: int) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB and return BGR."""

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(L)
    return cv2.cvtColor(cv2.merge([L2, a, b]), cv2.COLOR_LAB2BGR)


def _unsharp_mask(image: np.ndarray, *, radius: float, amount: float) -> np.ndarray:
    """Gaussian-blur + weighted-subtraction unsharp mask."""

    blur = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=radius)
    return cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0.0)


def enhance_contrast(
    image: np.ndarray,
    *,
    clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    tile_size: int = DEFAULT_CLAHE_TILE_SIZE,
    unsharp_radius: float = DEFAULT_UNSHARP_RADIUS,
    unsharp_amount: float = DEFAULT_UNSHARP_AMOUNT,
    color_saturation_limit: float = COLOR_SATURATION_LIMIT,
    bimodal_midtone_fraction: float = BIMODAL_MIDTONE_FRACTION,
    light_fraction_floor: float = LIGHT_FRACTION_FLOOR,
    skip_inverted_polarity: bool = True,
    skip_colorful: bool = True,
    skip_already_bimodal: bool = True,
    skip_low_light_fraction: bool = True,
    force: bool = False,
) -> StepResult:
    """Smart-gated CLAHE + unsharp enhancement for OCR.

    Args:
        image: BGR uint8 page image (typically the output of step 6
            ``marginalia``).
        clip_limit: CLAHE clip limit on the L channel. Higher = more
            local contrast (also more noise).
        tile_size: CLAHE grid tile size (``(tile, tile)`` cells).
        unsharp_radius: Gaussian blur σ for the unsharp mask. ``0``
            disables sharpening.
        unsharp_amount: High-frequency residual weight added back to
            the image. ``0`` disables sharpening.
        color_saturation_limit: Skip when mean HSV-S exceeds this.
        bimodal_midtone_fraction: Skip when fewer than this fraction of
            pixels lies in the [50, 200] midtone band (page is already
            bimodal and CLAHE would have nothing to do).
        light_fraction_floor: Skip when fewer than this fraction of
            pixels are brighter than ``LIGHT_THRESHOLD`` (page is a
            cover/photo plate, not a text-on-paper page).
        skip_inverted_polarity: Honour the polarity gate (mirrors the
            same flag on ``correct_illumination`` and
            ``remove_bleed_through``).
        skip_colorful: Honour the colour-saturation gate.
        skip_already_bimodal: Honour the histogram-density gate.
        skip_low_light_fraction: Honour the light-fraction gate.
        force: Ignore every gate and always apply enhancement. Useful
            for debugging and the A/B harness.

    Returns:
        :class:`StepResult` whose ``metrics["applied"]`` records whether
        CLAHE ran and ``metrics["reason"]`` (when skipped) explains
        why. When applied, ``metrics`` also includes ``midtoneFraction``,
        ``meanSaturation``, and ``backgroundMeanPolarityProbe`` so the
        audit notebook can correlate gating outcomes with image stats.
    """
    arr = ensure_bgr(image)
    polarity = detect_page_polarity(arr)
    saturation = _mean_saturation(arr)
    midtone = _midtone_fraction(arr)
    light = _light_fraction(arr)

    params: dict[str, Any] = {
        "clip_limit": float(clip_limit),
        "tile_size": int(tile_size),
        "unsharp_radius": float(unsharp_radius),
        "unsharp_amount": float(unsharp_amount),
        "color_saturation_limit": float(color_saturation_limit),
        "bimodal_midtone_fraction": float(bimodal_midtone_fraction),
        "light_fraction_floor": float(light_fraction_floor),
        "skip_inverted_polarity": bool(skip_inverted_polarity),
        "skip_colorful": bool(skip_colorful),
        "skip_already_bimodal": bool(skip_already_bimodal),
        "skip_low_light_fraction": bool(skip_low_light_fraction),
        "force": bool(force),
    }

    base_metrics: dict[str, Any] = {
        "background_mean_polarity_probe": polarity["background_mean"],
        "global_mean": polarity["global_mean"],
        "mean_saturation": saturation,
        "midtone_fraction": midtone,
        "light_fraction": light,
    }

    if not force:
        if skip_inverted_polarity and polarity["is_inverted"]:
            return StepResult(
                image=arr,
                step="enhance",
                params=params,
                metrics={**base_metrics, "applied": False, "reason": "inverted_polarity"},
            )
        if skip_colorful and saturation > color_saturation_limit:
            return StepResult(
                image=arr,
                step="enhance",
                params=params,
                metrics={**base_metrics, "applied": False, "reason": "colorful_content"},
            )
        if skip_already_bimodal and midtone < bimodal_midtone_fraction:
            return StepResult(
                image=arr,
                step="enhance",
                params=params,
                metrics={**base_metrics, "applied": False, "reason": "already_bimodal"},
            )
        if skip_low_light_fraction and light < light_fraction_floor:
            return StepResult(
                image=arr,
                step="enhance",
                params=params,
                metrics={**base_metrics, "applied": False, "reason": "low_light_fraction"},
            )

    enhanced = _clahe_lab(arr, clip=clip_limit, tile=tile_size)
    if unsharp_radius > 0 and unsharp_amount > 0:
        enhanced = _unsharp_mask(
            enhanced, radius=unsharp_radius, amount=unsharp_amount
        )

    return StepResult(
        image=enhanced,
        step="enhance",
        params=params,
        metrics={
            **base_metrics,
            "applied": True,
            "input_mean": float(to_grayscale(arr).mean()),
            "output_mean": float(to_grayscale(enhanced).mean()),
        },
    )
