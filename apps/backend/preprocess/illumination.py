"""Illumination correction via flat-field normalisation (plan §6 Stage 2).

影印本 scans typically show uneven shading: bright in the centre, dim
near the gutter, with the brightness gradient strongest along the spine.
We undo this by:

1. Estimating the *background* lightness field with a heavily-blurred
   morphological close (replaces text with the background's local
   median).
2. Dividing the original by the background to flatten the field.
3. Re-stretching to [0, 255] so the OCR engines see crisp text.

This is exactly the "flat-field correction" recipe the plan asks for and
the standard preprocessing step in OCR-D / kraken pipelines. Done in
grayscale because illumination is a multiplicative scalar field; the
result is broadcast back to the original BGR channels.
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

DEFAULT_BACKGROUND_KSIZE: int = 51
"""Morphological close kernel size for background estimation. ~5% of a
typical 2000-px page width — large enough to swallow text glyphs but
small enough to follow the illumination gradient."""

DEFAULT_GAIN_FLOOR: float = 0.4
"""Lower bound on the per-pixel gain so dark text isn't blown out by a
near-zero background estimate."""

DEFAULT_BACKGROUND_STD_LIMIT: float = 35.0
"""Maximum allowed standard deviation of the estimated background field.
A truly flat-fielded body page has bg_std < 25 (corpus median ~19, p95
~32); pages with photos, cover art, or large solid colour blocks (e.g.
the blue book cover that triggered this guard at bg_std=41) come in
well above this and would be over-corrected by the percentile re-stretch
(the cream paper region shifts toward the dominant blue). Tuned on a
1,133-page body-page sample (skips ~5% — the same pages that benefit
least from a flat-field model anyway)."""


def _estimate_background(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Return an estimate of the local-background brightness field."""

    ksize = max(3, ksize | 1)  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # A heavy Gaussian smooths the closed field into a true illumination
    # estimate (otherwise the kernel boundary shows up as halos).
    return cv2.GaussianBlur(closed, (ksize, ksize), 0)


def correct_illumination(
    image: np.ndarray,
    *,
    background_ksize: int = DEFAULT_BACKGROUND_KSIZE,
    gain_floor: float = DEFAULT_GAIN_FLOOR,
    background_std_limit: float = DEFAULT_BACKGROUND_STD_LIMIT,
    skip_inverted_polarity: bool = True,
    force: bool = False,
) -> StepResult:
    """Flatten the per-pixel illumination of a scanned page.

    The algorithm operates on luminance; the resulting gain map is then
    applied per BGR channel so the colour balance is preserved.

    The step is a **no-op** when its flat-field assumption breaks down:

    - **Inverted polarity** (light foreground on dark background, e.g. a
      printed book cover with white text on coloured back): the
      morphological close + percentile rescale would darken the
      foreground, washing it into the background. Skipped unless
      ``skip_inverted_polarity=False``.
    - **Non-uniform background** (``background_std > background_std_limit``):
      cover pages, photos, and decorative layouts have multimodal
      backgrounds that the single-gain-field model can't represent.
      Skipped unless ``force=True``.

    Args:
        image: BGR uint8 page image.
        background_ksize: Kernel size (odd) for the background estimator.
        gain_floor: Minimum allowed background value (in [0, 1]) used when
            dividing. Prevents division-by-zero blow-up in pixels where the
            estimated background was nearly black.
        background_std_limit: If the estimated background's std deviation
            exceeds this, skip (the page isn't well modelled by a single
            illumination field). Default tuned on the corpus.
        skip_inverted_polarity: Pass through inverted-polarity pages
            instead of darkening them.
        force: Override every safety check and always apply the
            correction.

    Returns:
        :class:`StepResult` with ``step='illumination'``. When skipped,
        ``image`` is the input unchanged and ``metrics['applied']=False``
        with a populated ``metrics['reason']``.
    """
    bgr = ensure_bgr(image)
    gray = to_grayscale(bgr)

    polarity = detect_page_polarity(bgr)
    if skip_inverted_polarity and polarity["is_inverted"] and not force:
        return StepResult(
            image=bgr,
            step="illumination",
            params={
                "background_ksize": background_ksize,
                "gain_floor": gain_floor,
                "background_std_limit": background_std_limit,
                "skip_inverted_polarity": skip_inverted_polarity,
                "force": force,
            },
            metrics={
                "applied": False,
                "reason": "inverted_polarity",
                "background_mean_polarity_probe": polarity["background_mean"],
                "global_mean": polarity["global_mean"],
            },
        )

    background = _estimate_background(gray, background_ksize).astype(np.float32)
    bg_std = float(np.std(background))
    if not force and bg_std > background_std_limit:
        return StepResult(
            image=bgr,
            step="illumination",
            params={
                "background_ksize": background_ksize,
                "gain_floor": gain_floor,
                "background_std_limit": background_std_limit,
                "skip_inverted_polarity": skip_inverted_polarity,
                "force": force,
            },
            metrics={
                "applied": False,
                "reason": "non_uniform_background",
                "background_mean": float(np.mean(background)),
                "background_std": bg_std,
                "background_std_limit": background_std_limit,
                "input_mean": float(np.mean(gray)),
            },
        )

    bg_norm = background / 255.0
    bg_norm = np.maximum(bg_norm, float(gain_floor))

    bgr_f = bgr.astype(np.float32)
    flat = bgr_f / bg_norm[:, :, None]

    p_low, p_high = np.percentile(flat, [1.0, 99.0])
    if p_high > p_low + 1e-6:
        flat = (flat - p_low) * (255.0 / (p_high - p_low))
    out = np.clip(flat, 0, 255).astype(np.uint8)

    metrics: dict[str, Any] = {
        "applied": True,
        "background_mean": float(np.mean(background)),
        "background_std": bg_std,
        "input_mean": float(np.mean(gray)),
        "output_mean": float(np.mean(to_grayscale(out))),
        "percentile_low": float(p_low),
        "percentile_high": float(p_high),
    }
    return StepResult(
        image=out,
        step="illumination",
        params={
            "background_ksize": background_ksize,
            "gain_floor": gain_floor,
            "background_std_limit": background_std_limit,
            "skip_inverted_polarity": skip_inverted_polarity,
            "force": force,
        },
        metrics=metrics,
    )
