"""Bleed-through removal via colour-channel separation (plan §6 Stage 2).

Old-paper scans typically show *reverse-side* text bleeding through:
faint, lower-saturation ghosts of the verso glyphs sitting between the
recto's crisp dark strokes. Because the verso ink usually penetrated less
deeply, the bleed-through is preferentially preserved in *one* colour
channel (most often the red channel of a yellow-tinted paper) while the
foreground text is uniformly dark across all channels.

Strategy (deterministic, ~20 ms per page):

1. Pick the *brightest* of the three BGR channels at every pixel
   (``np.max``). This emphasises the channel where the bleed has been
   absorbed least and the paper is brightest, suppressing the bleed
   signal which sits in the absorbed channel(s).
2. Build a *paper mask* via Otsu binarisation of that channel — the
   "paper" pixels (above Otsu) are pure background.
3. For paper pixels, replace the original BGR with the mean paper colour
   (so faint bleed ghosts become uniform background).
4. Keep the foreground (dark, ink) pixels untouched so OCR confidence is
   preserved on the real text.

The output is a noticeably cleaner page that doesn't disturb the
foreground glyphs. Caveats:

- Aggressive removal can clip very faint footnote glyphs. We expose a
  ``foreground_quantile`` knob to recover them at the cost of leaving
  more bleed.
- For pure black-on-white modern scans the step is essentially a no-op,
  which is fine — Otsu on a clean page picks a very tight cut and the
  paper-mean replacement is a near-identity.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from apps.backend.preprocess.base import StepResult, detect_page_polarity, ensure_bgr

DEFAULT_FOREGROUND_QUANTILE: float = 0.30
"""Pixels darker than this quantile of the brightest-channel histogram are
kept as foreground; the rest become paper. Lower = more permissive
(keeps faint marks); higher = more aggressive (kills more bleed)."""

DEFAULT_PAPER_COLOR_TOLERANCE: float = 30.0
"""L2 distance (in 0–255 BGR space) within which a "paper" pixel is
considered close enough to the global paper mean to be safely repainted
with that mean. Pixels FURTHER than this from the paper mean are
preserved verbatim — this is what keeps **multimodal pages** (e.g. a
book cover with white text on blue background, or a body page with a
red seal on cream paper) from having their non-paper bright regions
silently recoloured. The threshold roughly corresponds to a JND
("just-noticeable-difference") in 8-bit colour."""


def remove_bleed_through(
    image: np.ndarray,
    *,
    foreground_quantile: float = DEFAULT_FOREGROUND_QUANTILE,
    paper_color_tolerance: float = DEFAULT_PAPER_COLOR_TOLERANCE,
    skip_inverted_polarity: bool = True,
    force: bool = False,
) -> StepResult:
    """Repaint near-paper pixels with the page's mean paper colour.

    Bleed-through is suppressed by replacing the paper-class pixels with
    a single representative paper colour. The original implementation
    replaced **every** paper-class pixel, which silently destroyed light
    foreground content on covers / illustrated pages (white-on-blue text
    has a "paper" max-channel but is **not** paper). This version is
    **distance-aware**: paper-class pixels are only repainted when their
    BGR colour is within ``paper_color_tolerance`` of the global paper
    mean. Outliers (white text, red seals, cream margins on a
    blue-dominant page) are preserved.

    The step also no-ops on **inverted-polarity** pages (dark backgrounds
    with light foreground): on those the foreground / paper labelling
    inverts and the algorithm would erase the very text we want to keep.

    Args:
        image: BGR uint8 page image.
        foreground_quantile: Quantile of the per-pixel max-channel
            histogram below which a pixel is treated as foreground.
        paper_color_tolerance: L2 BGR distance below which paper-class
            pixels are repainted; outliers are preserved. 0 disables the
            distance gate (legacy behaviour).
        skip_inverted_polarity: Pass through inverted-polarity pages.
        force: Disable every safety check; always apply the legacy
            replace-every-paper-pixel behaviour (useful only for ablation
            studies in Phase 4).

    Returns:
        :class:`StepResult` with ``step='bleed'``.
    """
    bgr = ensure_bgr(image)

    polarity = detect_page_polarity(bgr)
    if skip_inverted_polarity and polarity["is_inverted"] and not force:
        return StepResult(
            image=bgr,
            step="bleed",
            params={
                "foreground_quantile": foreground_quantile,
                "paper_color_tolerance": paper_color_tolerance,
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

    max_chan = np.max(bgr, axis=2)
    threshold = float(np.quantile(max_chan, foreground_quantile))
    otsu_thr, _ = cv2.threshold(max_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cutoff = float(min(threshold, otsu_thr))
    foreground_mask = max_chan < cutoff

    paper_mask = ~foreground_mask
    paper_pixels = bgr[paper_mask]
    if paper_pixels.size == 0:
        return StepResult(
            image=bgr,
            step="bleed",
            params={
                "foreground_quantile": foreground_quantile,
                "paper_color_tolerance": paper_color_tolerance,
                "skip_inverted_polarity": skip_inverted_polarity,
                "force": force,
            },
            metrics={
                "cutoff": cutoff,
                "otsu_threshold": float(otsu_thr),
                "foreground_pixels": int(foreground_mask.sum()),
                "paper_pixels": 0,
                "applied": False,
                "reason": "no_paper_pixels",
            },
        )

    paper_mean = paper_pixels.mean(axis=0)
    paper_std_per_channel = paper_pixels.std(axis=0)
    paper_mean_u8 = paper_mean.astype(np.uint8)

    if paper_color_tolerance > 0 and not force:
        # Distance-aware: only repaint paper pixels close to the mean.
        # Compute per-pixel L2 distance to the paper mean in float32.
        diff = bgr.astype(np.float32) - paper_mean[None, None, :]
        distance = np.sqrt(np.sum(diff * diff, axis=2))
        replace_mask = paper_mask & (distance <= paper_color_tolerance)
    else:
        replace_mask = paper_mask

    out = bgr.copy()
    out[replace_mask] = paper_mean_u8
    preserved_paper = int(paper_mask.sum() - replace_mask.sum())

    metrics: dict[str, Any] = {
        "cutoff": cutoff,
        "otsu_threshold": float(otsu_thr),
        "foreground_pixels": int(foreground_mask.sum()),
        "paper_pixels": int(paper_mask.sum()),
        "paper_pixels_replaced": int(replace_mask.sum()),
        "paper_pixels_preserved": preserved_paper,
        "paper_mean_bgr": [int(c) for c in paper_mean_u8.tolist()],
        "paper_std_bgr": [round(float(s), 2) for s in paper_std_per_channel.tolist()],
        "paper_color_tolerance": paper_color_tolerance,
        "applied": True,
    }
    return StepResult(
        image=out,
        step="bleed",
        params={
            "foreground_quantile": foreground_quantile,
            "paper_color_tolerance": paper_color_tolerance,
            "skip_inverted_polarity": skip_inverted_polarity,
            "force": force,
        },
        metrics=metrics,
    )
