"""Phase-2 OCR pre-processing pipeline (plan §6 Stage 2).

Six deterministic image-processing steps run on every scanned ``(:PAGE
{mode='ocr'})`` *before* the dual-OCR engines see it:

1. :mod:`apps.backend.preprocess.deskew` — Hough-line skew correction.
2. :mod:`apps.backend.preprocess.dewarp` — OpenCV remap (DocUNet hook reserved).
3. :mod:`apps.backend.preprocess.illumination` — flat-field illumination correction.
4. :mod:`apps.backend.preprocess.bleed` — bleed-through removal via channel separation.
5. :mod:`apps.backend.preprocess.page_split` — double-page split via vertical-projection valley.
6. :mod:`apps.backend.preprocess.marginalia` — 天頭/地腳/版心 separation by edge-projection.
7. :mod:`apps.backend.preprocess.enhance` — smart-gated CLAHE + unsharp for low-contrast facsimiles.

All six steps expose the same interface (:class:`StepResult`) and the
orchestrator (:mod:`apps.backend.pipeline.preprocess`) wires them in
order. Every intermediate variant is persisted to MinIO under
``ancient-pages/<document_id>/page_<n>/<variant>.png`` for downstream
auditing (plan §6 Stage 2 "All variants persisted to MinIO with
provenance").

Per AGENTS.md §10 we keep this module dependency-light: only NumPy +
OpenCV (headless) are imported; PIL is used by callers, not here.
"""

from __future__ import annotations

from apps.backend.preprocess.base import (
    PreprocessProvenance,
    PreprocessStep,
    StepResult,
)
from apps.backend.preprocess.bleed import remove_bleed_through
from apps.backend.preprocess.deskew import deskew_image, detect_skew_angle
from apps.backend.preprocess.dewarp import dewarp_image
from apps.backend.preprocess.enhance import enhance_contrast
from apps.backend.preprocess.illumination import correct_illumination
from apps.backend.preprocess.marginalia import (
    MarginaliaRegions,
    detect_marginalia_regions,
    separate_marginalia,
)
from apps.backend.preprocess.page_split import (
    SplitDecision,
    detect_double_page,
    split_double_page,
)

__all__ = [
    "MarginaliaRegions",
    "PreprocessProvenance",
    "PreprocessStep",
    "SplitDecision",
    "StepResult",
    "correct_illumination",
    "deskew_image",
    "detect_double_page",
    "detect_marginalia_regions",
    "detect_skew_angle",
    "dewarp_image",
    "enhance_contrast",
    "remove_bleed_through",
    "separate_marginalia",
    "split_double_page",
]
