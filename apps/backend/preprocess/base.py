"""Shared types for the Phase-2 preprocessing steps (plan §6 Stage 2).

Every step takes a ``numpy.ndarray`` image (BGR uint8) and returns a
:class:`StepResult` that the orchestrator chains into the next step. The
:class:`PreprocessProvenance` dataclass aggregates each step's parameters +
metrics so the orchestrator can persist a single JSON blob to
``PAGE.preprocessingProvenance`` for downstream auditing.

The base module intentionally has no third-party imports beyond NumPy so
the step modules can import it without dragging OpenCV through the call
chain when only the dataclasses are needed (e.g., notebook reporting).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

import numpy as np


def _snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def _to_camel_dict(value: Any) -> Any:
    """Recursively rewrite dict keys from snake_case to camelCase.

    Used by :class:`PreprocessProvenance.to_dict` so the serialised
    provenance blob respects the project-wide Neo4j property-key convention
    (AGENTS.md §4: ``camelCase`` props).
    """
    if isinstance(value, dict):
        return {_snake_to_camel(k): _to_camel_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_camel_dict(v) for v in value]
    return value

PreprocessStep = Literal[
    "deskew",
    "dewarp",
    "illumination",
    "bleed",
    "split",
    "marginalia",
    "enhance",
]


@dataclass
class StepResult:
    """Uniform return for every Phase-2 preprocessing step.

    Attributes:
        image: The processed BGR uint8 image. Steps that emit multiple
            outputs (page split, marginalia) MUST return the *primary*
            image here and surface siblings via ``extras``.
        step: The step name (matches :data:`PreprocessStep`).
        params: Parameters actually used (after any defaulting). Recorded
            in provenance so a re-run is reproducible.
        metrics: Quantitative outcomes (skew angle, illumination gain,
            bleed-through pixels removed, etc.).
        extras: Optional sibling outputs keyed by short name (e.g.
            ``"left"`` / ``"right"`` for :func:`split_double_page`,
            ``"marginalia_top"`` for :func:`separate_marginalia`).
    """

    image: np.ndarray
    step: PreprocessStep
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def to_summary(self) -> dict[str, Any]:
        """Provenance-friendly summary (drops the heavy image arrays)."""

        return {
            "step": self.step,
            "shape": list(self.image.shape) if self.image is not None else None,
            "dtype": str(self.image.dtype) if self.image is not None else None,
            "params": self.params,
            "metrics": self.metrics,
            "extras": sorted(self.extras.keys()),
        }


@dataclass
class PreprocessProvenance:
    """Audit trail for one page's full 6-step preprocessing run.

    Persisted in :func:`apps.backend.pipeline.preprocess.preprocess_page`
    as ``PAGE.preprocessingProvenance`` (JSON-encoded) so the verifier
    (Phase 9), the active-learning prioritizer (Phase 5), and the HITL UI
    (Phase 10) can all see *which* steps fired with *which* parameters
    and produced *which* sibling images.
    """

    page_id: str
    document_id: str
    source_uri: str
    final_uri: str
    target_dpi: int
    steps: list[dict[str, Any]] = field(default_factory=list)
    variant_uris: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    # Optional siblings (set when page-split or marginalia create children).
    # Each is the MinIO key for the sibling page image — the orchestrator
    # writes a separate `(:PAGE)` node with `role='marginalia'` etc.
    split_children: list[str] = field(default_factory=list)
    marginalia_children: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Camel-cased serialisation for Neo4j storage (AGENTS.md §4).

        Inner step summaries already use ``params`` / ``metrics`` /
        ``extras`` snake-case keys at construction time; we recursively
        rewrite to camelCase so the on-disk JSON matches the property-key
        convention the verifier and HITL UI will expect.
        """
        return _to_camel_dict(asdict(self))


class StepCallable(Protocol):
    """Structural type for a step callable (used by the orchestrator)."""

    def __call__(self, image: np.ndarray, **kwargs: Any) -> StepResult:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Image-IO helpers (kept here so the step modules don't each re-implement
# the BGR/grayscale dance and so tests can stub them).
# ---------------------------------------------------------------------------


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Coerce a 2D grayscale or 4-channel BGRA image into a 3-channel BGR uint8 array.

    Args:
        image: NumPy image (H×W, H×W×3, or H×W×4) in uint8 / float.

    Returns:
        Contiguous BGR uint8 ``np.ndarray``.

    Raises:
        ValueError: For unsupported shapes / dtypes.
    """
    if image is None:
        raise ValueError("ensure_bgr: image is None")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            # BGRA -> BGR (drop alpha; opencv channel order assumed).
            return np.ascontiguousarray(arr[:, :, :3])
        if arr.shape[2] == 3:
            return np.ascontiguousarray(arr)
    raise ValueError(f"ensure_bgr: unsupported image shape {arr.shape}, dtype {arr.dtype}")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 grayscale view of ``image``."""

    arr = ensure_bgr(image)
    # Manual luminance to avoid pulling cv2 into the base import path.
    g = arr.astype(np.float32)
    y = 0.114 * g[:, :, 0] + 0.587 * g[:, :, 1] + 0.299 * g[:, :, 2]
    return np.clip(y, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Polarity / page-character detection (used by illumination + bleed to skip
# pages where their "dark ink on light paper" assumption is violated).
# ---------------------------------------------------------------------------


INVERTED_POLARITY_BG_MEAN: float = 110.0
"""Background-brightness threshold below which a page is treated as
"inverted polarity" (light text on dark paper, e.g. printed cover with
white characters on a coloured back). Both illumination correction and
bleed-through removal silently no-op on inverted pages because their
algorithms would over-correct toward the dark background and erase the
light foreground."""


def detect_page_polarity(image: np.ndarray) -> dict[str, float | bool]:
    """Cheap heuristic page-polarity probe.

    The "background" is approximated as the brightest 50 % of pixels;
    on a normal scan that's the paper, on an inverted cover that's the
    coloured back. A background brighter than
    :data:`INVERTED_POLARITY_BG_MEAN` is "normal"; darker is "inverted".

    Args:
        image: BGR uint8 page image.

    Returns:
        ``{'background_mean': float, 'global_mean': float, 'is_inverted': bool}``.
        ``is_inverted`` is the only field downstream steps need; the
        means are recorded in metrics for auditing.
    """

    gray = to_grayscale(image)
    median = float(np.median(gray))
    bg_pixels = gray[gray >= median]
    background_mean = float(bg_pixels.mean()) if bg_pixels.size else float(gray.mean())
    return {
        "background_mean": background_mean,
        "global_mean": float(gray.mean()),
        "is_inverted": background_mean < INVERTED_POLARITY_BG_MEAN,
    }
