"""Step 7 of the canonical pipeline: half/full-width kana unification.

Always on for ``ja`` and ``mixed`` languages. mojimoji wraps the Japanese
NFKC-equivalent half/full-width conversions cleanly. Default direction is
half-to-full (``han_to_zen``) so that mixed-script Tang-era pages with
Japanese editorial layers (e.g. 唐令拾遗補) end up with a single normalized
kana form.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ZH_LIKE_LANGS = {"zh", "zh-classical", "zh-modern"}


def normalize(text: str, *, lang: str = "zh") -> str:
    """Apply mojimoji half<->full-width unification when language is ja/mixed.

    Args:
        text: Input string.
        lang: Language tag. ``zh*`` languages -> no-op (preserves classical
            Chinese intent). ``ja*`` and anything else -> half_to_full.

    Returns:
        Normalized text. Returns input unchanged if mojimoji is unavailable.
    """
    if not text:
        return text
    if (lang or "").lower() in ZH_LIKE_LANGS:
        return text
    try:
        import mojimoji
    except ImportError:
        logger.warning(
            "mojimoji not installed; kana normalization is a no-op. "
            "uv add mojimoji to enable."
        )
        return text
    try:
        return mojimoji.han_to_zen(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("mojimoji.han_to_zen failed; returning input: %s", exc)
        return text
