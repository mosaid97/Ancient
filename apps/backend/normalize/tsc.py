"""Step 3 of the canonical pipeline: Traditional <-> Simplified (T-S).

Default direction is ``s2t`` (simplified -> traditional) because the corpus
is dominated by 古籍 in traditional script and modern reprints sometimes
mix the two. The verifier matches a query against the canonical (T) form.

Falls back gracefully if ``opencc-python-reimplemented`` is missing —
rather than crash the whole pipeline, the step becomes a no-op and logs a
warning.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "s2t"  # 简 -> 繁


@lru_cache(maxsize=4)
def _get_converter(config: str):
    """Cache an OpenCC converter per config (``s2t``, ``t2s``, etc.)."""
    try:
        import opencc

        return opencc.OpenCC(config)
    except ImportError:
        logger.warning(
            "opencc-python-reimplemented not installed; T-S normalization "
            "is a no-op. `uv add opencc-python-reimplemented` to enable."
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenCC init failed for config=%s: %s", config, exc)
        return None


def normalize(text: str, config: str = DEFAULT_CONFIG) -> str:
    """Convert between Traditional and Simplified Chinese.

    Args:
        text: Input string.
        config: An OpenCC config name, e.g. ``s2t``, ``t2s``, ``s2tw``,
            ``tw2sp``. Default ``s2t``.

    Returns:
        Converted text. Returns ``text`` unchanged if OpenCC is unavailable.
    """
    if not text:
        return text
    conv = _get_converter(config)
    if conv is None:
        return text
    try:
        return conv.convert(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenCC convert failed (%s); returning input: %s", config, exc)
        return text
