"""Step 1 of the canonical pipeline: Unicode NFC normalization.

NFC is "always on" per plan §2.7. Unicode-equivalent codepoints (e.g. composed
vs. decomposed CJK ideographs, full-width latin) are unified. This is the
cheapest and most universally-safe transformation.
"""

from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    """Return ``unicodedata.normalize('NFC', text)``.

    Args:
        text: Any string.

    Returns:
        The NFC form of ``text``.
    """
    if not text:
        return text
    return unicodedata.normalize("NFC", text)
