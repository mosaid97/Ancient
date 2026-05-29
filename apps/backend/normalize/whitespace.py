"""Step 2 of the canonical pipeline: whitespace collapse.

Classical-Chinese editions interleave a wide variety of whitespace and
punctuation-adjacent space (full-width space U+3000, no-break space U+00A0,
zero-width joiners, etc.). This step:

1. Strips zero-width chars (ZWJ U+200D, ZWNJ U+200C, ZWSP U+200B, BOM U+FEFF).
2. Replaces every horizontal whitespace run with a single ASCII space.
3. Strips leading/trailing whitespace.

It deliberately does NOT touch line breaks (``\\n``) — chunking is line-aware.
"""

from __future__ import annotations

import re

ZERO_WIDTH = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\ufeff",  # BOM
}

_HORIZ_WS = re.compile(r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")


def normalize(text: str) -> str:
    """Strip zero-width characters and collapse horizontal whitespace.

    Args:
        text: Input string.

    Returns:
        Text with zero-width chars removed and runs of horizontal whitespace
        collapsed to a single ASCII space. Newlines preserved.
    """
    if not text:
        return text
    if any(ch in ZERO_WIDTH for ch in text):
        text = "".join(ch for ch in text if ch not in ZERO_WIDTH)
    text = _HORIZ_WS.sub(" ", text)
    return text.strip(" ")
