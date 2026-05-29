"""Inline LLM-OCR output validation (plan §6 Stage 3 hardening).

Runs immediately after every LLM OCR call (DeepSeek-OCR, Qwen-VL-OCR) and
**before** the result is written to Neo4j. If the output is classified as a
hallucination, the engine wrapper zeros the text, sets ``error`` to
``'validation_failed:<reason>'``, and returns the result so the pipeline
writer stores it as ``status='empty'`` rather than polluting the fused text.

Failure modes catalogued during the 2026-05-19 Phase-3 audit (403/2941
body pages, 13.7% hallucination rate) — see AGENTS.md §11:

1. **english_image_description** — model switches to image-description mode
   ("This image displays...", "I cannot read...") instead of transcribing.
2. **internal_token_leakage** — DeepSeek-OCR leaks ``<|ref|>`` / ``<|det|>``
   spatial-coordinate tokens from its fine-tuning annotation format.
3. **html_table_markup** — ``<table>`` / ``<td>`` output from document-layout
   encoding.  :func:`apply_validation` *first* strips HTML tags and
   re-validates the residual; if the stripped text contains sufficient CJK
   content the cleaned version is returned rather than blocking outright.
4. **sequential_number_table** — ``| 1 | 2 | 3 |`` counting column positions
   instead of reading text.
5. **low_cjk_ratio** — for texts longer than 100 chars, fewer than 2% of
   meaningful characters are CJK Unified Ideographs; covers ColN tables,
   circled-number sequences, pure-Latin outputs, etc.
6. **repeated_char** — a single character (CJK or ASCII) accounts for > 55%
   of the compact (whitespace-stripped) output; model got stuck in a loop.
7. **digit_dominated** — more than 40% of characters are ASCII digits (line
   number streams, coordinate dumps).

Special cases that are **not** hallucinations:

- ``（空页）`` / ``（空頁）`` — the model correctly identified a blank page.
  These are returned as ``(True, None)`` so the engine marks the page
  ``status='empty'`` rather than a failure.
- Markdown tables containing real CJK content (e.g. ``| 梁 | 北魏 |``) pass
  the CJK-ratio check and are therefore **kept** — they represent genuine
  table pages that the LLM correctly formatted.
- Very short texts (≤ 10 chars) are not validated for CJK ratio or character
  distribution; they are returned as-is and will be classified as ``'empty'``
  by the pipeline if they produce no useful content.
- HTML-wrapped bibliography pages: HTML tags are stripped and the residual
  text is re-validated.  If the residual has CJK content, the **stripped**
  text is accepted (not the original HTML).
"""

from __future__ import annotations

import re
from collections import Counter

#: Strip HTML/XML tags for the html_table_markup rescue path.
_HTML_TAG_RE: re.Pattern[str] = re.compile(r"<[^>]+>", re.DOTALL)
#: Strip HTML entities (e.g. &nbsp; &#40; &#x4E00;) before CJK ratio check.
_HTML_ENTITY_RE: re.Pattern[str] = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
#: CJK ratio required for stripped-HTML text to be accepted (generous — we
#: only need a few CJK chars to confirm this is real content, not a skeleton).
_HTML_STRIP_CJK_THRESHOLD: float = 0.05

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Strings the LLM is supposed to return for a blank/image-only page.
_EMPTY_RESPONSES: frozenset[str] = frozenset({
    "（空页）", "（空頁）", "(空页)", "(空頁）", "（空）",
    "（空頁)", "(空頁)", "（空白页）", "（空白頁）",
})

#: CJK Unified Ideographs ranges checked by the ratio heuristic.
_CJK_RANGES: list[tuple[int, int]] = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x20000, 0x2A6DF), # CJK Extension B (SMP)
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
]

#: Prefixes that flag English image-description mode.
_ENGLISH_DESCRIPTION_PREFIXES: tuple[str, ...] = (
    "This image",
    "The image",
    "I cannot",
    "I'm unable",
    "I am unable",
    "This appears",
    "This page",
    "This photograph",
    "This is an image",
    "This is a page",
)

#: Internal model tokens that should never appear in OCR output.
_INTERNAL_TOKEN_PATTERNS: re.Pattern[str] = re.compile(
    r"<\|ref\|>|<\|det\|>|<\|/ref\|>|<\|/det\|>", re.ASCII
)

#: Minimum text length before applying distributional checks.
_MIN_LEN_FOR_DISTRIBUTION_CHECK: int = 100

#: Minimum text length before applying CJK-ratio check.
_MIN_LEN_FOR_CJK_CHECK: int = 100

#: CJK ratio threshold — texts below this are flagged low_cjk_ratio.
_CJK_RATIO_THRESHOLD: float = 0.02

#: Single-character dominance threshold — char > this fraction → flagged.
_REPEATED_CHAR_THRESHOLD: float = 0.55

#: Digit fraction threshold.
_DIGIT_RATIO_THRESHOLD: float = 0.40

# ---------------------------------------------------------------------------
# Traditional-script enforcement constants (2026-05-23)
# ---------------------------------------------------------------------------
# These pairs are characters that are *systematically* replaced in Simplified
# Chinese.  A traditional-script OCR output should contain ZERO instances from
# _SIMP_EXCLUSIVE_CHARS.  A single stray simplified char is allowed (OCR noise),
# but if more than _SIMP_EXCLUSIVE_THRESHOLD appear relative to traditional
# counterparts we flag simplified_conversion.
#
# The list is intentionally conservative: only high-frequency pairs where the
# simplified form is unambiguously distinct (i.e. not a valid classical variant).
# ---------------------------------------------------------------------------
_TRAD_SIMP_PAIRS: tuple[tuple[str, str], ...] = (
    # (traditional, simplified)
    ("議", "议"), ("書", "书"), ("國", "国"), ("會", "会"), ("發", "发"),
    ("傳", "传"), ("體", "体"), ("當", "当"), ("說", "说"), ("時", "时"),
    ("學", "学"), ("從", "从"), ("應", "应"), ("經", "经"), ("處", "处"),
    ("總", "总"), ("關", "关"), ("節", "节"), ("點", "点"), ("問", "问"),
    ("兩", "两"), ("術", "术"), ("難", "难"), ("號", "号"), ("實", "实"),
    ("開", "开"), ("長", "长"), ("貞", "贞"), ("頒", "颁"), ("編", "编"),
    ("歷", "历"), ("陽", "阳"), ("樣", "样"), ("達", "达"), ("線", "线"),
    ("劉", "刘"), ("劍", "剑"), ("軍", "军"), ("農", "农"), ("廣", "广"),
    ("觀", "观"), ("漢", "汉"), ("華", "华"), ("來", "来"), ("對", "对"),
)
_TRAD_EXCLUSIVE: frozenset[str] = frozenset(t for t, _ in _TRAD_SIMP_PAIRS)
_SIMP_EXCLUSIVE: frozenset[str] = frozenset(s for _, s in _TRAD_SIMP_PAIRS)

#: If simplified-exclusive char count exceeds this absolute count AND this
#: fraction of the combined trad+simp count, flag the output.
_SIMP_EXCL_ABS_THRESHOLD: int = 3
_SIMP_EXCL_RATIO_THRESHOLD: float = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_cjk(c: str) -> bool:
    cp = ord(c)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _cjk_ratio(text: str) -> float:
    """Fraction of meaningful characters that are CJK."""
    cjk = sum(1 for c in text if _is_cjk(c))
    meaningful = sum(1 for c in text if c not in " \t\n|-./:0123456789")
    return cjk / meaningful if meaningful > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_llm_ocr_output(text: str) -> tuple[bool, str | None]:
    """Validate an LLM OCR response for hallucination patterns.

    Args:
        text: Raw text returned by the LLM (already ``.strip()``-ed by the
            engine wrapper before calling this function).

    Returns:
        A ``(is_valid, reason)`` tuple.

        - ``(True, None)`` → output is acceptable (may still be empty if the
          model correctly identified a blank page).
        - ``(False, reason_code)`` → hallucination detected; the engine wrapper
          should zero the text and set ``error='validation_failed:<reason>'``.
    """
    if not text:
        return True, None  # empty string — handled upstream as status='empty'

    stripped = text.strip()

    # ── Legitimate empty-page responses ────────────────────────────────────
    # Return EMPTY STRING (not the marker text) so the pipeline classifies
    # the page as status='empty' rather than status='ok' with 4-char text.
    # Storing "（空页）" as ok-text would corrupt fusion alignment.
    if stripped in _EMPTY_RESPONSES:
        return True, None  # caller receives ("", None) via apply_validation

    # ── Rule 1: English image-description mode ─────────────────────────────
    for prefix in _ENGLISH_DESCRIPTION_PREFIXES:
        if stripped.startswith(prefix):
            return False, "english_image_description"

    # ── Rule 2: Internal model token leakage ───────────────────────────────
    if _INTERNAL_TOKEN_PATTERNS.search(stripped):
        return False, "internal_token_leakage"

    # ── Rule 3: HTML table markup ───────────────────────────────────────────
    if "<table>" in stripped or "<td>" in stripped:
        return False, "html_table_markup"

    # ── Rule 4: Sequential number table ────────────────────────────────────
    # Catches "| 1 | 2 | 3 |" and "| 1 | 2 |" patterns.
    if re.search(r"\|\s*1\s*\|\s*2\s*\|", stripped):
        return False, "sequential_number_table"

    # Short texts skip the distributional checks below.
    if len(stripped) < _MIN_LEN_FOR_DISTRIBUTION_CHECK:
        return True, None

    # ── Rule 5: Low CJK ratio ──────────────────────────────────────────────
    if len(stripped) >= _MIN_LEN_FOR_CJK_CHECK:
        ratio = _cjk_ratio(stripped)
        if ratio < _CJK_RATIO_THRESHOLD:
            return False, f"low_cjk_ratio:{ratio:.3f}"

    # ── Rule 6: Repeated single character ──────────────────────────────────
    compact = stripped.replace("\n", "").replace(" ", "")
    if len(compact) >= 20:
        ctr = Counter(compact)
        most_char, most_count = ctr.most_common(1)[0]
        if most_count / len(compact) > _REPEATED_CHAR_THRESHOLD:
            return False, f"repeated_char:{most_char!r}"

    # ── Rule 7: Digit-dominated output ─────────────────────────────────────
    digit_count = sum(1 for c in stripped if c.isdigit())
    if digit_count / len(stripped) > _DIGIT_RATIO_THRESHOLD:
        return False, "digit_dominated"

    return True, None


def check_traditional_script(text: str) -> tuple[bool, str | None]:
    """Detect whether an LLM silently converted traditional characters to simplified.

    Called when ``expected_script='traditional'`` (primary-tier 影印本).  Uses
    a targeted set of 45 high-frequency trad/simp character pairs to count
    how many simplified-exclusive chars appear versus their traditional
    counterparts.  A text is flagged only when **both** conditions hold:

    * absolute simplified-exclusive count ≥ :data:`_SIMP_EXCL_ABS_THRESHOLD` (3)
    * simplified-exclusive / (simplified-exclusive + traditional-exclusive)
      ratio > :data:`_SIMP_EXCL_RATIO_THRESHOLD` (0.30)

    The dual threshold prevents false positives on short texts where a few
    genuine OCR mis-hits could contain a simplified-looking glyph.

    Args:
        text: OCR output text (already passed :func:`validate_llm_ocr_output`).

    Returns:
        ``(True, None)`` if the script looks traditional (or indeterminate).
        ``(False, 'simplified_conversion:<simp_count>/<total>')`` if the text
        appears to have been converted to simplified.
    """
    simp_count = sum(1 for c in text if c in _SIMP_EXCLUSIVE)
    trad_count = sum(1 for c in text if c in _TRAD_EXCLUSIVE)
    total = simp_count + trad_count
    if total == 0:
        return True, None  # No indicator chars at all — can't judge
    ratio = simp_count / total
    if simp_count >= _SIMP_EXCL_ABS_THRESHOLD and ratio > _SIMP_EXCL_RATIO_THRESHOLD:
        return False, f"simplified_conversion:{simp_count}/{total}"
    return True, None


def apply_validation(
    text: str,
    *,
    expected_script: str | None = None,
) -> tuple[str, str | None]:
    """Convenience wrapper: returns ``(validated_text, error_or_none)``.

    If validation passes, returns the original text unchanged.

    **HTML rescue path**: if the raw text fails due to ``html_table_markup``,
    HTML tags are stripped and the residual is re-validated.  If the stripped
    text passes (CJK ratio ≥ ``_HTML_STRIP_CJK_THRESHOLD``), the **stripped**
    text is returned as valid — so bibliography pages wrapped in ``<table>``
    tags are salvaged rather than discarded.

    **Traditional-script check**: when ``expected_script='traditional'``,
    :func:`check_traditional_script` is run after the main validation.  An
    output that passes the hallucination check but contains a high proportion
    of simplified-exclusive characters is flagged as
    ``'validation_failed:simplified_conversion:N/T'`` and zeroed, so it does
    not corrupt the fusion alignment for primary-tier facsimiles.

    If all validation attempts fail, returns ``('', 'validation_failed:<reason>')``.

    Engine wrappers call this instead of :func:`validate_llm_ocr_output`
    directly to avoid repeating the if-branch logic.

    Args:
        text: Stripped LLM output text.
        expected_script: When ``'traditional'``, also run the simplified-
            conversion detector.  ``None`` skips the script check.

    Returns:
        ``(text_to_use, error_string_or_none)``
    """
    is_valid, reason = validate_llm_ocr_output(text)
    if is_valid:
        # For empty-page marker responses, return empty string so the pipeline
        # sets status='empty' rather than keeping "（空页）" as ok-text.
        stripped = text.strip()
        if stripped in _EMPTY_RESPONSES:
            return "", None

        # Traditional-script check (primary-tier pages only).
        if expected_script == "traditional":
            is_trad, trad_reason = check_traditional_script(stripped)
            if not is_trad:
                return "", f"validation_failed:{trad_reason}"

        return text, None

    # HTML rescue: strip tags + entities, re-validate residual
    if reason == "html_table_markup":
        stripped_html = _HTML_TAG_RE.sub(" ", text)
        stripped_html = _HTML_ENTITY_RE.sub(" ", stripped_html)
        stripped_html = re.sub(r" {2,}", " ", stripped_html).strip()
        if stripped_html and _cjk_ratio(stripped_html) >= _HTML_STRIP_CJK_THRESHOLD:
            is_stripped_valid, stripped_reason = validate_llm_ocr_output(stripped_html)
            if is_stripped_valid:
                # Also run script check on the rescued text.
                if expected_script == "traditional":
                    is_trad, trad_reason = check_traditional_script(stripped_html)
                    if not is_trad:
                        return "", f"validation_failed:{trad_reason}"
                return stripped_html, None

    return "", f"validation_failed:{reason}"
