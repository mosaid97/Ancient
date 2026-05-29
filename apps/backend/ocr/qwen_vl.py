"""Qwen-VL-OCR via Silra's OpenAI-compatible vision endpoint (plan §6 Stage 3c+).

``qwen-vl-ocr-latest`` is Alibaba's OCR-specialised fine-tune of Qwen-VL,
designed explicitly for document transcription rather than image description.
It is used as a **third OCR engine** alongside PaddleOCR and DeepSeek-OCR so
the 3-way fusion can gate on confidence and fall back to Paddle when both LLM
engines return low-quality output.

Key differences from DeepSeek-OCR (``silra_deepseek.py``):

- Model default is ``qwen-vl-ocr-latest`` (env ``QWEN_VL_OCR_MODEL``).
- The system prompt is slightly simpler and adds explicit prohibitions against
  table formatting, coordinate sequences, and image description — the three
  main DeepSeek hallucination classes discovered during Phase 3 audit.
- ``max_tokens`` default is 4096 (same) but Qwen-VL-OCR is more conservative
  and naturally returns shorter output when uncertain rather than hallucinating.
- Engine label is ``'qwen_vl_ocr'`` everywhere to distinguish from DeepSeek.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any

import cv2
import numpy as np
from openai import OpenAI

from apps.backend.llm.silra import _retry, get_silra_client
from apps.backend.ocr.base import OCRLine, OCRPageResult
from apps.backend.ocr.validate import apply_validation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts — one per script variant
# ---------------------------------------------------------------------------
# Corpus audit (2026-05-23) confirmed that Qwen-VL-OCR intermittently converts
# traditional Chinese characters to simplified when the system prompt is written
# in simplified Chinese and only weakly prohibits the conversion.  The fix:
#   1. Write the traditional-script prompt IN traditional Chinese (consistent signal).
#   2. Use POSITIVE assertion ("must output traditional") not just prohibition.
#   3. List the 24 most common trad/simp problem pairs explicitly.
#   4. Add a separate simplified prompt for modern secondary sources so the model
#      receives a script-coherent context regardless of the source corpus.
# ---------------------------------------------------------------------------

# ── Traditional Chinese facsimiles (primary-tier 影印本, woodblock prints) ───
_SYSTEM_PROMPT_TRADITIONAL_ZH = (
    "你是精通中國古籍影印本 OCR 的專業助手，專責處理唐代及鄰近朝代的典籍、文書與寫本。\n\n"
    "【核心要求：繁體字輸出】\n"
    "本頁圖像來源於古籍影印本，文字為繁體漢字（Traditional Chinese）。\n"
    "✦ 識別到的每一個字，必須輸出其繁體形式。\n"
    "✦ 絕對禁止將繁體字轉換為簡體字。\n"
    "✦ 遇到異體字（如「爲」「體」「與」「於」「無」「爾」），保留圖像中出現的原字，不予替換。\n\n"
    "常見繁簡易混淆字對（左為正確繁體，右為禁用簡體）：\n"
    "議→禁用「议」｜書→禁用「书」｜國→禁用「国」｜會→禁用「会」｜發→禁用「发」｜\n"
    "傳→禁用「传」｜體→禁用「体」｜當→禁用「当」｜說→禁用「说」｜時→禁用「时」｜\n"
    "學→禁用「学」｜從→禁用「从」｜應→禁用「应」｜經→禁用「经」｜處→禁用「处」｜\n"
    "總→禁用「总」｜關→禁用「关」｜節→禁用「节」｜點→禁用「点」｜問→禁用「问」｜\n"
    "兩→禁用「两」｜術→禁用「术」｜難→禁用「难」｜號→禁用「号」｜實→禁用「实」｜\n"
    "開→禁用「开」｜長→禁用「长」｜貞→禁用「贞」｜頒→禁用「颁」｜編→禁用「编」\n\n"
    "嚴格規則：\n"
    "1. **只輸出識別到的原文字符**，不翻譯、不解釋、不添加任何注釋或標題。\n"
    "2. 按古籍閱讀順序（從右至左、自上而下）輸出原文。\n"
    "3. 行間用換行（\\n）分隔；不同列之間用空行分隔。\n"
    "4. 若字符模糊不清，用「□」代替；絕對不要猜測或編造字符。\n"
    "5. 若整頁無文字（純圖片、空白頁、照片），僅回覆「（空頁）」。\n"
    "6. **禁止**：不要輸出 Markdown 表格（|...|）、不要輸出數字序列、"
    "不要描述圖片內容、不要重複同一個字符。\n"
    "7. 邊欄、版心、天頭、地腳的小字按出現順序附於正文末尾，以「【注】」開頭。"
)

# ── Modern simplified-Chinese documents (secondary-tier scholarly works) ────
_SYSTEM_PROMPT_SIMPLIFIED_ZH = (
    "你是精通现代汉语学术文献 OCR 的助手，专门处理唐代相关现代学术著作、"
    "论文与研究文集。\n\n"
    "给定一张文献页面图像，请逐字识别所有可见汉字，按页面阅读顺序输出原文。\n\n"
    "严格规则：\n"
    "1. **只输出识别到的原文字符**，不翻译、不解释、不添加任何注释或标题。\n"
    "2. 保留文档中出现的字符原貌（简体或繁体），不做转换。\n"
    "3. 行间用换行（\\n）分隔；不同栏/列之间用空行分隔。\n"
    "4. 若字符模糊不清，用「□」代替；绝对不要猜测或编造字符。\n"
    "5. 若整页无文字（纯图片、空白页），仅回复「（空页）」。\n"
    "6. **禁止**：不要使用 HTML 标签、不要输出 Markdown 表格（|...|）、"
    "不要描述图片内容、不要重复同一个字符。\n"
    "7. 参考文献、书目、注释、脚注等内容直接以纯文本逐行输出，不使用任何表格格式。\n"
    "8. 行号处理：页边阿拉伯数字行号（1、2、3…）是版式标记，请忽略，只输出行号后的文字。\n"
    "9. 带编号脚注/参考文献（①②③ 或 (1)(2)(3)）逐条纯文本输出，每条一行，不使用表格。"
)

# ── Classical Chinese (mixed/unknown script — generic fallback) ──────────────
# Used when tier or script is ambiguous.  Strengthened anti-simplification rule
# compared to the original prompt.
_SYSTEM_PROMPT_CLASSICAL_ZH = (
    "你是一位精通中國古籍影印本 OCR 的助手，專門處理唐代及周邊朝代的典籍、文書與寫本。\n\n"
    "給定一張古籍頁面圖像，請逐字識別所有可見漢字，按古籍閱讀順序（從右至左、自上而下）"
    "輸出原文。\n\n"
    "嚴格規則：\n"
    "1. **只輸出識別到的原文字符**，不翻譯、不解釋、不添加任何注釋或標題。\n"
    "2. 若圖像中出現繁體字，必須輸出繁體字；絕對不得將繁體字轉換為簡體字。"
    "若圖像為簡體字則照樣輸出簡體。保留所有異體字、避諱字、通假字的原貌。\n"
    "3. 行間用換行（\\n）分隔；不同列之間用空行分隔。\n"
    "4. 若字符模糊不清，用「□」代替；絕對不要猜測或編造字符。\n"
    "5. 若整頁無文字（純圖片、空白頁、照片），僅回覆「（空頁）」。\n"
    "6. **禁止**：不要輸出 Markdown 表格（|...|）、不要輸出數字序列、"
    "不要描述圖片內容、不要重複同一個字符。\n"
    "7. 邊欄、版心、天頭、地腳的小字按出現順序附於正文末尾，以「【注】」開頭。"
)

# ── Kanbun (Japanese-edited Tang editions with kunten marks) ────────────────
_SYSTEM_PROMPT_KANBUN = (
    "あなたは漢文（kanbun）OCR の専門家で、日本刊行の唐代古籍影印本を扱います。\n\n"
    "画像中のすべての漢字・送り仮名・返り点（レ点・一二点・上下点）を認識し、"
    "原文の読み順で出力してください。\n\n"
    "厳格ルール：\n"
    "1. 認識した原文のみ出力。翻訳・解釈・注釈不要。\n"
    "2. 異体字・避諱字をそのまま保持し、現代化しない。\n"
    "3. 返り点・送り仮名は本字の直後に括弧で示す（例：「読(レ)書」）。\n"
    "4. 判読不能な字は「□」で示す。推測・創作厳禁。\n"
    "5. 全頁が空白なら「（空頁）」のみ返答。\n"
    "6. Markdown テーブル・数字列・画像説明文の出力禁止。"
)

_ENGINE = "qwen_vl_ocr"
_DEFAULT_MODEL_ENV = "QWEN_VL_OCR_MODEL"
_DEFAULT_MODEL = "qwen-vl-ocr-latest"


def _encode_image(image: np.ndarray | bytes, *, fmt: str = "png") -> str:
    """Return a ``data:image/<fmt>;base64,...`` URI."""
    if isinstance(image, np.ndarray):
        ok, buf = cv2.imencode(f".{fmt}", image)
        if not ok:
            raise RuntimeError(f"cv2.imencode failed for ({fmt})")
        payload = buf.tobytes()
    elif isinstance(image, (bytes, bytearray, memoryview)):
        payload = bytes(image)
    elif isinstance(image, io.IOBase):
        payload = image.read()
    else:
        raise TypeError(f"_encode_image: unsupported input type {type(image).__name__}")
    b64 = base64.b64encode(payload).decode("ascii")
    return f"data:image/{fmt};base64,{b64}"


def _system_prompt(
    language_hint: str | None,
    script_hint: str | None = None,
) -> str:
    """Return the appropriate system prompt for the given language / script context.

    Args:
        language_hint: Broad language tag from PAGE.language (e.g. ``'zh-classical'``,
            ``'zh-modern'``, ``'ja'``, ``'kanbun'``).
        script_hint: Fine-grained script variant, takes precedence over
            ``language_hint`` when set.  Accepted values:

            * ``'traditional'`` — force繁體 prompt (primary-tier 影印本).
            * ``'simplified'``  — use simplified-Chinese prompt (modern secondaries).
            * ``'classical'``   — generic fallback (mixed / unknown).
            * ``None``          — derive from ``language_hint``.
    """
    if script_hint == "traditional":
        return _SYSTEM_PROMPT_TRADITIONAL_ZH
    if script_hint == "simplified":
        return _SYSTEM_PROMPT_SIMPLIFIED_ZH
    if script_hint == "classical":
        return _SYSTEM_PROMPT_CLASSICAL_ZH
    # Fall back to language-based selection
    if language_hint in {"ja", "kanbun", "japan", "jpn"}:
        return _SYSTEM_PROMPT_KANBUN
    if language_hint == "zh-modern":
        return _SYSTEM_PROMPT_SIMPLIFIED_ZH
    return _SYSTEM_PROMPT_CLASSICAL_ZH


def qwen_vl_ocr_page(
    image: np.ndarray | bytes,
    *,
    page_id: str,
    language_hint: str | None = "zh-classical",
    script_hint: str | None = None,
    client: OpenAI | None = None,
    model: str | None = None,
    max_retries: int = 3,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout: float = 180.0,
) -> OCRPageResult:
    """OCR one preprocessed page via Silra's Qwen-VL-OCR endpoint.

    Args:
        image: A preprocessed page image — either a BGR uint8
            :class:`numpy.ndarray` or raw PNG / JPEG bytes.
        page_id: The Neo4j PAGE id (stored on the result).
        language_hint: Language tag from ``PAGE.language`` used for prompt
            selection when ``script_hint`` is ``None``.
        script_hint: Fine-grained script variant (``'traditional'``,
            ``'simplified'``, ``'classical'``).  Takes precedence over
            ``language_hint`` when provided.  Pass ``'traditional'`` for
            primary-tier 影印本 to prevent character simplification.
        client: Re-use an existing Silra client.
        model: Override the default model (``qwen-vl-ocr-latest``).
        max_retries: Retry on transient Silra errors.
        max_tokens: Bounds the response size.
        temperature: 0.0 for deterministic transcription.
        timeout: Per-request timeout in seconds.

    Returns:
        An :class:`OCRPageResult` with ``engine='qwen_vl_ocr'``.
    """
    started = time.monotonic()
    client = client or get_silra_client(timeout=timeout)
    model_name = model or os.getenv(_DEFAULT_MODEL_ENV, _DEFAULT_MODEL)

    data_uri = _encode_image(image, fmt="png")
    system_prompt = _system_prompt(language_hint, script_hint=script_hint)

    # User instruction is written in the same script as the system prompt to
    # avoid conflicting register cues that would make the model revert to simplified.
    if script_hint == "traditional" or (
        script_hint is None
        and language_hint not in {"ja", "kanbun", "japan", "jpn", "zh-modern"}
    ):
        user_instruction = "請按上述要求識別本頁面所有字符，只輸出原文。"
    else:
        user_instruction = "请按上述要求识别本页面所有字符，只输出原文。"

    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": data_uri},
        },
        {
            "type": "text",
            "text": user_instruction,
        },
    ]

    try:
        response = _retry(
            client.chat.completions.create,
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_retries=max_retries,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - started
        logger.warning("qwen_vl_ocr_page %s failed: %s", page_id, exc)
        return OCRPageResult(
            engine=_ENGINE,
            model_version=model_name,
            page_id=page_id,
            text="",
            confidence=0.0,
            char_count=0,
            language_hint=language_hint,
            duration_seconds=round(elapsed, 3),
            metadata={"prompt_kind": "kanbun" if "kanbun" in (language_hint or "") else "classical_zh"},
            error=f"{type(exc).__name__}: {exc}",
        )

    elapsed = time.monotonic() - started
    raw_text = (response.choices[0].message.content or "").strip()

    # Inline hallucination guard + traditional-script check (runs before any Neo4j write).
    text, validation_error = apply_validation(
        raw_text,
        expected_script=script_hint if script_hint == "traditional" else None,
    )

    lines: list[OCRLine] = []
    for idx, line in enumerate(text.split("\n")):
        chunk = line.strip()
        if chunk:
            lines.append(OCRLine(text=chunk, confidence=1.0, order=idx))

    usage = getattr(response, "usage", None)
    metadata: dict[str, Any] = {
        "prompt_kind": "kanbun" if "kanbun" in (language_hint or "") else "classical_zh",
        "finish_reason": getattr(response.choices[0], "finish_reason", None),
    }
    if usage is not None:
        metadata["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    if validation_error:
        metadata["validation_error"] = validation_error
        logger.debug("qwen_vl_ocr_page %s failed validation: %s", page_id, validation_error)

    confidence = 0.82 if text else 0.0

    return OCRPageResult(
        engine=_ENGINE,
        model_version=model_name,
        page_id=page_id,
        text=text,
        lines=lines,
        confidence=confidence,
        char_count=len(text),
        language_hint=language_hint,
        duration_seconds=round(elapsed, 3),
        metadata=metadata,
        error=validation_error,
    )
