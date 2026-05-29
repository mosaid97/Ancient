"""Edition + editorial-layer metadata extraction from filenames + (optionally) LLM.

Plan §6 Stage 1 calls for ``DOCUMENT.edition`` / ``publisher`` /
``publication_year`` / ``editorial_layers`` populated from filename + LLM
extraction. This module ships the cheap deterministic regex pass; the LLM
enrichment is opt-in (off by default to control Silra spend) and falls back
gracefully.

Editorial-layer ontology (plan §2.6):

- ``校點`` — punctuation only.
- ``校訂`` — collation against editions; carries variant apparatus.
- ``箋解`` — scholarly commentary.
- ``疏議`` — state-issued traditional commentary (e.g., 唐律疏議's 疏議 layer).
- ``pure-source`` — no editorial layer; pure ancient text.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

EditorialLayerType = Literal["校點", "校訂", "箋解", "疏議", "pure-source"]


@dataclass
class EditorialLayer:
    type: EditorialLayerType
    author: str = ""
    year: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "author": self.author, "year": self.year}


@dataclass
class EditionMetadata:
    title: str
    edition: str = ""
    publisher: str = ""
    publication_year: int | None = None
    publication_period: str = ""
    editorial_layers: list[EditorialLayer] = field(default_factory=list)
    primary_author: str = ""
    secondary_author: str = ""
    raw_filename: str = ""
    confidence: float = 0.0
    extracted_via: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "edition": self.edition,
            "publisher": self.publisher,
            "publication_year": self.publication_year,
            "publication_period": self.publication_period,
            "editorial_layers": [el.to_dict() for el in self.editorial_layers],
            "primary_author": self.primary_author,
            "secondary_author": self.secondary_author,
            "raw_filename": self.raw_filename,
            "confidence": self.confidence,
            "extracted_via": self.extracted_via,
        }


# ---------------------------------------------------------------------------
# Regex-based extraction (cheap, deterministic, ~70% coverage on the corpus).
# ---------------------------------------------------------------------------

# Strip the common library / mirror suffixes injected by z-library / 1lib.
_LIB_SUFFIXES_RE = re.compile(
    r"\s*\(?\s*(z-library\.sk|1lib\.sk|z-lib\.sk|z-lib\.org|Z-Library|"
    r"libgen|annas-archive)\b[^)]*\)?",
    re.IGNORECASE,
)
_TRAILING_DUP_INDEX_RE = re.compile(r"\(\d+\)$")
# Outer paren groups are half-width "(...)"; full-width "（...）" are content
# (commonly dynasty markers like "（五代）" inside an author group).
_PAREN_GROUP_RE = re.compile(r"\(([^()]+)\)")
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
# Inner full-width parens (e.g. dynasty markers like 『（五代）』).
_INNER_PAREN_RE = re.compile(r"[（(][^()）]*[)）]")
# Some titles are wrapped in full-width 《...》 brackets — strip them.
_TITLE_BRACKET_RE = re.compile(r"^[《【\[](.+?)[》】\]]$")
# Filenames like "<author>_<title>" (modern academic papers).
_AUTHOR_TITLE_UNDERSCORE_RE = re.compile(r"^([\u4e00-\u9fff]{2,5})_(.+)$")
# Author-role suffixes commonly tacked onto names in 古籍 filenames.
_AUTHOR_ROLE_SUFFIX_RE = re.compile(
    r"(撰|著|編|编|注|疏|箋|笺|釋|释|主編|主编|校點|校点|校訂|校订|"
    r"點校|点校|箋解|笺解|校注)+\s*$"
)

# Editorial-layer keyword cues — order matters (most specific first).
# Each entry is (keyword, EditorialLayerType).
_EDITORIAL_CUES: list[tuple[str, EditorialLayerType]] = [
    ("疏議", "疏議"),
    ("疏议", "疏議"),
    ("箋解", "箋解"),
    ("笺解", "箋解"),
    ("箋注", "箋解"),
    ("校訂", "校訂"),
    ("校订", "校訂"),
    ("点校", "校點"),
    ("點校", "校點"),
    ("校點", "校點"),
    ("校点", "校點"),
    ("補編", "校訂"),
    ("补编", "校訂"),
    ("校注", "校訂"),
    ("校释", "箋解"),
    ("校釋", "箋解"),
]

# Known publisher cues. Tang-era reprints are dominated by 中華書局 / 上海古籍 /
# 商務印書館 / 北京大學出版社. The list is small on purpose; LLM enrichment
# fills the long tail.
_KNOWN_PUBLISHERS = [
    "中華書局",
    "中华书局",
    "上海古籍",
    "商務印書館",
    "商务印书馆",
    "北京大學出版社",
    "北京大学出版社",
    "北京師範大學出版社",
    "北京师范大学出版社",
    "上海人民出版社",
    "人民出版社",
    "鳳凰出版社",
    "凤凰出版社",
]


def _clean_filename_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = _LIB_SUFFIXES_RE.sub("", stem)
    stem = _TRAILING_DUP_INDEX_RE.sub("", stem).strip()
    return stem.strip(" -_.")


def _split_title_and_metadata(stem: str) -> tuple[str, list[str], str]:
    """Heuristic split into ``(title, paren_groups, prefix_author)``.

    Handles three filename shapes:

    1. ``<title> (<group1>) (<group2>) ...``  — classical 古籍 reprints.
    2. ``<author>_<title>``                    — modern academic papers
       (no parens; surface ``prefix_author`` so the caller can fill the
       primary author slot).
    3. ``《<title>》 (...)`` / ``【...】 ...``    — title in full-width brackets.
    """
    prefix_author = ""
    candidate = stem
    paren_match = re.search(r"\(", candidate)
    if paren_match:
        title = candidate[: paren_match.start()].strip()
        paren_groups = _PAREN_GROUP_RE.findall(candidate[paren_match.start():])
        groups = [p.strip() for p in paren_groups if p.strip()]
    else:
        title = candidate.strip()
        groups = []

    # Pull full-width paren content from inside the title into its own group,
    # then strip it from the displayed title (e.g. "册府元龟（点校本 校订本）"
    # -> title="册府元龟", group="点校本 校订本").
    fw_groups = re.findall(r"（([^（）]+)）", title)
    for fwg in fw_groups:
        cleaned = fwg.strip()
        if cleaned:
            groups.append(cleaned)
    title = re.sub(r"（[^（）]+）", "", title).strip()

    bracket_m = _TITLE_BRACKET_RE.match(title)
    if bracket_m:
        title = bracket_m.group(1).strip()

    # If there are no paren groups, also try the "<author>_<title>" form.
    if not groups:
        m = _AUTHOR_TITLE_UNDERSCORE_RE.match(title)
        if m:
            prefix_author, title = m.group(1), m.group(2).strip()

    return title, groups, prefix_author


def _detect_editorial_layers(text: str) -> list[EditorialLayer]:
    found: list[EditorialLayer] = []
    seen: set[str] = set()
    for cue, layer_type in _EDITORIAL_CUES:
        if cue in text and layer_type not in seen:
            found.append(EditorialLayer(type=layer_type))
            seen.add(layer_type)
    return found


def _detect_publisher(text: str) -> str:
    for pub in _KNOWN_PUBLISHERS:
        if pub in text:
            return pub
    return ""


def _detect_year(text: str) -> int | None:
    matches = _YEAR_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def _looks_like_author(token: str) -> tuple[bool, str]:
    """Return ``(is_author, cleaned_name)`` for a candidate token.

    Strips inner dynasty markers (e.g. ``（五代）``), trailing role markers
    (``撰``/``著``/``校點``…), and any whitespace. A cleaned token of length
    2-6 with no ASCII characters counts as an author.
    """
    cleaned = _INNER_PAREN_RE.sub("", token)
    cleaned = _AUTHOR_ROLE_SUFFIX_RE.sub("", cleaned).strip()
    if not cleaned:
        return False, ""
    if 2 <= len(cleaned) <= 6 and not any(ch.isascii() for ch in cleaned):
        return True, cleaned
    return False, ""


def extract_from_filename(filename: str) -> EditionMetadata:
    """Cheap deterministic extraction from a filename.

    Args:
        filename: Just the file/dir name, with or without extension.

    Returns:
        :class:`EditionMetadata` populated to whatever the regex layer can
        infer. ``confidence`` is bounded to ``[0, 0.7]`` for filename-only
        extraction; LLM enrichment can push it higher.
    """
    stem = _clean_filename_stem(filename)
    title, paren_groups, prefix_author = _split_title_and_metadata(stem)

    full = " ".join([title, *paren_groups])

    publisher = _detect_publisher(full)
    year = _detect_year(full)
    layers = _detect_editorial_layers(full)

    primary_author = prefix_author
    secondary_author = ""
    edition = ""
    edition_tags = (
        "本",
        "校點",
        "校点",
        "校訂",
        "校订",
        "點校",
        "点校",
        "箋解",
        "笺解",
        "校注",
        "影印",
        "宋元",
        "明刊",
        "清刊",
    )
    for group in paren_groups:
        sub = group.strip()
        if not sub:
            continue
        # Some groups mix authors and edition markers (e.g.
        # "（五代）王定保 撰 阳羡生校点"). Tokenize first so edition tokens
        # don't poison the whole group.
        tokens = [t for t in re.split(r"[；;,，、 ]+", sub) if t.strip()]
        non_edition_tokens = [
            t for t in tokens if not any(tag in t for tag in edition_tags)
        ]
        if (
            not edition
            and non_edition_tokens != tokens
            and any(any(tag in t for tag in edition_tags) for t in tokens)
            and not non_edition_tokens
        ):
            # Pure edition group like "(点校本 校订本)".
            edition = sub
            continue
        if not edition and not non_edition_tokens:
            edition = sub
            continue
        if not edition and len(non_edition_tokens) < len(tokens):
            edition_token = next(
                (t for t in tokens if any(tag in t for tag in edition_tags)),
                "",
            )
            if edition_token:
                edition = edition_token
        for c in non_edition_tokens:
            ok, cleaned = _looks_like_author(c)
            if not ok:
                continue
            if not primary_author:
                primary_author = cleaned
            elif not secondary_author and cleaned != primary_author:
                secondary_author = cleaned

    period = ""
    if year:
        if year >= 2000:
            period = "modern-zh-reprint-2000s"
        elif year >= 1980:
            period = "modern-zh-reprint-1980s-1990s"
        elif year >= 1949:
            period = "modern-zh-reprint-PRC"
        else:
            period = f"pre-1949-{year // 10 * 10}s"

    confidence = 0.0
    if title:
        confidence += 0.2
    if publisher:
        confidence += 0.15
    if year:
        confidence += 0.15
    if layers:
        confidence += 0.1
    if primary_author:
        confidence += 0.1

    return EditionMetadata(
        title=title or stem,
        edition=edition,
        publisher=publisher,
        publication_year=year,
        publication_period=period,
        editorial_layers=layers,
        primary_author=primary_author,
        secondary_author=secondary_author,
        raw_filename=filename,
        confidence=min(confidence, 0.7),
        extracted_via=["filename_regex"],
    )


# ---------------------------------------------------------------------------
# EPUB OPF enrichment (zero-cost, reads the already-open EPUB manifest).
# ---------------------------------------------------------------------------


def enrich_from_opf(base: EditionMetadata, opf_hints: dict[str, Any]) -> EditionMetadata:
    """Merge EPUB OPF Dublin-Core fields into an existing :class:`EditionMetadata`.

    The EPUB's ``<dc:creator>`` string often contains the editorial cue (e.g.
    ``（五代）王定保 撰 阳羡生校点``) that would normally be parsed from a rich
    filename.  When the corpus filename has been stripped to the bare title
    (per the 2026-05-18 renaming convention), the OPF is the only place that
    survives intact.

    Merge semantics: already-populated fields in *base* are **not** overwritten;
    OPF only fills the gaps and appends new editorial-layer types.

    Args:
        base: Pre-computed regex extraction (from :func:`extract_from_filename`).
        opf_hints: Dict from ``epub_reader.read_opf_hints`` with keys
            ``creators``, ``publishers``, ``titles``, ``dates``
            (each a list of strings).

    Returns:
        New :class:`EditionMetadata` instance.  ``extracted_via`` gains
        ``"epub_opf"`` when at least one OPF field changed something.
    """
    creators: list[str] = opf_hints.get("creators") or []
    publishers_opf: list[str] = opf_hints.get("publishers") or []
    dates_opf: list[str] = opf_hints.get("dates") or []

    # --- 1. Editorial layers from creator strings ---
    existing_types: set[str] = {el.type for el in base.editorial_layers}
    extra_layers: list[EditorialLayer] = []
    for creator in creators:
        for layer in _detect_editorial_layers(creator):
            if layer.type in existing_types:
                continue
            # Attribute the layer: find the last CJK token that looks like an
            # author after stripping inner parens + the matched role suffix.
            stripped = _INNER_PAREN_RE.sub("", creator).strip()
            for token in reversed(re.split(r"\s+", stripped)):
                ok, name = _looks_like_author(token)
                if ok:
                    layer.author = name
                    break
            extra_layers.append(layer)
            existing_types.add(layer.type)

    merged_layers = base.editorial_layers + extra_layers

    # --- 2. Publisher ---
    publisher = base.publisher
    if not publisher:
        # Prefer a known publisher string; fall back to first OPF value.
        for pub in publishers_opf:
            detected = _detect_publisher(pub)
            if detected:
                publisher = detected
                break
        if not publisher and publishers_opf:
            publisher = publishers_opf[0].strip()

    # --- 3. Publication year ---
    year = base.publication_year
    if year is None:
        for date_str in dates_opf:
            y = _detect_year(date_str)
            if y:
                year = y
                break

    # --- 4. Authors from creator strings ---
    primary_author = base.primary_author
    secondary_author = base.secondary_author
    for creator in creators:
        # Skip tool-generated / URL entries (Epubor, eCore, http://…).
        if any(skip in creator for skip in ("Epubor", "eCore", "http://")):
            continue
        stripped = _INNER_PAREN_RE.sub("", creator).strip()
        for token in re.split(r"\s+", stripped):
            ok, name = _looks_like_author(token)
            if not ok:
                continue
            if not primary_author:
                primary_author = name
            elif not secondary_author and name != primary_author:
                secondary_author = name

    # --- 5. Derive period if year changed ---
    period = base.publication_period
    if year and not period:
        if year >= 2000:
            period = "modern-zh-reprint-2000s"
        elif year >= 1980:
            period = "modern-zh-reprint-1980s-1990s"
        elif year >= 1949:
            period = "modern-zh-reprint-PRC"
        else:
            period = f"pre-1949-{year // 10 * 10}s"

    # --- 6. Confidence + provenance ---
    changed = bool(
        extra_layers
        or (publisher and not base.publisher)
        or (year and base.publication_year is None)
        or (primary_author and not base.primary_author)
    )
    confidence = base.confidence
    if changed:
        confidence = min(confidence + 0.15, 0.7)
    extracted_via = [*base.extracted_via, "epub_opf"] if changed else base.extracted_via[:]

    return EditionMetadata(
        title=base.title,
        edition=base.edition,
        publisher=publisher,
        publication_year=year,
        publication_period=period,
        editorial_layers=merged_layers,
        primary_author=primary_author,
        secondary_author=secondary_author,
        raw_filename=base.raw_filename,
        confidence=confidence,
        extracted_via=extracted_via,
    )


# ---------------------------------------------------------------------------
# Optional LLM enrichment (deepseek-chat).
# ---------------------------------------------------------------------------

LLM_PROMPT_TEMPLATE = """你是 Ancient China 知识图谱项目的元数据抽取助手。
请从下面的文件名中抽取古籍版本信息，返回严格的 JSON。

文件名: "{filename}"
首页前 800 字 (可能为空):
\"\"\"
{snippet}
\"\"\"

返回 JSON 字段:
- title: 书名（不含编辑/著者/出版社/年份等）
- edition: 版本说明（如「点校本」「校订本」「影印本」）
- publisher: 出版社（中华书局／上海古籍出版社 等），未知则空字串
- publication_year: 整数公历年；未知则 null
- primary_author: 原始作者（古代作者，如「魏徵」「欧阳修」）
- secondary_author: 现代编校者（如「劉俊文」「王钦若」）
- editorial_layers: 列表，元素为 {{type, author, year}}, type ∈ {{校點,校訂,箋解,疏議,pure-source}}
- confidence: 0-1 浮点
仅返回 JSON,不要 markdown 代码块。
"""


def llm_enrich(
    filename: str,
    snippet: str = "",
    *,
    base: EditionMetadata | None = None,
    max_retries: int = 2,
) -> EditionMetadata:
    """Enrich a base regex extraction by asking deepseek-chat.

    Falls back silently to ``base`` (or filename-extracted) if the LLM call
    fails or its response can't be parsed. Cheap (a few hundred tokens).

    Args:
        filename: File/dir name.
        snippet: Optional first-page text snippet (helps the model).
        base: Pre-computed regex extraction; if ``None``, run :func:`extract_from_filename`.
        max_retries: Retry budget for the Silra call.

    Returns:
        :class:`EditionMetadata` with ``extracted_via`` updated to include
        ``"deepseek-chat"`` if the enrichment succeeded.
    """
    base = base or extract_from_filename(filename)
    try:
        from apps.backend.llm.silra import chat_completion
    except ImportError as exc:
        logger.warning("Silra client unavailable; skipping llm_enrich: %s", exc)
        return base

    prompt = LLM_PROMPT_TEMPLATE.format(filename=filename, snippet=(snippet or "")[:800])
    try:
        resp = chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
            max_retries=max_retries,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("llm_enrich failed (%s); returning base: %s", type(exc).__name__, exc)
        return base
    raw = (resp.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("llm_enrich: response was not valid JSON: %s", raw[:200])
        return base

    layers_raw = data.get("editorial_layers") or []
    layers: list[EditorialLayer] = []
    for entry in layers_raw:
        if not isinstance(entry, dict):
            continue
        type_ = entry.get("type", "")
        if type_ in {"校點", "校訂", "箋解", "疏議", "pure-source"}:
            layers.append(
                EditorialLayer(
                    type=type_,  # type: ignore[arg-type]
                    author=str(entry.get("author") or ""),
                    year=entry.get("year"),
                )
            )

    title = data.get("title") or base.title
    edition = data.get("edition") or base.edition
    publisher = data.get("publisher") or base.publisher
    year = data.get("publication_year") or base.publication_year
    primary = data.get("primary_author") or base.primary_author
    secondary = data.get("secondary_author") or base.secondary_author
    confidence = float(data.get("confidence", base.confidence + 0.2))

    period = base.publication_period
    if year:
        if year >= 2000:
            period = "modern-zh-reprint-2000s"
        elif year >= 1980:
            period = "modern-zh-reprint-1980s-1990s"
        elif year >= 1949:
            period = "modern-zh-reprint-PRC"
        else:
            period = f"pre-1949-{int(year) // 10 * 10}s"

    return EditionMetadata(
        title=title,
        edition=edition,
        publisher=publisher,
        publication_year=year,
        publication_period=period,
        editorial_layers=layers or base.editorial_layers,
        primary_author=primary,
        secondary_author=secondary,
        raw_filename=filename,
        confidence=min(max(confidence, base.confidence), 0.99),
        extracted_via=[*base.extracted_via, "deepseek-chat"],
    )
