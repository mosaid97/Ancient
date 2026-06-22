"""LLM-based relevance + evidence scorer (NOT the citation gate).

This module sends candidate chunks to ``deepseek-chat`` and asks it to
score each one on:

- **relevance** (0–10): does this chunk address the query topic at all?
- **evidence** (0–10): does this chunk contain specific, citable evidence
  (date, law clause, name, event, policy description) that would support
  answering the query?

A combined ``verification_score`` = 0.5 × relevance + 0.5 × evidence is
computed.  Chunks below ``min_score`` are filtered out; survivors are
re-ranked by ``verification_score × cosine_score``.

This is **not** the deterministic zero-hallucination citation verifier;
that lives in :mod:`apps.backend.agents.verifier` and answers a different
question ("is this span literally present in the source?"). Keep the two
mental models separate.

Public API
----------
score_evidence(query, results, *, model, max_chunks, min_score, client)
    -> (list[EvidenceItem], EvidenceRunReport)

EvidenceItem              — scored result dataclass
EvidenceRunReport         — summary of one scorer run

Per AGENTS.md §5: deepseek-chat for reasoning tasks; exponential backoff via
silra.chat_completion.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from apps.backend.llm.silra import ANCIENT_CHINA_SYSTEM_PROMPT, chat_completion, get_silra_client
from apps.backend.pipeline.search import SearchResult

log = logging.getLogger(__name__)

_MAX_CHUNKS_PER_CALL = 10   # keep prompts manageable
_DEFAULT_MIN_SCORE = 3.0    # out of 10
_CHUNK_TEXT_LIMIT = 400     # chars shown to the scorer per chunk


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvidenceItem:
    """A search result augmented with LLM relevance + evidence scores."""

    result: SearchResult
    relevance: float    # LLM-assigned 0–10
    evidence: float     # LLM-assigned 0–10
    reasoning: str      # LLM's brief justification (1–2 sentences)
    verification_score: float = 0.0   # 0.5 × relevance + 0.5 × evidence

    # Final combined rank score: verification_score × cosine_score
    combined_score: float = 0.0

    def __post_init__(self) -> None:
        self.verification_score = round(
            0.5 * self.relevance + 0.5 * self.evidence, 3
        )
        self.combined_score = round(
            (self.verification_score / 10.0) * self.result.score, 6
        )

    def to_dict(self) -> dict[str, Any]:
        d = self.result.to_dict()
        d.update(
            {
                "relevance": self.relevance,
                "evidence": self.evidence,
                "reasoning": self.reasoning,
                "verification_score": self.verification_score,
                "combined_score": self.combined_score,
                "trust_score": self.result.spine.trust_score(),
            }
        )
        return d


@dataclass
class EvidenceRunReport:
    """Summary of an evidence-scoring run."""

    query: str
    chunks_input: int = 0
    chunks_scored: int = 0
    chunks_filtered: int = 0
    chunks_failed: int = 0
    chunks_missing_from_llm: int = 0
    llm_calls: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    scoredAt: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "chunks_input": self.chunks_input,
            "chunks_scored": self.chunks_scored,
            "chunks_filtered": self.chunks_filtered,
            "chunks_failed": self.chunks_failed,
            "chunks_missing_from_llm": self.chunks_missing_from_llm,
            "llm_calls": self.llm_calls,
            "duration_seconds": round(self.duration_seconds, 3),
            "errors": self.errors[:20],
            "scoredAt": self.scoredAt,
        }


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_SCORER_SYSTEM_PROMPT = (
    ANCIENT_CHINA_SYSTEM_PROMPT
    + "\n\n"
    "你是學術文獻驗證助手。給定一個查詢和若干文本片段，"
    "請對每個片段的相關性（relevance）和證據質量（evidence）各打分（0–10）。\n"
    "相關性：片段是否涉及查詢主題（0=完全無關，10=直接相關）。\n"
    "證據質量：片段是否包含可引用的具體史料（人名、律文條款、"
    "年份、事件、政令等）以支持回答查詢（0=無具體依據，10=充分明確）。\n"
    "請對輸入中的每一個編號片段都返回一個對應的 JSON 物件，"
    "不可省略或新增。只返回 JSON 陣列，不要包含任何 markdown 代碼塊。"
)

_SCORER_USER_TEMPLATE = """\
查詢：{query}

文本片段（{n} 個）：
{chunks_block}

請對每個片段返回：
[
  {{
    "id": <片段序號，從 1 開始>,
    "relevance": <0–10>,
    "evidence": <0–10>,
    "reasoning": "<一句話理由>"
  }},
  ...
]
"""


def _build_chunks_block(results: list[SearchResult]) -> str:
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        doc = r.spine.document_title or "?"
        ch = r.spine.chapter_title or ""
        excerpt = r.text[:_CHUNK_TEXT_LIMIT].replace("\n", " ").strip()
        lines.append(f"[{i}] 《{doc}》{ch}\n{excerpt}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------


def _parse_scores(raw: str, n: int) -> tuple[list[dict[str, Any]], list[int]]:
    """Extract the JSON array from the LLM response.

    Returns ``(items, missing_ids)`` — the parsed items keyed by their
    declared ``id`` and the list of 1..n ids that the model omitted.
    Raises ``ValueError`` if the response can't be parsed at all.
    """
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM response: {raw[:200]!r}")
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error: {exc} — raw: {text[start:end+1][:300]!r}") from exc

    if not isinstance(items, list):
        raise ValueError("Expected JSON array")

    parsed: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for item in items:
        try:
            idx = int(item.get("id", 0))
        except (TypeError, ValueError):
            continue
        if idx < 1 or idx > n or idx in seen_ids:
            continue
        seen_ids.add(idx)
        parsed.append(
            {
                "id": idx,
                "relevance": max(0.0, min(10.0, float(item.get("relevance", 0)))),
                "evidence": max(0.0, min(10.0, float(item.get("evidence", 0)))),
                "reasoning": str(item.get("reasoning", "")).strip()[:300],
            }
        )

    missing = sorted(set(range(1, n + 1)) - seen_ids)
    return parsed, missing


def score_evidence(
    query: str,
    results: list[SearchResult],
    *,
    model: str | None = None,
    max_chunks: int = 20,
    min_score: float = _DEFAULT_MIN_SCORE,
    client: OpenAI | None = None,
) -> tuple[list[EvidenceItem], EvidenceRunReport]:
    """Score SearchResults for evidence quality using deepseek-chat.

    Calls the LLM in batches of ``_MAX_CHUNKS_PER_CALL``. Results whose
    ``verification_score`` (average of relevance + evidence) is below
    ``min_score`` are filtered out; survivors are re-ranked by
    ``combined_score`` descending.

    Chunks the LLM omits from its response are logged on the run report
    (``chunks_missing_from_llm``) and treated as filtered rather than
    silently zero-scored — silent zeros previously tanked good results
    when the model renumbered or skipped an entry.

    Args:
        query: Original user query.
        results: List of :class:`SearchResult` from :func:`search`.
        model: Override chat model; defaults to ``CHAT_LLM_MODEL``.
        max_chunks: Cap on input chunks (trims tail by cosine score).
        min_score: Minimum ``verification_score`` (0–10) to keep a result.
        client: Re-use an existing Silra client.

    Returns:
        ``(items, report)``. ``items`` is sorted by ``combined_score``
        descending.
    """
    report = EvidenceRunReport(query=query)
    t_start = time.time()

    if not results:
        report.duration_seconds = time.time() - t_start
        return [], report

    llm_client = client or get_silra_client()
    chunks = results[:max_chunks]
    report.chunks_input = len(chunks)

    all_items: list[EvidenceItem] = []

    for batch_start in range(0, len(chunks), _MAX_CHUNKS_PER_CALL):
        batch = chunks[batch_start : batch_start + _MAX_CHUNKS_PER_CALL]
        user_msg = _SCORER_USER_TEMPLATE.format(
            query=query,
            n=len(batch),
            chunks_block=_build_chunks_block(batch),
        )
        messages = [
            {"role": "system", "content": _SCORER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            resp = chat_completion(
                messages,
                model=model,
                client=llm_client,
                temperature=0.0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content or ""
            report.llm_calls += 1
        except Exception as exc:
            log.error("score_evidence: LLM call failed for batch %d: %s", batch_start, exc)
            report.errors.append(f"batch {batch_start}: {exc}")
            report.chunks_failed += len(batch)
            continue

        try:
            parsed, missing = _parse_scores(raw, len(batch))
        except ValueError as exc:
            log.warning("score_evidence: parse failed for batch %d: %s", batch_start, exc)
            report.errors.append(f"parse batch {batch_start}: {exc}")
            report.chunks_failed += len(batch)
            continue

        if missing:
            log.warning(
                "score_evidence: batch %d missing %d/%d chunks from LLM response (ids=%s)",
                batch_start, len(missing), len(batch), missing,
            )
            report.chunks_missing_from_llm += len(missing)

        score_map: dict[int, dict[str, Any]] = {p["id"]: p for p in parsed}

        # Only emit items the LLM actually scored; omissions surface in
        # report.chunks_missing_from_llm rather than being silently zeroed.
        for idx, result in enumerate(batch, 1):
            scores = score_map.get(idx)
            if scores is None:
                continue
            all_items.append(
                EvidenceItem(
                    result=result,
                    relevance=scores["relevance"],
                    evidence=scores["evidence"],
                    reasoning=scores["reasoning"],
                )
            )

    report.chunks_scored = len(all_items)

    kept = [item for item in all_items if item.verification_score >= min_score]
    report.chunks_filtered = report.chunks_scored - len(kept)

    kept.sort(key=lambda x: x.combined_score, reverse=True)

    report.duration_seconds = round(time.time() - t_start, 3)
    log.info(
        "score_evidence: query=%r input=%d scored=%d kept=%d missing=%d calls=%d in %.2fs",
        query[:40],
        report.chunks_input,
        report.chunks_scored,
        len(kept),
        report.chunks_missing_from_llm,
        report.llm_calls,
        report.duration_seconds,
    )
    return kept, report


