"""Verifier pipeline — Phase 7b LLM-based citation evidence scoring.

For a given user query and a list of :class:`SearchResult` objects from
Phase 7 :func:`search`, this module sends the chunk texts to ``deepseek-chat``
and asks it to score each chunk on two axes:

- **relevance** (0–10): Does this chunk address the query topic at all?
- **evidence** (0–10): Does this chunk contain specific, citable evidence
  that would directly support answering the query (date, law clause, name,
  event, policy description, etc.)?

A combined ``verification_score`` = 0.5 × relevance + 0.5 × evidence is
computed.  Chunks below ``min_score`` are filtered out.  Results are re-ranked
by ``verification_score × cosine_score`` (reciprocal rank fusion is deferred
to Phase 9 HITL).

Public API
----------
verify_chunks(query, results, *, model, max_chunks, min_score, client)
    -> list[VerificationItem]

VerificationItem          — scored result dataclass
VerifierRunReport         — summary of a verifier run

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
_CHUNK_TEXT_LIMIT = 400     # chars shown to the verifier per chunk


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VerificationItem:
    """A search result augmented with verifier scores."""

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
class VerifierRunReport:
    """Summary of a verifier run."""

    query: str
    chunks_input: int = 0
    chunks_verified: int = 0
    chunks_filtered: int = 0
    chunks_failed: int = 0
    llm_calls: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    verifiedAt: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "chunks_input": self.chunks_input,
            "chunks_verified": self.chunks_verified,
            "chunks_filtered": self.chunks_filtered,
            "chunks_failed": self.chunks_failed,
            "llm_calls": self.llm_calls,
            "duration_seconds": round(self.duration_seconds, 3),
            "errors": self.errors[:20],
            "verifiedAt": self.verifiedAt,
        }


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_VERIFIER_SYSTEM_PROMPT = (
    ANCIENT_CHINA_SYSTEM_PROMPT
    + "\n\n"
    "你是學術文獻驗證助手。給定一個查詢和若干文本片段，"
    "請對每個片段的相關性（relevance）和證據質量（evidence）各打分（0–10）。\n"
    "相關性：片段是否涉及查詢主題（0=完全無關，10=直接相關）。\n"
    "證據質量：片段是否包含可引用的具體史料（人名、律文條款、"
    "年份、事件、政令等）以支持回答查詢（0=無具體依據，10=充分明確）。\n"
    "只返回 JSON 陣列，不要包含任何 markdown 代碼塊。"
)

_VERIFIER_USER_TEMPLATE = """\
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
# Core verify function
# ---------------------------------------------------------------------------


def _parse_scores(raw: str, n: int) -> list[dict[str, Any]]:
    """Extract the JSON array from the LLM response.

    Handles cases where the model wraps the JSON in markdown fences.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Find first '[' … ']' block
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

    # Normalise
    parsed: list[dict[str, Any]] = []
    for item in items:
        parsed.append(
            {
                "id": int(item.get("id", 0)),
                "relevance": max(0.0, min(10.0, float(item.get("relevance", 0)))),
                "evidence": max(0.0, min(10.0, float(item.get("evidence", 0)))),
                "reasoning": str(item.get("reasoning", "")).strip()[:300],
            }
        )
    return parsed


def verify_chunks(
    query: str,
    results: list[SearchResult],
    *,
    model: str | None = None,
    max_chunks: int = 20,
    min_score: float = _DEFAULT_MIN_SCORE,
    client: OpenAI | None = None,
) -> tuple[list[VerificationItem], VerifierRunReport]:
    """Score SearchResults for evidence quality using deepseek-chat.

    Calls the LLM in batches of ``_MAX_CHUNKS_PER_CALL``.  Results below
    ``min_score`` (average of relevance + evidence) are filtered out.  The
    surviving items are re-ranked by ``combined_score``.

    Args:
        query: Original user query.
        results: List of :class:`SearchResult` from :func:`search`.
        model: Override chat model; defaults to ``CHAT_LLM_MODEL``.
        max_chunks: Cap on input chunks (trims tail by cosine score).
        min_score: Minimum ``verification_score`` (0–10) to keep a result.
        client: Re-use an existing Silra client.

    Returns:
        Tuple of (verified_items, report).  ``verified_items`` is sorted by
        ``combined_score`` descending.
    """
    report = VerifierRunReport(query=query)
    t_start = time.time()

    if not results:
        report.duration_seconds = time.time() - t_start
        return [], report

    llm_client = client or get_silra_client()
    chunks = results[:max_chunks]
    report.chunks_input = len(chunks)

    all_items: list[VerificationItem] = []

    # Process in batches of _MAX_CHUNKS_PER_CALL
    for batch_start in range(0, len(chunks), _MAX_CHUNKS_PER_CALL):
        batch = chunks[batch_start : batch_start + _MAX_CHUNKS_PER_CALL]
        user_msg = _VERIFIER_USER_TEMPLATE.format(
            query=query,
            n=len(batch),
            chunks_block=_build_chunks_block(batch),
        )
        messages = [
            {"role": "system", "content": _VERIFIER_SYSTEM_PROMPT},
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
            log.error("verify_chunks: LLM call failed for batch %d: %s", batch_start, exc)
            report.errors.append(f"batch {batch_start}: {exc}")
            report.chunks_failed += len(batch)
            continue

        try:
            parsed = _parse_scores(raw, len(batch))
        except ValueError as exc:
            log.warning("verify_chunks: parse failed for batch %d: %s", batch_start, exc)
            report.errors.append(f"parse batch {batch_start}: {exc}")
            report.chunks_failed += len(batch)
            continue

        # Map scores back to results by 1-based id within the batch
        score_map: dict[int, dict[str, Any]] = {p["id"]: p for p in parsed}

        for idx, result in enumerate(batch, 1):
            scores = score_map.get(idx)
            if scores is None:
                # LLM missed this chunk; give it zeros
                scores = {"relevance": 0.0, "evidence": 0.0, "reasoning": ""}
            item = VerificationItem(
                result=result,
                relevance=scores["relevance"],
                evidence=scores["evidence"],
                reasoning=scores["reasoning"],
            )
            all_items.append(item)

    report.chunks_verified = len(all_items)

    # Filter by min_score
    kept = [item for item in all_items if item.verification_score >= min_score]
    report.chunks_filtered = report.chunks_verified - len(kept)

    # Sort by combined_score descending
    kept.sort(key=lambda x: x.combined_score, reverse=True)

    report.duration_seconds = round(time.time() - t_start, 3)
    log.info(
        "verify_chunks: query=%r input=%d verified=%d kept=%d calls=%d in %.2fs",
        query[:40],
        report.chunks_input,
        report.chunks_verified,
        len(kept),
        report.llm_calls,
        report.duration_seconds,
    )
    return kept, report
