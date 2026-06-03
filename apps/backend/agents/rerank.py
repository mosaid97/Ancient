"""C2: BGE-reranker-v2-gemma cross-encoder reranker (plan §0.6 C2).

Reranks top-100 hybrid retrieval candidates down to top-20 using
FlagEmbedding's FlagReranker (BGE-reranker-v2-gemma). The model is
lazy-loaded and cached on first use.

Public API
----------
rerank(query, candidates, *, top_k, threshold) -> list[RerankResult]
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

_RERANKER_MODEL = "BAAI/bge-reranker-v2-gemma"
_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_TOP_K = 20
_DEFAULT_THRESHOLD = 0.0

_reranker_instance: Any = None
_reranker_type: str = "none"


class _SentenceTransformerReranker:
    """Bi-encoder cosine reranker using a cached SentenceTransformer model."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        self._model = SentenceTransformer(model_name)

    def compute_score(self, pairs: list[tuple[str, str]], normalize: bool = True) -> list[float]:
        queries = [q for q, _ in pairs]
        docs = [d for _, d in pairs]
        import numpy as np
        q_emb = self._model.encode(queries, normalize_embeddings=True)
        d_emb = self._model.encode(docs, normalize_embeddings=True)
        scores = (q_emb * d_emb).sum(axis=1).tolist()
        if normalize:
            mn, mx = min(scores), max(scores)
            rng = mx - mn or 1.0
            scores = [(s - mn) / rng for s in scores]
        return scores


def _get_reranker():
    global _reranker_instance, _reranker_type
    if _reranker_instance is None:
        # Try primary BGE model first.
        # bge-reranker-v2-gemma is an LLM-based (decoder-only) reranker → requires
        # FlagLLMReranker, NOT FlagReranker (which is encoder-only and fails with
        # GemmaTokenizer).  Run offline so startup doesn't attempt hub.hf.co.
        try:
            import os
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from FlagEmbedding import FlagLLMReranker  # type: ignore[import]
            _reranker_instance = FlagLLMReranker(_RERANKER_MODEL, use_fp16=True)
            _reranker_type = "bge"
            log.info("Loaded reranker: %s", _RERANKER_MODEL)
        except Exception as exc:
            log.warning("BGE reranker unavailable (%s), falling back to %s", exc, _FALLBACK_MODEL)
            try:
                _reranker_instance = _SentenceTransformerReranker(_FALLBACK_MODEL)
                _reranker_type = "minilm"
                log.info("Loaded fallback reranker: %s", _FALLBACK_MODEL)
            except Exception as exc2:
                log.error("All rerankers failed: %s", exc2)
                raise
    return _reranker_instance


@dataclass
class RerankResult:
    """Single reranked candidate."""

    chunk_id: str
    retrieval_score: float   # original RRF / vector score
    rerank_score: float      # cross-encoder score (normalized 0–1)
    rank: int                # 1-based rank after reranking

    @property
    def combined_score(self) -> float:
        """0.7 × rerank + 0.3 × retrieval (retrieval capped at 1.0)."""
        return 0.7 * self.rerank_score + 0.3 * min(self.retrieval_score, 1.0)


def rerank(
    query: str,
    candidates: list[tuple[str, str, float]],
    *,
    top_k: int = _DEFAULT_TOP_K,
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[RerankResult]:
    """Rerank candidate chunks with BGE-reranker-v2-gemma.

    Args:
        query: The user query string.
        candidates: List of (chunk_id, chunk_text, retrieval_score) tuples.
            Typically the top-100 from RRF fusion.
        top_k: Number of results to return after reranking.
        threshold: Minimum rerank score to include (default 0 = keep all).

    Returns:
        List of :class:`RerankResult` sorted by rerank_score descending.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, text) for _, text, _ in candidates]

    try:
        # Mini-batch to avoid MPS/CUDA OOM on long candidate lists (Apple MPS
        # rejects matmul kernels above ~2^28 elements; 16 pairs is safe).
        batch_size = int(os.getenv("RERANK_BATCH_SIZE", "16"))
        scores: list[float] = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            raw = reranker.compute_score(chunk, normalize=True)
            if isinstance(raw, float):
                raw = [raw]
            scores.extend(float(s) for s in raw)
    except Exception as exc:
        log.warning("Reranker scoring failed, returning retrieval order: %s", exc)
        return [
            RerankResult(
                chunk_id=cid,
                retrieval_score=ret_score,
                rerank_score=ret_score,
                rank=i + 1,
            )
            for i, (cid, _, ret_score) in enumerate(candidates[:top_k])
        ]

    # Combine and sort
    combined = [
        (cid, ret_score, rr_score)
        for (cid, _, ret_score), rr_score in zip(candidates, scores)
        if rr_score >= threshold
    ]
    combined.sort(key=lambda x: x[2], reverse=True)

    return [
        RerankResult(
            chunk_id=cid,
            retrieval_score=ret_score,
            rerank_score=rr_score,
            rank=i + 1,
        )
        for i, (cid, ret_score, rr_score) in enumerate(combined[:top_k])
    ]
