"""In-process verifier-outcome telemetry.

Search throughput is high and writing one Neo4j node per successful
verification would dominate query latency, so we keep aggregate counts
and a small ring-buffer of recent failures in memory. Counters are
process-local and reset on restart — that is the right trade-off for a
research-grade dashboard whose job is to show *recent* trust signals,
not historical bookkeeping. For long-term trends use the eval harness
(``eval/harness.py``) and persisted bench runs.

Counter buckets mirror :data:`VerifierResult.outcome`:

- ``ok``                   — span confirmed in canonical/raw source
- ``translation_match``    — span only matched LLM vernacular
- ``not_applicable``       — no verbatim span in the query
- ``insufficient_evidence``— failed gate (dropped from ribbon)

Thread-safe: a single ``Lock`` guards both the counter dict and the
ring buffer; the critical section is a handful of dict/deque ops so
contention is negligible.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Any

_OUTCOMES = (
    "ok",
    "translation_match",
    "not_applicable",
    "insufficient_evidence",
)
_RECENT_CAP = 100

_lock = threading.Lock()
_counts: Counter[str] = Counter({o: 0 for o in _OUTCOMES})
_recent: deque[dict[str, Any]] = deque(maxlen=_RECENT_CAP)
_started_at = time.time()


def record(outcome: str, *, chunk_id: str | None = None,
           failure_mode: str | None = None, span: str | None = None,
           tier: str | None = None) -> None:
    """Tally one verifier outcome. Safe to call from any worker thread."""
    bucket = outcome if outcome in _OUTCOMES else "insufficient_evidence"
    with _lock:
        _counts[bucket] += 1
        if bucket == "insufficient_evidence" and failure_mode:
            _recent.append({
                "ts": time.time(),
                "chunk_id": chunk_id,
                "failure_mode": failure_mode,
                "span": (span or "")[:80],
                "tier": tier,
            })


def snapshot() -> dict[str, Any]:
    """Return current counters + verified rate + recent failures.

    The ``verified_rate`` is ``ok / (ok + insufficient_evidence)`` — i.e.
    of the times the verifier had something to check, how often did it
    pass. ``not_applicable`` and ``translation_match`` are excluded
    because they describe queries the gate doesn't speak to.
    """
    with _lock:
        counts = dict(_counts)
        recent = list(_recent)

    gateable = counts["ok"] + counts["insufficient_evidence"]
    verified_rate = (counts["ok"] / gateable) if gateable else None

    total = sum(counts.values())
    return {
        "counts": counts,
        "total": total,
        "verified_rate": verified_rate,
        "uptime_seconds": round(time.time() - _started_at, 1),
        "recent_failures": list(reversed(recent)),
    }


def reset() -> None:
    """Test hook — clear all counters."""
    global _started_at
    with _lock:
        for k in _OUTCOMES:
            _counts[k] = 0
        _recent.clear()
        _started_at = time.time()
