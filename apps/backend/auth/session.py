"""In-process session store mapping ``session_id`` → derived key + user.

Design
------
- Session IDs are 256-bit URL-safe random tokens stored in an HttpOnly,
  SameSite=Strict, Secure cookie.
- The store is a process-local dict guarded by an ``RLock``. Restart kills
  every session (intentional — the derived AES key never persists to disk
  or to Redis, which would reduce the zero-knowledge guarantee).
- A background sweeper task should be wired into the FastAPI lifespan;
  for the single-tenant case we accept that idle sessions hang around
  until the next operation triggers the inline cleanup.

The derived AES-GCM key for the user's API-key vault is held INSIDE the
session entry. Code that needs to decrypt a stored API key reads the key
out of the session via :func:`get_session_key`. The key bytes are never
serialised or written to logs.
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

SESSION_COOKIE = "ancient_session"
DEFAULT_TTL_SECONDS = 8 * 60 * 60   # 8 hours of inactivity → re-login


@dataclass
class SessionEntry:
    user_id: int
    username: str
    # 32-byte AES-256 key derived from the user's password at login.
    # NEVER persist this — restart loses every session deliberately.
    derived_key: bytes
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    ip: Optional[str] = None
    user_agent: Optional[str] = None


_lock = threading.RLock()
_store: dict[str, SessionEntry] = {}


def new_session_id() -> str:
    """Cryptographically random 256-bit ID, URL-safe base64."""
    return secrets.token_urlsafe(32)


def create(
    *,
    user_id: int,
    username: str,
    derived_key: bytes,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> str:
    """Insert a new session and return its session_id."""
    if len(derived_key) != 32:
        raise ValueError("derived_key must be 32 bytes")
    sid = new_session_id()
    with _lock:
        _store[sid] = SessionEntry(
            user_id=user_id,
            username=username,
            derived_key=derived_key,
            ip=ip,
            user_agent=user_agent,
        )
    return sid


def get(session_id: str, *, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> SessionEntry | None:
    """Return the session if alive, else None (also evicts on miss)."""
    if not session_id:
        return None
    with _lock:
        entry = _store.get(session_id)
        if entry is None:
            return None
        if time.time() - entry.last_seen_at > ttl_seconds:
            _store.pop(session_id, None)
            return None
        entry.last_seen_at = time.time()
        return entry


def get_session_key(session_id: str) -> bytes | None:
    """Convenience for callers that just need the AES key bytes."""
    entry = get(session_id)
    return entry.derived_key if entry else None


def destroy(session_id: str) -> None:
    """Drop a session and overwrite its derived key in memory."""
    if not session_id:
        return
    with _lock:
        entry = _store.pop(session_id, None)
        if entry is not None:
            # Best-effort scrub. Python doesn't guarantee bytes mutation,
            # but rebinding makes the original object eligible for GC and
            # leaves nothing useful in the entry.
            entry.derived_key = b"\x00" * 32


def destroy_all_for_user(user_id: int) -> int:
    """Drop every session belonging to ``user_id`` (e.g. on password change)."""
    removed = 0
    with _lock:
        for sid in [s for s, e in _store.items() if e.user_id == user_id]:
            _store.pop(sid, None)
            removed += 1
    return removed


def active_count() -> int:
    with _lock:
        return len(_store)


def reset_for_tests() -> None:
    with _lock:
        _store.clear()
