"""FastAPI dependencies that any router can use to require a session.

Usage::

    @router.get("/...")
    async def endpoint(user: SessionEntry = Depends(require_user)) -> dict:
        ...

The dependency reads the session cookie, looks up the in-memory store,
and 401s when the session is missing or expired. It updates ``last_seen_at``
so each authenticated request extends the idle TTL.
"""
from __future__ import annotations

from fastapi import HTTPException, Request

from apps.backend.auth import session


def require_user(request: Request) -> session.SessionEntry:
    sid = request.cookies.get(session.SESSION_COOKIE, "")
    entry = session.get(sid)
    if entry is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return entry


def optional_user(request: Request) -> session.SessionEntry | None:
    """Same lookup but returns None instead of 401 — for endpoints that
    behave differently when authed (e.g. pipeline scripts vs. UI calls)."""
    sid = request.cookies.get(session.SESSION_COOKIE, "")
    return session.get(sid)
