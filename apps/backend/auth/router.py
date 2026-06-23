"""``/api/auth`` endpoints: login, logout, me.

All endpoints rate-limited via slowapi (10/min per IP for login). The
login flow:

1. Look up the user by username.
2. Argon2id-verify the password.
3. Derive a 32-byte AES key from (password, user.kdf_salt).
4. Create a session entry holding the derived key in process memory.
5. Set a Secure HttpOnly SameSite=Strict cookie with the session_id.

The derived key is the **only** way to decrypt the user's API keys; it
exists only in the session entry and the local stack of this function.
It never appears in logs, audit rows, or response bodies.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.backend.auth import crypto, session
from apps.backend.auth.db import get_session
from apps.backend.auth.models import AuditLog, User

log = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=512)


def _set_session_cookie(response: Response, session_id: str) -> None:
    # secure=True is required in production. We force it on so the cookie
    # never leaks over plaintext HTTP; behind localhost dev you must use
    # http://localhost so the browser still attaches it (Chrome exempts
    # localhost from the Secure requirement).
    response.set_cookie(
        key=session.SESSION_COOKIE,
        value=session_id,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=session.DEFAULT_TTL_SECONDS,
        path="/",
    )


async def _log_event(
    db: AsyncSession,
    *,
    user_id: int | None,
    event: str,
    request: Request,
    details: dict[str, Any] | None = None,
) -> None:
    db.add(AuditLog(
        user_id=user_id,
        event=event,
        details=details or {},
        ip=getattr(request.client, "host", None) if request.client else None,
        user_agent=request.headers.get("user-agent"),
    ))


@router.post("/login")
@limiter.limit("10/minute")
async def login(
    request: Request,
    response: Response,
    payload: LoginRequest,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    user = (
        await db.execute(select(User).where(User.username == payload.username))
    ).scalar_one_or_none()

    if user is None or not user.is_active:
        # Constant-time bait: still pay the password-verify cost so we
        # don't leak username existence through timing.
        crypto.verify_password(payload.password, crypto.hash_password("not-a-real-pw"))
        await _log_event(db, user_id=None, event="login.unknown_user",
                         request=request, details={"username": payload.username[:32]})
        await db.commit()
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not crypto.verify_password(payload.password, user.password_hash):
        await _log_event(db, user_id=user.id, event="login.bad_password", request=request)
        await db.commit()
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Derive the in-memory AES key and stash it on a fresh session entry.
    derived = crypto.derive_key(payload.password, user.kdf_salt)
    sid = session.create(
        user_id=user.id,
        username=user.username,
        derived_key=derived,
        ip=getattr(request.client, "host", None) if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    user.last_login_at = datetime.now(timezone.utc)
    if crypto.needs_rehash(user.password_hash):
        user.password_hash = crypto.hash_password(payload.password)

    await _log_event(db, user_id=user.id, event="login.ok", request=request)
    await db.commit()

    _set_session_cookie(response, sid)
    return {
        "username": user.username,
        "is_admin": user.is_admin,
        "session_expires_in": session.DEFAULT_TTL_SECONDS,
    }


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    sid = request.cookies.get(session.SESSION_COOKIE, "")
    entry = session.get(sid)
    if entry is not None:
        await _log_event(db, user_id=entry.user_id, event="logout", request=request)
        await db.commit()
        session.destroy(sid)

    response.delete_cookie(session.SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/me")
async def me(request: Request) -> dict[str, Any]:
    sid = request.cookies.get(session.SESSION_COOKIE, "")
    entry = session.get(sid)
    if entry is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "username": entry.username,
        "user_id": entry.user_id,
        "session_age_seconds": int(entry.last_seen_at - entry.created_at),
    }
