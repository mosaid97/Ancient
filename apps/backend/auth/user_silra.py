"""Resolve the logged-in user's Silra key on each authenticated request.

The session entry holds the derived AES key; the DB holds the ciphertext.
This helper decrypts on demand and returns a configured ``openai.OpenAI``
client. The decrypted plaintext lives only on the local stack frame of
the caller — it is not cached and never returned to the client.

Two helpers:

- :func:`get_user_silra_client` — FastAPI dependency yielding the client.
- :func:`build_user_silra_client` — plain coroutine for use outside the
  request pipeline (e.g. background tasks triggered from an endpoint).

Both fall back to the env-based client when no user session is present
*and* the request is unauthenticated, so pipeline scripts that hit the
HTTP layer (rare) still work. For end-user endpoints that *require*
authentication, depend on :func:`apps.backend.auth.deps.require_user`
first so the 401 path is taken.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException
from openai import OpenAI
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from apps.backend.auth import crypto, session
from apps.backend.auth.db import get_session
from apps.backend.auth.deps import optional_user
from apps.backend.auth.models import ApiKey
from apps.backend.llm.silra import get_silra_client


async def _decrypt_user_silra_key(
    user: session.SessionEntry,
    db: AsyncSession,
    *,
    provider: str = "silra",
) -> Optional[str]:
    """Return plaintext Silra key for the user's most recently-used row,
    or None if the user has no stored Silra key (caller can choose to
    fall back to env)."""
    row = (await db.execute(
        select(ApiKey)
        .where(ApiKey.user_id == user.user_id, ApiKey.provider == provider)
        .order_by(ApiKey.last_used_at.desc().nulls_last(), ApiKey.updated_at.desc())
        .limit(1)
    )).scalar_one_or_none()
    if row is None:
        return None
    try:
        plaintext = crypto.decrypt_secret(user.derived_key, row.nonce, row.ciphertext)
    except Exception:
        # Stored under an old password — the session's derived key can't
        # unwrap it. Surface as no-key so the caller falls back / errors
        # cleanly.
        return None
    await db.execute(
        update(ApiKey)
        .where(ApiKey.id == row.id)
        .values(last_used_at=datetime.now(timezone.utc))
    )
    await db.commit()
    return plaintext


async def get_user_silra_client(
    user: session.SessionEntry | None = Depends(optional_user),
    db: AsyncSession = Depends(get_session),
) -> OpenAI:
    """FastAPI dependency: yield an OpenAI client using the user's key.

    If the request is unauthenticated, fall back to the env-based
    client so pipeline scripts (translation / embedding) keep working.
    Authenticated requests with no stored Silra key get a 412 so the
    UI can prompt the user to add a key.
    """
    if user is None:
        # Unauthenticated; assume offline pipeline or boot probe.
        return get_silra_client()

    plaintext = await _decrypt_user_silra_key(user, db)
    if plaintext is None:
        raise HTTPException(
            status_code=412,
            detail="No Silra API key is stored. Add one in the API Keys panel.",
        )
    return get_silra_client(api_key=plaintext)
