"""``/api/keys`` — encrypted API-key vault CRUD.

The vault stores external API keys (Silra, OpenAI, …) so the user does
not have to keep their keys in plaintext ``.env`` files alongside the
shared server.

Encryption properties
---------------------
- Each key is encrypted with AES-256-GCM using a key derived from the
  user's password at login time. The server cannot decrypt at rest — the
  derived key only exists in the session entry (in process memory).
- Adding or rotating a key needs an authenticated session: the derived
  key in the session is used to encrypt the new plaintext.
- Reading a key (e.g. for the Silra client) also needs an authenticated
  session because we need the derived key to decrypt.
- Deleting a key only needs a session; no decryption involved.

Endpoint summary
----------------
- ``GET    /api/keys``            list keys with metadata only (no plaintext)
- ``POST   /api/keys``            create a new key (provides plaintext)
- ``PUT    /api/keys/{id}``       rotate the stored plaintext or rename
- ``DELETE /api/keys/{id}``       remove the row
- ``GET    /api/keys/{id}/reveal`` returns plaintext ONCE — for one-time
                                  display in the UI right after creation
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.backend.auth import crypto, session
from apps.backend.auth.db import get_session
from apps.backend.auth.deps import require_user
from apps.backend.auth.models import ApiKey, AuditLog

router = APIRouter()


class ApiKeyCreate(BaseModel):
    label: str = Field(..., min_length=1, max_length=120)
    provider: str = Field("silra", min_length=1, max_length=64)
    plaintext: str = Field(..., min_length=4, max_length=4096)


class ApiKeyUpdate(BaseModel):
    label: Optional[str] = Field(None, min_length=1, max_length=120)
    plaintext: Optional[str] = Field(None, min_length=4, max_length=4096)


def _public(row: ApiKey) -> dict[str, Any]:
    return {
        "id": row.id,
        "label": row.label,
        "provider": row.provider,
        "last4": row.last4,
        "fingerprint": row.fingerprint[:12],
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
    }


async def _audit(
    db: AsyncSession,
    *,
    user_id: int,
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


@router.get("")
async def list_keys(
    user: session.SessionEntry = Depends(require_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = (await db.execute(
        select(ApiKey).where(ApiKey.user_id == user.user_id).order_by(ApiKey.created_at)
    )).scalars().all()
    return {"keys": [_public(r) for r in rows], "total": len(rows)}


@router.post("")
async def create_key(
    request: Request,
    payload: ApiKeyCreate,
    user: session.SessionEntry = Depends(require_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    # Encrypt with the session-derived key. No plaintext ever touches the DB.
    nonce, ct = crypto.encrypt_secret(user.derived_key, payload.plaintext)

    fp = crypto.fingerprint(payload.plaintext)
    existing = (await db.execute(
        select(ApiKey).where(
            ApiKey.user_id == user.user_id,
            ApiKey.fingerprint == fp,
        )
    )).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"This key is already stored as {existing.label!r}.",
        )

    row = ApiKey(
        user_id=user.user_id,
        label=payload.label,
        provider=payload.provider,
        nonce=nonce,
        ciphertext=ct,
        last4=crypto.last4(payload.plaintext),
        fingerprint=fp,
    )
    db.add(row)
    await _audit(db, user_id=user.user_id, event="key.created", request=request,
                 details={"label": payload.label, "provider": payload.provider})
    await db.commit()
    await db.refresh(row)
    return _public(row)


@router.put("/{key_id}")
async def update_key(
    key_id: int,
    request: Request,
    payload: ApiKeyUpdate,
    user: session.SessionEntry = Depends(require_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    row = (await db.execute(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.user_id)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="API key not found")

    changed: list[str] = []
    if payload.label is not None and payload.label != row.label:
        row.label = payload.label
        changed.append("label")

    if payload.plaintext is not None:
        nonce, ct = crypto.encrypt_secret(user.derived_key, payload.plaintext)
        row.nonce = nonce
        row.ciphertext = ct
        row.last4 = crypto.last4(payload.plaintext)
        row.fingerprint = crypto.fingerprint(payload.plaintext)
        changed.append("plaintext")

    if not changed:
        return _public(row)

    row.updated_at = datetime.now(timezone.utc)
    await _audit(db, user_id=user.user_id, event="key.updated", request=request,
                 details={"id": row.id, "changed": changed})
    await db.commit()
    await db.refresh(row)
    return _public(row)


@router.delete("/{key_id}")
async def delete_key(
    key_id: int,
    request: Request,
    user: session.SessionEntry = Depends(require_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    row = (await db.execute(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.user_id)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="API key not found")
    label = row.label
    await db.delete(row)
    await _audit(db, user_id=user.user_id, event="key.deleted", request=request,
                 details={"id": key_id, "label": label})
    await db.commit()
    return {"ok": True, "id": key_id, "label": label}


@router.get("/{key_id}/reveal")
async def reveal_key(
    key_id: int,
    request: Request,
    user: session.SessionEntry = Depends(require_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Return the plaintext for one-time display in the UI.

    This is the only endpoint that returns a plaintext key. We audit it
    so a leaked session can be detected via the audit log.
    """
    row = (await db.execute(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.user_id)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="API key not found")
    try:
        plaintext = crypto.decrypt_secret(user.derived_key, row.nonce, row.ciphertext)
    except Exception:
        # InvalidTag — derived key mismatch (password rotated since this
        # key was stored). Don't leak that fact to the user; just 404.
        raise HTTPException(
            status_code=410,
            detail="This key cannot be decrypted with the current password. "
                   "It was probably stored before a password rotation.",
        )
    await _audit(db, user_id=user.user_id, event="key.revealed", request=request,
                 details={"id": key_id, "label": row.label})
    await db.commit()
    return {"id": row.id, "label": row.label, "plaintext": plaintext}
