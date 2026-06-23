"""SQLAlchemy ORM models for the single-user (extensible) auth schema.

Tables
------
- ``users``     The (currently single) admin account. Argon2id password hash.
- ``api_keys``  Per-user encrypted external API keys (Silra / OpenAI / …).
                The ciphertext is AES-GCM encrypted with a key derived from
                the user's password — server cannot decrypt at rest.
- ``audit_log`` Append-only audit trail of auth + key-vault events.

Forward-compatibility for multi-user: ``api_keys`` and ``audit_log`` are
keyed on ``user_id``, so adding a second user requires only enabling a
signup/invite endpoint — no schema migration.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)

    # Argon2id password hash — never the raw password. The hash also encodes
    # the salt + parameters so verification is self-contained.
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)

    # Per-user salt for the key-derivation step. We re-derive an AES key on
    # every login (so the server never stores it at rest); the salt + the
    # password produce the same key each session.
    kdf_salt: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)

    is_admin: Mapped[bool] = mapped_column(default=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"<User id={self.id} username={self.username!r}>"


class ApiKey(Base):
    """An external-provider API key, stored ciphertext-only.

    The plaintext key is encrypted client-of-the-DB (in this process) with
    AES-GCM using a key derived from the owner's password. The DB row holds
    the nonce + ciphertext + a SHA-256 fingerprint of the plaintext (used
    only to detect duplicates and to display the last-4 chars in the UI —
    never the plaintext itself).
    """

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Logical identifier the user picks ("Silra prod", "OpenAI personal").
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    # Which provider/integration this key talks to.
    provider: Mapped[str] = mapped_column(String(64), nullable=False, default="silra")

    # AES-GCM components. Nonce is 12 bytes; ciphertext includes the 16-byte
    # auth tag appended (cryptography.hazmat AESGCM output format).
    nonce: Mapped[bytes] = mapped_column(LargeBinary(12), nullable=False)
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Last-4 chars of the plaintext key, plus a fingerprint for dedupe.
    # The fingerprint is NOT a password equivalent; it's a one-way digest.
    last4: Mapped[str] = mapped_column(String(8), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    user: Mapped[User] = relationship(back_populates="api_keys")

    __table_args__ = (
        UniqueConstraint("user_id", "label", name="uq_api_keys_user_label"),
        Index("ix_api_keys_user_provider", "user_id", "provider"),
    )


class AuditLog(Base):
    """Append-only audit log. Used for both auth and key-vault events."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # e.g. "login.ok", "login.bad_password", "key.created", "key.deleted".
    event: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Free-form JSON details. Never include raw key material here.
    details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    ip: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
