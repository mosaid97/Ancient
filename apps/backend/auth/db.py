"""SQLAlchemy 2.0 async engine + ``get_session`` FastAPI dependency.

The engine is created lazily on first request so that test code can monkey-
patch ``DATABASE_URL`` before connecting.

The ``DATABASE_URL`` env var should use the ``postgresql+asyncpg://`` scheme
(set by docker-compose / .env.example). For local development without
Postgres the test suite swaps in an in-memory SQLite engine.
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine() -> AsyncEngine:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. See .env.example for the expected "
            "postgresql+asyncpg://… connection string."
        )
    # SQLAlchemy doesn't auto-translate sync psycopg URLs to async asyncpg,
    # so we accept either spelling and patch it.
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return create_async_engine(url, pool_pre_ping=True, future=True)


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session, rolling back on error."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def reset_engine_for_tests() -> None:
    """Test hook — drop the cached engine + session factory."""
    global _engine, _session_factory
    _engine = None
    _session_factory = None
