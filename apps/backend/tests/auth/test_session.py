"""Unit tests for the in-process session store."""
from __future__ import annotations

import secrets
import time

import pytest

from apps.backend.auth import session


@pytest.fixture(autouse=True)
def _reset():
    session.reset_for_tests()
    yield
    session.reset_for_tests()


def _key() -> bytes:
    return secrets.token_bytes(32)


def test_create_get_roundtrip():
    sid = session.create(user_id=1, username="admin", derived_key=_key())
    entry = session.get(sid)
    assert entry is not None
    assert entry.user_id == 1
    assert entry.username == "admin"


def test_missing_session_returns_none():
    assert session.get("never-created") is None
    assert session.get("") is None


def test_destroy_drops_session_and_scrubs_key():
    k = _key()
    sid = session.create(user_id=1, username="admin", derived_key=k)
    entry = session.get(sid)
    assert entry is not None
    session.destroy(sid)
    assert session.get(sid) is None


def test_get_session_key_helper():
    k = _key()
    sid = session.create(user_id=1, username="admin", derived_key=k)
    assert session.get_session_key(sid) == k
    session.destroy(sid)
    assert session.get_session_key(sid) is None


def test_idle_eviction():
    sid = session.create(user_id=1, username="admin", derived_key=_key())
    # Backdate last_seen so the entry is past the very short TTL we pass.
    entry = session.get(sid, ttl_seconds=10_000)
    assert entry is not None
    entry.last_seen_at = time.time() - 1000
    assert session.get(sid, ttl_seconds=10) is None


def test_destroy_all_for_user_clears_only_that_user():
    s1 = session.create(user_id=1, username="alice", derived_key=_key())
    s2 = session.create(user_id=2, username="bob", derived_key=_key())
    s3 = session.create(user_id=1, username="alice", derived_key=_key())
    n = session.destroy_all_for_user(1)
    assert n == 2
    assert session.get(s1) is None
    assert session.get(s3) is None
    assert session.get(s2) is not None


def test_create_rejects_short_key():
    with pytest.raises(ValueError):
        session.create(user_id=1, username="x", derived_key=b"\x00" * 16)


def test_active_count():
    assert session.active_count() == 0
    session.create(user_id=1, username="a", derived_key=_key())
    session.create(user_id=2, username="b", derived_key=_key())
    assert session.active_count() == 2
