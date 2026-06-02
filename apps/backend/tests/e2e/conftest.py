"""E2E conftest: skip all tests in this directory when the server is not reachable.

This allows `pytest` (the full suite) to run cleanly in CI without a live server.
Run E2E tests explicitly:  uv run pytest apps/backend/tests/e2e/ --browser chromium
"""
from __future__ import annotations

import socket

import pytest

_SERVER_HOST = "localhost"
_SERVER_PORT = 8013


def _server_is_up() -> bool:
    try:
        with socket.create_connection((_SERVER_HOST, _SERVER_PORT), timeout=2):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    if _server_is_up():
        return
    skip = pytest.mark.skip(
        reason=f"E2E server not reachable at {_SERVER_HOST}:{_SERVER_PORT} — "
               "start with: uv run python scripts/run_server.py"
    )
    for item in items:
        if "e2e" in str(item.fspath):
            item.add_marker(skip)
