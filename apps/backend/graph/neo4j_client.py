"""Neo4j driver factory + health probe per AGENTS.md §4.

All callers should obtain a driver via :func:`get_driver` and use either the
returned driver's own ``session()`` or :func:`driver_session` (an alias kept
for symmetry with other client modules).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from neo4j import Driver, GraphDatabase, Session

logger = logging.getLogger(__name__)


def get_driver(
    uri: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> Driver:
    """Build a Neo4j driver from environment variables.

    Args:
        uri: Override; defaults to ``NEO4J_URI`` env var
            (``bolt://localhost:7687`` outside Docker, ``bolt://neo4j:7687``
            inside).
        username: Override; defaults to ``NEO4J_USERNAME``.
        password: Override; defaults to ``NEO4J_PASSWORD``.

    Returns:
        A connected :class:`neo4j.Driver`. Caller is responsible for ``close()``
        — prefer ``with`` blocks per AGENTS.md §4.

    Raises:
        RuntimeError: If any of the three env vars is missing.
    """
    uri = uri or os.getenv("NEO4J_URI")
    username = username or os.getenv("NEO4J_USERNAME")
    password = password or os.getenv("NEO4J_PASSWORD")
    missing = [
        n
        for n, v in [
            ("NEO4J_URI", uri),
            ("NEO4J_USERNAME", username),
            ("NEO4J_PASSWORD", password),
        ]
        if not v
    ]
    if missing:
        raise RuntimeError(
            f"Missing Neo4j env vars: {', '.join(missing)} — load .env via "
            "python-dotenv before calling get_driver()."
        )
    # Silence `UNRECOGNIZED` notifications (e.g. "the relationship type
    # CONSIST_OF is not available"). They appear on every OPTIONAL MATCH
    # against a partially-populated graph and have no diagnostic value —
    # a genuine typo still fails as a hard Cypher error. Other classes
    # (DEPRECATION, PERFORMANCE, SECURITY, …) are left enabled.
    return GraphDatabase.driver(
        uri,
        auth=(username, password),
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )


@contextmanager
def driver_session(
    driver: Driver | None = None,
    database: str | None = None,
) -> Iterator[Session]:
    """Yield a Neo4j session, building a driver if one isn't supplied.

    Closes the driver only when this function created it. Useful in scripts /
    notebooks where you want a one-shot connection.
    """
    owned_driver = driver is None
    drv = driver or get_driver()
    try:
        kwargs: dict[str, Any] = {}
        if database:
            kwargs["database"] = database
        with drv.session(**kwargs) as session:
            yield session
    finally:
        if owned_driver:
            drv.close()


def ping(driver: Driver | None = None) -> dict[str, Any]:
    """Health probe: confirm connectivity and report server version.

    Returns:
        Dict::

            {
                "ok": bool,
                "uri": str,
                "server_version": str,
                "edition": str,
                "database": str,
                "constraint_count": int,
                "vector_index_count": int,
                "errors": list[str],
            }
    """
    errors: list[str] = []
    uri = os.getenv("NEO4J_URI", "")
    server_version = ""
    edition = ""
    database = ""
    constraint_count = 0
    vector_index_count = 0

    try:
        owned_driver = driver is None
        drv = driver or get_driver()
    except RuntimeError as exc:
        return {
            "ok": False,
            "uri": uri,
            "server_version": "",
            "edition": "",
            "database": "",
            "constraint_count": 0,
            "vector_index_count": 0,
            "errors": [str(exc)],
        }

    try:
        with drv.session() as session:
            comp = session.run(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions, edition"
            ).single()
            if comp is not None:
                server_version = (comp["versions"] or [""])[0]
                edition = comp["edition"]
            db_row = session.run(
                "CALL db.info() YIELD name RETURN name"
            ).single()
            if db_row is not None:
                database = db_row["name"]
            constraint_count = (
                session.run(
                    "SHOW CONSTRAINTS YIELD name RETURN count(*) AS n"
                ).single()
                or {"n": 0}
            )["n"]
            vector_index_count = (
                session.run(
                    "SHOW INDEXES YIELD type WHERE type = 'VECTOR' "
                    "RETURN count(*) AS n"
                ).single()
                or {"n": 0}
            )["n"]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"neo4j probe failed ({type(exc).__name__}): {exc}")
    finally:
        if owned_driver:
            drv.close()

    return {
        "ok": not errors,
        "uri": uri,
        "server_version": server_version,
        "edition": edition,
        "database": database,
        "constraint_count": constraint_count,
        "vector_index_count": vector_index_count,
        "errors": errors,
    }
