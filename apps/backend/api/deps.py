"""Shared FastAPI dependencies — Neo4j driver."""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

load_dotenv()


@lru_cache(maxsize=1)
def _make_driver() -> Driver:
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "AncientChina"),
        ),
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )


def get_driver() -> Driver:
    return _make_driver()
