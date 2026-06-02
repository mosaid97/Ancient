"""Launch the Ancient Chinese Search Engine web UI.

Usage:
    uv run python scripts/run_server.py
    uv run python scripts/run_server.py --port 8080 --reload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Ancient Chinese Search Engine web server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    print(f"\n  古籍搜索引擎 — Ancient Chinese Search Engine")
    print(f"  UI: http://{args.host}:{args.port}")
    print(f"  API docs: http://{args.host}:{args.port}/docs\n")

    uvicorn.run(
        "apps.backend.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
