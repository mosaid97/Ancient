"""FastAPI application — Ancient Chinese Search Engine UI backend.

The API runs unauthenticated and must be bound to localhost only.
Add an auth layer before any non-localhost exposure.

Launch:
    uv run python scripts/run_server.py
    # or directly:
    uv run uvicorn apps.backend.api.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from apps.backend.api.routers import (  # noqa: E402
    admin,
    documents,
    hitl,
    interactive,
    media,
    search,
    upload,
)
from slowapi import _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402

from apps.backend.auth import keys as auth_keys  # noqa: E402
from apps.backend.auth import router as auth_router  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="古籍搜索引擎 — Ancient Chinese Search Engine",
    description="Research search platform for Tang-era Chinese corpus",
    version="1.0.0",
)

_cors_env = os.getenv("CORS_ORIGINS", "").strip()
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else [
        "http://localhost:8000", "http://127.0.0.1:8000",
        "http://localhost:8013", "http://127.0.0.1:8013",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Hook slowapi for the login rate limiter declared on auth_router.
app.state.limiter = auth_router.limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(search.router,    prefix="/api/search",      tags=["search"])
app.include_router(documents.router, prefix="/api/documents",   tags=["documents"])
app.include_router(media.router,     prefix="/api",             tags=["media"])
app.include_router(hitl.router,      prefix="/api/hitl",        tags=["hitl"])
app.include_router(upload.router,    prefix="/api/upload",      tags=["upload"])
app.include_router(interactive.router, prefix="/api/interactive", tags=["interactive"])
app.include_router(admin.router,     prefix="/api/admin",       tags=["admin"])
app.include_router(auth_router.router, prefix="/api/auth",      tags=["auth"])
app.include_router(auth_keys.router, prefix="/api/keys",        tags=["keys"])

# ---------------------------------------------------------------------------
# Static / SPA
# ---------------------------------------------------------------------------

_FRONTEND = Path(__file__).resolve().parents[3] / "apps" / "frontend"


@app.get("/health")
async def health_check() -> dict:
    """Public health probe — used by the frontend system-health panel."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(_FRONTEND / "index.html")


if (_FRONTEND / "static").exists():
    app.mount("/static", StaticFiles(directory=_FRONTEND / "static"), name="static")


@app.exception_handler(404)
async def spa_fallback(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    return FileResponse(_FRONTEND / "index.html")


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup():
    log.info("Ancient Chinese Search Engine API starting up")
    log.info("Frontend: %s", _FRONTEND)
    app.state.bm25_corpus = None
    app.state.reranker_ready = False

    async def _build_bm25():
        from apps.backend.retrieval.bm25 import BM25Corpus
        from apps.backend.api.deps import get_driver
        try:
            log.info("BM25Corpus: starting build (may take 1-2 min for 40K chunks)…")
            t0 = time.time()
            corpus = await asyncio.get_event_loop().run_in_executor(
                None, lambda: BM25Corpus.build(get_driver())
            )
            app.state.bm25_corpus = corpus
            log.info("BM25Corpus: ready — %d chunks indexed in %.1fs",
                     len(corpus.chunk_ids), time.time() - t0)
        except Exception as exc:
            log.error("BM25Corpus build failed: %s", exc)

    async def _warm_reranker():
        try:
            log.info("Reranker: pre-warming from cache…")
            t0 = time.time()
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: __import__(
                    "apps.backend.agents.rerank", fromlist=["_get_reranker"]
                )._get_reranker(),
            )
            app.state.reranker_ready = True
            log.info("Reranker: ready in %.1fs", time.time() - t0)
        except Exception as exc:
            log.error("Reranker pre-warm failed (will load on first request): %s", exc)

    asyncio.create_task(_build_bm25())
    asyncio.create_task(_warm_reranker())


@app.on_event("shutdown")
async def shutdown():
    from apps.backend.api.deps import _make_driver
    try:
        _make_driver().close()
    except Exception:
        pass
    log.info("Server shut down cleanly")
