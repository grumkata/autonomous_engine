from __future__ import annotations

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from config import get_settings
from db.engine import close_db, init_db
from llm.client import close_client, init_llm_client

# Routers live in the api/ sub-package
from api.health import router as health_router
from api.projects import router as projects_router
from api.tasks import router as tasks_router
from api.audit import router as audit_router
from api.learning import router as learning_router
from api.checkpoints import router as checkpoints_router
from api.governance import router as governance_router
from api.sse import router as sse_router

settings = get_settings()

UI_DIR = Path(__file__).parent / "ui"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    processors = shared + (
        [structlog.dev.ConsoleRenderer(colors=True)]
        if settings.debug
        else [structlog.processors.dict_tracebacks, structlog.processors.JSONRenderer()]
    )
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )


_configure_logging()
log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

def _install_signal_handlers() -> None:
    """
    Cancel all pending asyncio tasks on SIGINT/SIGTERM before uvicorn shuts down.
    Falls back silently on Windows where add_signal_handler is not supported —
    the lifespan teardown block still runs and cleans up DB + LLM client.
    """
    loop = asyncio.get_event_loop()

    async def _shutdown(sig_name: str) -> None:
        log.warning("signal.received", signal=sig_name)
        current = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks(loop) if t is not current]
        if tasks:
            log.info("shutdown.cancelling_tasks", count=len(tasks))
            for t in tasks:
                t.cancel()
            await asyncio.wait(tasks, timeout=10)
        log.info("shutdown.complete")
        loop.stop()

    try:
        loop.add_signal_handler(signal.SIGINT,  lambda: loop.create_task(_shutdown("SIGINT")))
        loop.add_signal_handler(signal.SIGTERM, lambda: loop.create_task(_shutdown("SIGTERM")))
    except (NotImplementedError, RuntimeError):
        log.debug("signal_handlers.not_supported_on_this_platform")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    log.info("engine.starting", version=settings.app_version, debug=settings.debug)
    await init_db(settings)
    await init_llm_client(settings)

    from orchestrator.skill_registry import init_skill_registry
    await init_skill_registry()
    log.info("engine.ready", ui="http://localhost:8000/ui")

    _install_signal_handlers()

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    log.info("engine.stopping")

    # Phase 7: checkpoint all active projects before killing background tasks
    try:
        from orchestrator.checkpoint_manager import get_checkpoint_manager
        saved = await get_checkpoint_manager().save_all_active()
        log.info("engine.shutdown_checkpoints", count=len(saved))
    except Exception as exc:
        log.warning("engine.shutdown_checkpoint_failed", error=str(exc))

    # Cancel any lingering background tasks (stalled orchestration runs, SSE generators, etc.)
    current = asyncio.current_task()
    background = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    if background:
        log.info("engine.cancelling_background_tasks", count=len(background))
        for t in background:
            t.cancel()
        await asyncio.wait(background, timeout=10)

    await close_client()
    await close_db()
    log.info("engine.stopped")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc)},
        )

    # ── API routers ──────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(projects_router)
    app.include_router(tasks_router)
    app.include_router(audit_router)
    app.include_router(learning_router)
    app.include_router(checkpoints_router)
    app.include_router(governance_router)
    app.include_router(sse_router)

    # ── Root redirect ────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/ui/")

    # ── UI static files ──────────────────────────────────────────────────────
    # Mounted AFTER API routes. html=True serves index.html for directory
    # requests and unknown sub-paths (SPA fallback).
    if not UI_DIR.exists():
        log.error("ui_dir.missing", path=str(UI_DIR))
    else:
        app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
        log.debug("ui.mounted", directory=str(UI_DIR))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
