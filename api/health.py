"""
Health and readiness endpoints.

GET /health       — liveness probe. Always returns 200 if the process is up.
GET /health/ready — readiness probe. Returns 503 if the DB is unreachable.
GET /health/info  — human-readable system info (version, config, status).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import Settings, get_settings
from db.engine import get_db

router = APIRouter(prefix="/health", tags=["health"])

# Track when the process started (for uptime reporting).
_START_TIME = time.monotonic()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class LivenessResponse(BaseModel):
    status: str = "ok"
    timestamp: datetime


class ReadinessResponse(BaseModel):
    status: str
    database: str
    uptime_seconds: float
    timestamp: datetime


class SystemInfoResponse(BaseModel):
    app_name: str
    version: str
    debug: bool
    database_url_scheme: str  # only the scheme, not credentials
    ollama_base_url: str
    default_model: str
    max_concurrent_tasks: int
    uptime_seconds: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/", response_model=LivenessResponse, summary="Liveness probe")
async def liveness() -> LivenessResponse:
    """
    Liveness probe — returns 200 as long as the process is running.
    Used by container orchestrators (Kubernetes, Docker) to know if the
    process needs to be restarted.
    """
    return LivenessResponse(timestamp=datetime.now(timezone.utc))


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def readiness(db: AsyncSession = Depends(get_db)) -> ReadinessResponse:
    """
    Readiness probe — verifies the database is reachable.
    Returns 503 if the DB connection fails.
    Used to delay routing traffic until the app is fully initialized.
    """
    try:
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unreachable: {exc}",
        )

    return ReadinessResponse(
        status="ready",
        database=db_status,
        uptime_seconds=round(time.monotonic() - _START_TIME, 2),
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/info", response_model=SystemInfoResponse, summary="System info")
async def system_info(
    settings: Settings = Depends(get_settings),
) -> SystemInfoResponse:
    """
    Human-readable system info. Not a secret endpoint — avoid exposing
    this publicly in production or put it behind auth (Phase 8).
    """
    scheme = settings.database_url.split("://")[0]
    return SystemInfoResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        database_url_scheme=scheme,
        ollama_base_url=settings.ollama_base_url,
        default_model=settings.ollama_default_model,
        max_concurrent_tasks=settings.max_concurrent_tasks,
        uptime_seconds=round(time.monotonic() - _START_TIME, 2),
        timestamp=datetime.now(timezone.utc),
    )
