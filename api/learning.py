"""
api/learning.py — Learning pipeline endpoints (Phase 9).

GET  /projects/{id}/insights           — insights extracted from a project
GET  /insights                         — global cross-project insight browser
POST /projects/{id}/retrospective      — manually trigger full retrospective
GET  /projects/{id}/retrospective      — get retrospective summary if run

Insights are memory objects with tier=insight. They are extracted
automatically during execution (continuous) and synthesised at project
closure (retrospective). These endpoints expose them for the UI and
let operators trigger a manual retrospective if needed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import get_db
from db.tables import MemoryObjectRow, ProjectRow
from orchestrator.learning import get_learning_service

router = APIRouter(tags=["learning"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class InsightOut(BaseModel):
    memory_id:   str
    title:       str
    content:     str
    confidence:  float
    status:      str        # draft | verified | canonical
    scope:       str        # project | cross_project | global
    tags:        list[str]
    created_at:  datetime
    project_id:  str | None


class RetrospectiveSummary(BaseModel):
    project_id:           str
    episodes:             int
    approved:             int
    failed:               int
    retried_to_success:   int
    insights_extracted:   int
    insights_promoted:    int
    error:                str | None = None


# ---------------------------------------------------------------------------
# Project insight browser
# ---------------------------------------------------------------------------


@router.get(
    "/projects/{project_id}/insights",
    response_model=list[InsightOut],
    summary="Insights extracted from a project",
)
async def project_insights(
    project_id: str,
    include_cross_project: bool = Query(default=True),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    db: AsyncSession = Depends(get_db),
) -> list[InsightOut]:
    """
    Return all insights associated with a project.
    If include_cross_project=True (default), also returns global insights
    not tied to any specific project (scope=cross_project or global).
    """
    row = await db.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(404, f"Project {project_id!r} not found.")

    import json
    # Project-scoped insights
    q = (
        select(MemoryObjectRow)
        .where(
            MemoryObjectRow.tier == "insight",
            MemoryObjectRow.project_id == project_id,
            MemoryObjectRow.confidence >= min_confidence,
        )
        .order_by(MemoryObjectRow.confidence.desc())
        .limit(limit)
    )
    rows = (await db.execute(q)).scalars().all()

    # Cross-project insights (no project_id, globally visible)
    if include_cross_project:
        xq = (
            select(MemoryObjectRow)
            .where(
                MemoryObjectRow.tier == "insight",
                MemoryObjectRow.project_id.is_(None),
                MemoryObjectRow.confidence >= min_confidence,
            )
            .order_by(MemoryObjectRow.confidence.desc())
            .limit(limit)
        )
        rows = list(rows) + list((await db.execute(xq)).scalars().all())

    return [
        InsightOut(
            memory_id=r.memory_id,
            title=r.title,
            content=r.content,
            confidence=r.confidence,
            status=r.status,
            scope=r.scope,
            tags=json.loads(r.tags_json),
            created_at=r.created_at,
            project_id=r.project_id,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Global insight browser
# ---------------------------------------------------------------------------


@router.get(
    "/insights",
    response_model=list[InsightOut],
    summary="Browse all insights across projects",
)
async def global_insights(
    status:         str | None = Query(default=None),
    min_confidence: float = Query(default=0.6, ge=0.0, le=1.0),
    tag:            str | None = Query(default=None),
    limit:          Annotated[int, Query(ge=1, le=500)] = 100,
    db: AsyncSession = Depends(get_db),
) -> list[InsightOut]:
    """
    Browse the global insight memory — all verified and canonical patterns
    extracted from every project.  Useful for understanding what the system
    has learned over time.
    """
    import json

    q = (
        select(MemoryObjectRow)
        .where(
            MemoryObjectRow.tier == "insight",
            MemoryObjectRow.confidence >= min_confidence,
        )
        .order_by(MemoryObjectRow.confidence.desc(), MemoryObjectRow.created_at.desc())
        .limit(limit)
    )

    if status:
        q = q.where(MemoryObjectRow.status == status)

    rows = (await db.execute(q)).scalars().all()

    # Filter by tag in Python (JSON column — not indexed)
    if tag:
        rows = [r for r in rows if tag in json.loads(r.tags_json)]

    return [
        InsightOut(
            memory_id=r.memory_id,
            title=r.title,
            content=r.content,
            confidence=r.confidence,
            status=r.status,
            scope=r.scope,
            tags=json.loads(r.tags_json),
            created_at=r.created_at,
            project_id=r.project_id,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Manual retrospective trigger
# ---------------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/retrospective",
    response_model=RetrospectiveSummary,
    summary="Manually trigger a project retrospective",
)
async def trigger_retrospective(
    project_id: str,
    background_tasks: BackgroundTasks,
    wait: bool = Query(
        default=False,
        description="If true, wait for completion and return the full summary. "
                    "If false (default), kick off in background and return immediately.",
    ),
    db: AsyncSession = Depends(get_db),
) -> RetrospectiveSummary:
    """
    Run the full learning retrospective for a project.

    By default runs in the background (returns immediately with zeroed counts).
    Set wait=true to block until complete and get the actual counts — useful
    for testing or when you need the result immediately.

    The engine runs this automatically at project closure. Use this endpoint
    to re-run it (e.g. after adding new tasks or manually approving tasks).
    """
    row = await db.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(404, f"Project {project_id!r} not found.")

    svc = get_learning_service()

    if wait:
        result = await svc.run_project_retrospective(project_id)
        return RetrospectiveSummary(**result)

    background_tasks.add_task(svc.run_project_retrospective, project_id)
    return RetrospectiveSummary(
        project_id=project_id,
        episodes=0,
        approved=0,
        failed=0,
        retried_to_success=0,
        insights_extracted=0,
        insights_promoted=0,
    )


# ---------------------------------------------------------------------------
# Promote insights manually
# ---------------------------------------------------------------------------


@router.post(
    "/insights/promote",
    summary="Manually promote eligible draft insights",
)
async def promote_insights() -> dict:
    """
    Run the promotion pass over all draft/verified insights.
    Normally triggered automatically by the retrospective. Call this
    manually after bulk-importing insights or during testing.
    """
    promoted = await get_learning_service()._promote_insights()
    return {"promoted": promoted}
