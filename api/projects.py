"""
Project CRUD endpoints.

POST   /projects                       — create project + auto-trigger planning
GET    /projects                       — list projects (filter by status)
GET    /projects/{project_id}          — get full project detail
PATCH  /projects/{project_id}/status   — advance project lifecycle stage
DELETE /projects/{project_id}          — soft-delete (archive)

Phase 2 change: POST / now immediately queues a background task that calls
orchestrator/planner.py → plan_and_run(). The endpoint still returns 201
synchronously — planning and execution happen in the background.
No manual POST /projects/{id}/run is needed for normal use.
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import get_db
from db.mappers import project_to_row, row_to_project
from db.tables import ProjectRow
from models.project import (
    CreateProjectRequest,
    Goal,
    Project,
    ProjectStatus,
    ProjectSummary,
)

router = APIRouter(prefix="/projects", tags=["projects"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=Project,
    status_code=status.HTTP_201_CREATED,
    summary="Create a project and auto-start planning",
)
async def create_project(
    body: CreateProjectRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> Project:
    """
    Create a project from raw intake text.

    After the row is committed, a background task calls the planning agent
    which decomposes raw_intake into a task graph and starts orchestration.
    No further API calls are needed — the engine runs autonomously.

    Watch progress via GET /projects/{id}/graph or the SSE stream at
    GET /projects/{id}/stream.
    """
    goals = [
        Goal(statement=stmt, priority=i + 1)
        for i, stmt in enumerate(body.initial_goals)
    ]

    project = Project(
        title=body.title,
        description=body.description,
        type=body.type,
        raw_intake=body.raw_intake,
        goals=goals,
        tags=body.tags,
        status=ProjectStatus.INTAKE,
    )

    row = project_to_row(project)
    db.add(row)
    await db.commit()

    # ── Phase 2: auto-trigger the planning + execution pipeline ─────────────
    # We capture project here (not just project_id) so the background task
    # has the full object without an extra DB round-trip.
    async def _plan_and_run() -> None:
        try:
            from orchestrator.planner import plan_and_run
            await plan_and_run(project)
        except Exception as exc:
            import structlog
            structlog.get_logger(__name__).error(
                "background.plan_and_run_failed",
                project_id=project.project_id,
                error=str(exc),
            )

    background_tasks.add_task(_plan_and_run)

    return project


@router.get(
    "/",
    response_model=list[ProjectSummary],
    summary="List projects",
)
async def list_projects(
    status_filter: Annotated[ProjectStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    db: AsyncSession = Depends(get_db),
) -> list[ProjectSummary]:
    q = (
        select(ProjectRow)
        .order_by(ProjectRow.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if status_filter:
        q = q.where(ProjectRow.status == status_filter.value)

    rows = (await db.execute(q)).scalars().all()
    return [ProjectSummary.from_project(row_to_project(row)) for row in rows]


@router.get(
    "/{project_id}",
    response_model=Project,
    summary="Get project detail",
)
async def get_project(project_id: str, db: AsyncSession = Depends(get_db)) -> Project:
    row = await db.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Project {project_id!r} not found.")
    return row_to_project(row)


@router.patch(
    "/{project_id}/status",
    response_model=Project,
    summary="Advance project status",
)
async def update_project_status(
    project_id: str,
    new_status: ProjectStatus,
    db: AsyncSession = Depends(get_db),
) -> Project:
    """Manually advance the project lifecycle. Engine also calls this automatically."""
    row = await db.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Project {project_id!r} not found.")
    row.status = new_status.value
    await db.commit()
    return row_to_project(row)


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Archive a project",
)
async def archive_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Soft-delete by setting status to ARCHIVED. All data is retained."""
    row = await db.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Project {project_id!r} not found.")
    row.status = ProjectStatus.ARCHIVED.value
    await db.commit()
    return Response(status_code=204)
