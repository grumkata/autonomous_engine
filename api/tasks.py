"""
Task CRUD endpoints + orchestration trigger.

POST   /tasks                         — create a task in a project
GET    /tasks?project_id=xxx          — list tasks (filter by status/dept)
GET    /tasks/{task_id}               — get full task detail
PATCH  /tasks/{task_id}/status        — manual status override
DELETE /tasks/{task_id}               — soft-archive

POST   /projects/{project_id}/run     — trigger autonomous orchestration
GET    /projects/{project_id}/graph   — task graph snapshot (for UI)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from db.engine import get_db
from db.tables import ProjectRow, TaskEdgeRow, TaskRow
from models.task import (
    CreateTaskRequest,
    DepartmentOwner,
    Task,
    TaskBudget,
    TaskEdge,
    TaskPriority,
    TaskStatus,
    TaskStatusUpdate,
    TaskType,
)

router = APIRouter(tags=["tasks"])


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _row_to_task(row: TaskRow) -> Task:
    from models.task import TaskAttempt
    return Task(
        task_id=row.task_id,
        project_id=row.project_id,
        type=row.type,
        title=row.title,
        description=row.description,
        status=row.status,
        priority=row.priority,
        owner_department=row.owner_department,
        parent_task_id=row.parent_task_id,
        depth_level=row.depth_level,
        assigned_agent_ids=json.loads(row.assigned_agent_ids_json),
        input_artifact_ids=json.loads(row.input_artifact_ids_json),
        expected_output_types=json.loads(row.expected_output_types_json),
        output_artifact_ids=json.loads(row.output_artifact_ids_json),
        validation_criteria=json.loads(row.validation_criteria_json),
        budget=TaskBudget(**json.loads(row.budget_json)),
        attempts=[TaskAttempt(**a) for a in json.loads(row.attempts_json)],
        attempt_count=row.attempt_count,
        quality_threshold=row.quality_threshold,
        requires_human_approval=row.requires_human_approval,
        created_at=row.created_at,
        updated_at=row.updated_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
    )


def _task_to_row(task: Task) -> TaskRow:
    return TaskRow(
        task_id=task.task_id,
        project_id=task.project_id,
        type=task.type.value,
        title=task.title,
        description=task.description,
        status=task.status.value,
        priority=task.priority.value,
        owner_department=task.owner_department.value,
        parent_task_id=task.parent_task_id,
        depth_level=task.depth_level,
        validation_criteria_json=json.dumps(task.validation_criteria),
        expected_output_types_json=json.dumps(task.expected_output_types),
        input_artifact_ids_json="[]",
        output_artifact_ids_json="[]",
        assigned_agent_ids_json="[]",
        budget_json=json.dumps(task.budget.model_dump(mode="json")),
        quality_threshold=task.quality_threshold,
        requires_human_approval=task.requires_human_approval,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


# ---------------------------------------------------------------------------
# Task CRUD
# ---------------------------------------------------------------------------


@router.post(
    "/tasks",
    response_model=Task,
    status_code=status.HTTP_201_CREATED,
    summary="Create a task",
)
async def create_task(
    body: CreateTaskRequest,
    db: AsyncSession = Depends(get_db),
) -> Task:
    """
    Create a task and optionally wire dependency edges.

    If `dependency_task_ids` is provided, edges are created so this task
    will only become READY once all listed upstream tasks are APPROVED.
    Tasks with no dependencies start as READY immediately.
    """
    # Verify project exists
    proj = await db.get(ProjectRow, body.project_id)
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project {body.project_id!r} not found.")

    # Determine initial status: READY if no deps, PLANNED if deps exist
    initial_status = (
        TaskStatus.PLANNED if body.dependency_task_ids else TaskStatus.READY
    )

    task = Task(
        project_id=body.project_id,
        type=body.type,
        title=body.title,
        description=body.description,
        owner_department=body.owner_department,
        priority=body.priority,
        validation_criteria=body.validation_criteria,
        requires_human_approval=body.requires_human_approval,
        status=initial_status,
    )

    row = _task_to_row(task)
    db.add(row)
    await db.flush()

    # Create dependency edges
    for dep_id in body.dependency_task_ids:
        dep_row = await db.get(TaskRow, dep_id)
        if not dep_row:
            raise HTTPException(
                status_code=404,
                detail=f"Dependency task {dep_id!r} not found.",
            )
        edge = TaskEdge(
            project_id=body.project_id,
            upstream_task_id=dep_id,
            downstream_task_id=task.task_id,
        )
        db.add(
            TaskEdgeRow(
                edge_id=edge.edge_id,
                project_id=edge.project_id,
                upstream_task_id=edge.upstream_task_id,
                downstream_task_id=edge.downstream_task_id,
                edge_type=edge.edge_type,
                created_at=edge.created_at,
            )
        )

    await db.commit()
    return task


@router.get(
    "/tasks",
    response_model=list[Task],
    summary="List tasks for a project",
)
async def list_tasks(
    project_id: Annotated[str, Query(description="Required — filter by project ID")],
    status_filter: Annotated[TaskStatus | None, Query(alias="status")] = None,
    department: Annotated[DepartmentOwner | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    db: AsyncSession = Depends(get_db),
) -> list[Task]:
    q = (
        select(TaskRow)
        .where(TaskRow.project_id == project_id)
        .order_by(TaskRow.depth_level.asc(), TaskRow.created_at.asc())
        .limit(limit)
    )
    if status_filter:
        q = q.where(TaskRow.status == status_filter.value)
    if department:
        q = q.where(TaskRow.owner_department == department.value)

    rows = (await db.execute(q)).scalars().all()
    return [_row_to_task(r) for r in rows]


@router.get(
    "/tasks/{task_id}",
    response_model=Task,
    summary="Get task detail",
)
async def get_task(task_id: str, db: AsyncSession = Depends(get_db)) -> Task:
    row = await db.get(TaskRow, task_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found.")
    return _row_to_task(row)


@router.patch(
    "/tasks/{task_id}/status",
    response_model=Task,
    summary="Manually override task status",
)
async def update_task_status(
    task_id: str,
    body: TaskStatusUpdate,
    db: AsyncSession = Depends(get_db),
) -> Task:
    """
    Allows human operators to force a task into any state.
    Useful for unblocking stuck tasks or overriding engine decisions.
    """
    row = await db.get(TaskRow, task_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found.")

    now = datetime.now(timezone.utc)
    row.status = body.status.value
    row.updated_at = now

    terminal = {TaskStatus.APPROVED, TaskStatus.FAILED, TaskStatus.ARCHIVED}
    if body.status in terminal and not row.completed_at:
        row.completed_at = now

    await db.commit()
    return _row_to_task(row)


@router.delete(
    "/tasks/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Archive a task",
)
async def archive_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    row = await db.get(TaskRow, task_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found.")
    row.status = TaskStatus.ARCHIVED.value
    row.updated_at = datetime.now(timezone.utc)
    await db.commit()
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Task graph view
# ---------------------------------------------------------------------------


@router.get(
    "/projects/{project_id}/graph",
    summary="Task graph snapshot",
)
async def get_task_graph(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Returns all tasks + edges for a project — useful for rendering the
    DAG in a frontend or inspecting orchestration state.
    """
    proj = await db.get(ProjectRow, project_id)
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project {project_id!r} not found.")

    task_rows = (
        await db.execute(select(TaskRow).where(TaskRow.project_id == project_id))
    ).scalars().all()

    edge_rows = (
        await db.execute(select(TaskEdgeRow).where(TaskEdgeRow.project_id == project_id))
    ).scalars().all()

    # Status summary counts
    status_counts: dict[str, int] = {}
    for row in task_rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1

    return {
        "project_id": project_id,
        "project_status": proj.status,
        "task_count": len(task_rows),
        "status_summary": status_counts,
        "tasks": [
            {
                "task_id": r.task_id,
                "title": r.title,
                "type": r.type,
                "status": r.status,
                "priority": r.priority,
                "owner_department": r.owner_department,
                "depth_level": r.depth_level,
                "attempt_count": r.attempt_count,
                "parent_task_id": r.parent_task_id,
            }
            for r in task_rows
        ],
        "edges": [
            {
                "edge_id": e.edge_id,
                "upstream_task_id": e.upstream_task_id,
                "downstream_task_id": e.downstream_task_id,
                "edge_type": e.edge_type,
            }
            for e in edge_rows
        ],
    }


# ---------------------------------------------------------------------------
# Orchestration trigger
# ---------------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/run",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start autonomous orchestration",
)
async def run_project(
    project_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Kick off the autonomous orchestration engine for this project.

    Returns 202 immediately. The engine runs in the background as an
    asyncio task and updates task/project status in the DB as it works.

    Poll GET /projects/{project_id} or GET /projects/{project_id}/graph
    for live status. The project transitions to REVIEW when the engine
    finishes (success, partial failure, or deadlock).

    Prerequisites:
      - Project must exist and have at least one READY or PLANNED task.
      - Tasks without dependencies start as READY automatically.
    """
    proj = await db.get(ProjectRow, project_id)
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project {project_id!r} not found.")

    if proj.status == "archived":
        raise HTTPException(status_code=409, detail="Cannot run an archived project.")

    # Check there are tasks to run
    task_count = len(
        (await db.execute(select(TaskRow).where(TaskRow.project_id == project_id))).scalars().all()
    )
    if task_count == 0:
        raise HTTPException(
            status_code=422,
            detail="Project has no tasks. Create tasks before triggering orchestration.",
        )

    settings = get_settings()

    async def _run_engine() -> None:
        from orchestrator.engine_orchestrator import OrchestrationEngine
        engine = OrchestrationEngine(settings)
        try:
            await engine.run_project(project_id)
        except Exception as exc:
            log_ref = __import__("structlog").get_logger(__name__)
            log_ref.error("engine.unhandled_exception", project_id=project_id, error=str(exc))

    background_tasks.add_task(_run_engine)

    return {
        "accepted": True,
        "project_id": project_id,
        "task_count": task_count,
        "message": (
            "Orchestration started. "
            "Poll GET /projects/{project_id}/graph for live status."
        ),
    }
