"""
api/checkpoints.py — Checkpoint management endpoints (Phase 7).

GET    /projects/{id}/checkpoints          — list checkpoints (newest first)
GET    /checkpoints/{checkpoint_id}        — get full checkpoint detail
POST   /projects/{id}/checkpoints          — manually trigger a checkpoint
POST   /checkpoints/{checkpoint_id}/restore — restore project to checkpoint state
DELETE /checkpoints/{checkpoint_id}        — delete a checkpoint
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import get_db
from db.tables import ProjectRow
from models.checkpoint import CheckpointTrigger
from orchestrator.checkpoint_manager import CheckpointSummary, get_checkpoint_manager

router = APIRouter(tags=["checkpoints"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class CheckpointDetail(BaseModel):
    checkpoint_id:  str
    project_id:     str
    trigger_reason: str
    task_count:     int
    approved_count: int
    in_progress_count: int
    iteration:      int
    created_at:     str
    # Summary counts per status — avoids sending the full snapshot blob
    task_status_summary: dict[str, int]


class RestoreResponse(BaseModel):
    project_id:     str
    checkpoint_id:  str
    tasks_restored: int
    message:        str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/projects/{project_id}/checkpoints",
    response_model=list[CheckpointSummary],
    summary="List checkpoints for a project",
)
async def list_checkpoints(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[CheckpointSummary]:
    """Return all checkpoints for a project, newest first."""
    row = await db.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(404, f"Project {project_id!r} not found.")
    return await get_checkpoint_manager().list_for_project(project_id)


@router.get(
    "/checkpoints/{checkpoint_id}",
    response_model=CheckpointDetail,
    summary="Get checkpoint detail",
)
async def get_checkpoint(checkpoint_id: str) -> CheckpointDetail:
    """Return metadata and status summary for a checkpoint."""
    ckp = await get_checkpoint_manager().get(checkpoint_id)
    if not ckp:
        raise HTTPException(404, f"Checkpoint {checkpoint_id!r} not found.")

    # Build a status summary from task_states without sending the full blob
    status_counts: dict[str, int] = {}
    for ts in ckp.task_states:
        s = ts.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    return CheckpointDetail(
        checkpoint_id=ckp.checkpoint_id,
        project_id=ckp.project_id,
        trigger_reason=ckp.trigger_reason,
        task_count=ckp.task_count,
        approved_count=ckp.approved_count,
        in_progress_count=ckp.in_progress_count,
        iteration=ckp.iteration,
        created_at=ckp.created_at.isoformat(),
        task_status_summary=status_counts,
    )


@router.post(
    "/projects/{project_id}/checkpoints",
    response_model=CheckpointSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Manually create a checkpoint",
)
async def create_checkpoint(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> CheckpointSummary:
    """
    Save a manual checkpoint immediately.
    Useful before risky governance operations (redirect, policy change).
    """
    row = await db.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(404, f"Project {project_id!r} not found.")

    ckp = await get_checkpoint_manager().save(
        project_id, trigger=CheckpointTrigger.MANUAL
    )
    return CheckpointSummary(
        checkpoint_id=ckp.checkpoint_id,
        project_id=ckp.project_id,
        trigger_reason=ckp.trigger_reason,
        task_count=ckp.task_count,
        approved_count=ckp.approved_count,
        iteration=ckp.iteration,
        created_at=ckp.created_at,
    )


@router.post(
    "/checkpoints/{checkpoint_id}/restore",
    response_model=RestoreResponse,
    summary="Restore project to a checkpoint state and re-run",
)
async def restore_checkpoint(
    checkpoint_id: str,
    background_tasks: BackgroundTasks,
) -> RestoreResponse:
    """
    Restore the project to this checkpoint's state.

    - Tasks that were IN_PROGRESS are reset to READY.
    - APPROVED / FAILED tasks are left as-is.
    - Project status is set back to EXECUTION.
    - The orchestration engine is re-launched in the background automatically.

    Returns immediately (202-style even though status is 200) — watch
    GET /projects/{id}/graph for live progress.
    """
    mgr = get_checkpoint_manager()
    ckp = await mgr.get(checkpoint_id)
    if not ckp:
        raise HTTPException(404, f"Checkpoint {checkpoint_id!r} not found.")

    project_id = await mgr.restore(checkpoint_id)

    # Re-launch the orchestration engine in the background
    async def _rerun() -> None:
        try:
            from config import get_settings
            from orchestrator.engine_orchestrator import OrchestrationEngine
            engine = OrchestrationEngine(get_settings())
            await engine.run_project(project_id)
        except Exception as exc:
            import structlog
            structlog.get_logger(__name__).error(
                "checkpoint.rerun_failed",
                project_id=project_id,
                error=str(exc),
            )

    background_tasks.add_task(_rerun)

    return RestoreResponse(
        project_id=project_id,
        checkpoint_id=checkpoint_id,
        tasks_restored=ckp.task_count,
        message=(
            "Project restored. Orchestration engine re-launched. "
            "Watch GET /projects/{project_id}/graph for live progress."
        ),
    )


@router.delete(
    "/checkpoints/{checkpoint_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a checkpoint",
)
async def delete_checkpoint(checkpoint_id: str) -> Response:
    """Permanently delete a checkpoint. Cannot be undone."""
    try:
        await get_checkpoint_manager().delete(checkpoint_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return Response(status_code=204)
