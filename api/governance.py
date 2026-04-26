"""
api/governance.py — Human governance and intervention endpoints (Phase 6).

All endpoints from spec §12:

  POST /projects/{id}/freeze        — halt dispatch; in-flight tasks complete
  POST /projects/{id}/resume        — re-enable dispatch after freeze
  POST /projects/{id}/redirect      — inject new goals/constraints mid-run
  POST /tasks/{id}/note             — attach a human note into task context
  POST /tasks/{id}/veto             — move task to NEEDS_REVIEW; block downstream
  POST /tasks/{id}/approve          — approve a NEEDS_REVIEW task; unlock downstream
  POST /memory/{id}/lock            — promote memory item to CANONICAL directly
  POST /memory/{id}/retire          — mark memory item retired; remove from retrieval
  PATCH /projects/{id}/policy       — modify thresholds, retry budgets, concurrency

Additional read endpoints:
  GET /projects/{id}/workspaces     — list workspace state for a project (Phase 3)
  GET /skills                       — list all loaded skills (Phase 4)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import get_db
from db.mappers import row_to_workspace
from db.tables import MemoryObjectRow, ProjectRow, SkillRow, TaskRow, WorkspaceRow
from models.task import TaskStatus
from orchestrator.audit import AuditEventType, AuditSeverity, audit
from orchestrator.checkpoint_manager import CheckpointTrigger, get_checkpoint_manager
from orchestrator.skill_registry import get_skill_registry
from orchestrator.workspace_manager import get_workspace_manager
from sqlalchemy import select

router = APIRouter(tags=["governance"])


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class RedirectRequest(BaseModel):
    additional_goals:       list[str] = Field(default_factory=list)
    additional_constraints: list[str] = Field(default_factory=list)
    reason:                 str       = Field(default="", max_length=1000)


class NoteRequest(BaseModel):
    note: str = Field(..., min_length=1, max_length=5000)


class VetoRequest(BaseModel):
    reason: str = Field(default="", max_length=1000)


class PolicyPatchRequest(BaseModel):
    """
    Any field left as None is not changed.
    """
    quality_threshold:    float | None = Field(default=None, ge=0.0, le=1.0)
    max_retries:          int   | None = Field(default=None, ge=0)
    max_concurrent_tasks: int   | None = Field(default=None, ge=1, le=100)


# ---------------------------------------------------------------------------
# Project-level governance
# ---------------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/freeze",
    summary="Freeze project — halt new task dispatch",
)
async def freeze_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Set the frozen flag on the project. The orchestration engine checks
    this flag every iteration and skips dispatch while frozen.
    In-flight tasks complete normally.
    """
    row = await db.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(404, f"Project {project_id!r} not found.")
    if row.frozen:
        return {"project_id": project_id, "frozen": True, "message": "Already frozen."}

    row.frozen     = True
    row.updated_at = datetime.now(timezone.utc)

    # Append a governance note to the project log
    notes = json.loads(row.notes_json)
    notes.append({
        "type":       "freeze",
        "message":    "Project frozen by operator.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    row.notes_json = json.dumps(notes)
    await db.commit()

    # Checkpoint on governance event so state is safe before any human changes
    try:
        await get_checkpoint_manager().save(
            project_id, trigger=CheckpointTrigger.HUMAN_INTERVENTION
        )
    except Exception:
        pass   # non-fatal

    audit(
        AuditEventType.HUMAN_FREEZE,
        project_id=project_id,
        actor_type="human",
        after={"frozen": True},
    )
    return {"project_id": project_id, "frozen": True, "message": "Project frozen."}


@router.post(
    "/projects/{project_id}/resume",
    summary="Resume project — re-enable task dispatch",
)
async def resume_project(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Clear the frozen flag. The engine resumes dispatching on the next iteration."""
    row = await db.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(404, f"Project {project_id!r} not found.")

    row.frozen     = False
    row.updated_at = datetime.now(timezone.utc)

    notes = json.loads(row.notes_json)
    notes.append({
        "type":       "resume",
        "message":    "Project resumed by operator.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    row.notes_json = json.dumps(notes)
    await db.commit()

    audit(
        AuditEventType.HUMAN_RESUME,
        project_id=project_id,
        actor_type="human",
        after={"frozen": False},
    )
    return {"project_id": project_id, "frozen": False, "message": "Project resumed."}


@router.post(
    "/projects/{project_id}/redirect",
    summary="Redirect project — inject new goals or constraints",
)
async def redirect_project(
    project_id: str,
    body: RedirectRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Inject additional goals or constraints into the project mid-execution.
    Appends to the existing lists — does not replace them.

    The planning agent will incorporate these on the next planning cycle
    if the project is re-planned. Currently-running tasks are not affected.
    """
    row = await db.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(404, f"Project {project_id!r} not found.")

    goals       = json.loads(row.goals_json)
    constraints = json.loads(row.constraints_json)

    from core.ids import prefixed_id
    now = datetime.now(timezone.utc).isoformat()

    for stmt in body.additional_goals:
        goals.append({
            "goal_id":          prefixed_id("goal"),
            "statement":        stmt,
            "priority":         len(goals) + 1,
            "success_criteria": [],
            "is_locked":        False,
            "created_at":       now,
        })

    for desc in body.additional_constraints:
        constraints.append({
            "constraint_id": prefixed_id("cst"),
            "type":          "domain",
            "description":   desc,
            "is_hard":       True,
            "source":        "human_redirect",
            "created_at":    now,
        })

    row.goals_json       = json.dumps(goals)
    row.constraints_json = json.dumps(constraints)
    row.updated_at       = datetime.now(timezone.utc)

    notes = json.loads(row.notes_json)
    notes.append({
        "type":       "redirect",
        "message":    body.reason or "Redirect by operator.",
        "goals_added": len(body.additional_goals),
        "constraints_added": len(body.additional_constraints),
        "created_at": now,
    })
    row.notes_json = json.dumps(notes)
    await db.commit()

    # Checkpoint after redirect so the new goals/constraints are preserved
    try:
        await get_checkpoint_manager().save(
            project_id, trigger=CheckpointTrigger.HUMAN_INTERVENTION
        )
    except Exception:
        pass

    audit(
        AuditEventType.HUMAN_REDIRECT,
        project_id=project_id,
        actor_type="human",
        goals_added=len(body.additional_goals),
        constraints_added=len(body.additional_constraints),
        reason=body.reason,
    )
    return {
        "project_id":        project_id,
        "goals_added":       len(body.additional_goals),
        "constraints_added": len(body.additional_constraints),
        "total_goals":       len(goals),
        "total_constraints": len(constraints),
    }


@router.patch(
    "/projects/{project_id}/policy",
    summary="Patch project execution policy",
)
async def patch_policy(
    project_id: str,
    body: PolicyPatchRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Modify per-project quality threshold and/or retry budget.
    Only non-None fields are changed.
    """
    row = await db.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(404, f"Project {project_id!r} not found.")

    changes: dict = {}

    # quality_threshold applies to all tasks that haven't started yet.
    # For already-planned tasks we patch them individually.
    if body.quality_threshold is not None:
        task_rows = (
            await db.execute(
                select(TaskRow).where(
                    TaskRow.project_id == project_id,
                    TaskRow.status.in_(["planned", "ready"]),
                )
            )
        ).scalars().all()
        for tr in task_rows:
            tr.quality_threshold = body.quality_threshold
        changes["quality_threshold"] = body.quality_threshold
        changes["tasks_updated"] = len(task_rows)

    if body.max_retries is not None:
        task_rows2 = (
            await db.execute(
                select(TaskRow).where(
                    TaskRow.project_id == project_id,
                    TaskRow.status.in_(["planned", "ready", "failed"]),
                )
            )
        ).scalars().all()
        for tr in task_rows2:
            budget = json.loads(tr.budget_json)
            budget["max_retries"] = body.max_retries
            tr.budget_json = json.dumps(budget)
        changes["max_retries"] = body.max_retries

    row.updated_at = datetime.now(timezone.utc)
    notes = json.loads(row.notes_json)
    notes.append({
        "type":       "policy_change",
        "changes":    changes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    row.notes_json = json.dumps(notes)
    await db.commit()

    audit(
        AuditEventType.HUMAN_POLICY_CHANGE,
        project_id=project_id,
        actor_type="human",
        changes=changes,
    )
    return {"project_id": project_id, "applied": changes}


# ---------------------------------------------------------------------------
# Task-level governance
# ---------------------------------------------------------------------------


@router.post(
    "/tasks/{task_id}/note",
    summary="Inject a human note into a task's agent context",
)
async def add_task_note(
    task_id: str,
    body: NoteRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Append a note to the task's human_notes list. On the next agent run
    this note is injected at the top of the context as a high-priority
    instruction (spec §12 note injection).
    """
    row = await db.get(TaskRow, task_id)
    if not row:
        raise HTTPException(404, f"Task {task_id!r} not found.")

    notes = json.loads(row.human_notes_json)
    notes.append(body.note)
    row.human_notes_json = json.dumps(notes)
    row.updated_at       = datetime.now(timezone.utc)
    await db.commit()

    audit(
        AuditEventType.HUMAN_NOTE_INJECTED,
        project_id=row.project_id,
        task_id=task_id,
        actor_type="human",
        note_preview=body.note[:120],
        total_notes=len(notes),
    )
    return {
        "task_id":     task_id,
        "note_count":  len(notes),
        "message":     "Note injected. Will appear in agent context on next run.",
    }


@router.post(
    "/tasks/{task_id}/veto",
    summary="Veto a task — move to NEEDS_REVIEW and block downstream",
)
async def veto_task(
    task_id: str,
    body: VetoRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Move the task to NEEDS_REVIEW. Downstream tasks will not unlock until
    the task is explicitly approved via POST /tasks/{id}/approve.
    """
    row = await db.get(TaskRow, task_id)
    if not row:
        raise HTTPException(404, f"Task {task_id!r} not found.")

    row.status     = TaskStatus.NEEDS_REVIEW.value
    row.updated_at = datetime.now(timezone.utc)

    # Inject veto reason as a human note so the agent sees it on re-run
    if body.reason:
        notes = json.loads(row.human_notes_json)
        notes.append(f"[VETO] {body.reason}")
        row.human_notes_json = json.dumps(notes)

    await db.commit()

    audit(
        AuditEventType.HUMAN_VETO,
        project_id=row.project_id,
        task_id=task_id,
        actor_type="human",
        severity=AuditSeverity.WARNING,
        before={"status": "in_progress"},
        after={"status": "needs_review"},
        reason=body.reason,
    )
    return {
        "task_id": task_id,
        "status":  TaskStatus.NEEDS_REVIEW.value,
        "reason":  body.reason or "(no reason given)",
    }


@router.post(
    "/tasks/{task_id}/approve",
    summary="Approve a NEEDS_REVIEW task — unlock downstream tasks",
)
async def approve_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Move the task from NEEDS_REVIEW to APPROVED. Downstream tasks whose
    only remaining dependency was this task will become READY on the next
    orchestration iteration.
    """
    row = await db.get(TaskRow, task_id)
    if not row:
        raise HTTPException(404, f"Task {task_id!r} not found.")

    if row.status not in (TaskStatus.NEEDS_REVIEW.value, TaskStatus.BLOCKED.value):
        raise HTTPException(
            409,
            f"Task is {row.status!r}, expected needs_review or blocked.",
        )

    now = datetime.now(timezone.utc)
    row.status       = TaskStatus.APPROVED.value
    row.completed_at = row.completed_at or now
    row.updated_at   = now
    await db.commit()

    audit(
        AuditEventType.HUMAN_APPROVAL,
        project_id=row.project_id,
        task_id=task_id,
        actor_type="human",
        before={"status": row.status},
        after={"status": "approved"},
    )
    return {"task_id": task_id, "status": TaskStatus.APPROVED.value}


# ---------------------------------------------------------------------------
# Memory governance
# ---------------------------------------------------------------------------


@router.post(
    "/memory/{memory_id}/lock",
    summary="Promote a memory item to CANONICAL directly",
)
async def lock_memory(
    memory_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Bypass the promotion pipeline and set a memory item's status to
    CANONICAL and write_policy to restricted (spec §12 memory locking).
    """
    row = await db.get(MemoryObjectRow, memory_id)
    if not row:
        raise HTTPException(404, f"Memory {memory_id!r} not found.")

    row.status       = "canonical"
    row.write_policy = "restricted"
    row.updated_at   = datetime.now(timezone.utc)
    await db.commit()

    audit(
        AuditEventType.MEMORY_LOCKED,
        actor_type="human",
        memory_id=memory_id,
        after={"status": "canonical", "write_policy": "restricted"},
    )
    return {
        "memory_id":   memory_id,
        "status":      "canonical",
        "write_policy": "restricted",
    }


@router.post(
    "/memory/{memory_id}/retire",
    summary="Retire a memory item — remove from active retrieval",
)
async def retire_memory(
    memory_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Mark a memory item as retired. Retired items are excluded from
    semantic retrieval queries (MemoryManager filters on status != retired).
    """
    row = await db.get(MemoryObjectRow, memory_id)
    if not row:
        raise HTTPException(404, f"Memory {memory_id!r} not found.")

    row.status     = "retired"
    row.updated_at = datetime.now(timezone.utc)
    await db.commit()

    audit(
        AuditEventType.MEMORY_RETIRED,
        actor_type="human",
        memory_id=memory_id,
        after={"status": "retired"},
    )
    return {"memory_id": memory_id, "status": "retired"}


# ---------------------------------------------------------------------------
# Read helpers (workspaces + skills)
# ---------------------------------------------------------------------------


@router.get(
    "/projects/{project_id}/workspaces",
    summary="List workspace state for a project",
)
async def list_project_workspaces(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Return all workspace rows for a project — used by the governance UI."""
    result = await db.execute(
        select(WorkspaceRow)
        .where(WorkspaceRow.project_id == project_id)
        .order_by(WorkspaceRow.created_at)
    )
    rows = result.scalars().all()
    return [
        {
            "workspace_id":      r.workspace_id,
            "department":        r.department,
            "status":            r.status,
            "artifact_count":    len(json.loads(r.artifact_ids_json)),
            "tasks_processed":   len(json.loads(r.task_ids_json)),
            "created_at":        r.created_at.isoformat(),
            "archived_at":       r.archived_at.isoformat() if r.archived_at else None,
        }
        for r in rows
    ]


@router.get(
    "/skills",
    summary="List all loaded skills",
)
async def list_skills(
    department: str | None = Query(default=None),
) -> list[dict]:
    """Return the skill registry — used by the UI skill selector."""
    skills = await get_skill_registry().list_skills(department=department)
    return [
        {
            "skill_id":    s.skill_id,
            "name":        s.name,
            "version":     s.version,
            "description": s.description,
            "departments": s.departments,
            "source":      s.source,
            "slug":        s.slug,
        }
        for s in skills
    ]


@router.get(
    "/projects/{project_id}/notes",
    summary="Get the governance notes log for a project",
)
async def get_project_notes(
    project_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Return the human-readable governance event log for a project."""
    row = await db.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(404, f"Project {project_id!r} not found.")
    return json.loads(row.notes_json)
