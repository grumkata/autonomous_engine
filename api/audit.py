"""
api/audit.py — Audit trail query endpoints (Phase 8).

GET /projects/{id}/audit          — project timeline (filterable)
GET /tasks/{id}/audit             — all events for one task
GET /tasks/{id}/replay            — full decision replay for one task
GET /audit/events                 — global event search (admin use)
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import get_db
from db.tables import ProjectRow, TaskRow
from models.audit import AuditEvent
from orchestrator.audit import TaskReplay, get_audit_service

router = APIRouter(tags=["audit"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class AuditEventOut(BaseModel):
    event_id:    str
    project_id:  str | None
    task_id:     str | None
    actor_type:  str
    actor_id:    str
    event_type:  str
    severity:    str
    before_state: dict
    after_state:  dict
    detail:       dict
    occurred_at:  datetime

    @classmethod
    def from_domain(cls, ev: AuditEvent) -> "AuditEventOut":
        return cls(
            event_id=ev.event_id,
            project_id=ev.project_id,
            task_id=ev.task_id,
            actor_type=ev.actor_type,
            actor_id=ev.actor_id,
            event_type=ev.event_type,
            severity=ev.severity,
            before_state=ev.before_state,
            after_state=ev.after_state,
            detail=ev.detail,
            occurred_at=ev.occurred_at,
        )


# ---------------------------------------------------------------------------
# Project timeline
# ---------------------------------------------------------------------------


@router.get(
    "/projects/{project_id}/audit",
    response_model=list[AuditEventOut],
    summary="Project audit timeline",
)
async def project_timeline(
    project_id: str,
    event_type: Annotated[list[str] | None, Query()] = None,
    severity:   str | None = Query(default=None),
    since:      datetime | None = Query(default=None),
    until:      datetime | None = Query(default=None),
    limit:      Annotated[int, Query(ge=1, le=1000)] = 200,
    db: AsyncSession = Depends(get_db),
) -> list[AuditEventOut]:
    """
    Return the audit timeline for a project, newest first.

    Filter by event_type (repeatable), severity, and time range.
    Use this to power the timeline view in the UI.
    """
    row = await db.get(ProjectRow, project_id)
    if not row:
        raise HTTPException(404, f"Project {project_id!r} not found.")

    svc = get_audit_service()
    events = await svc.query(
        project_id=project_id,
        event_types=event_type or None,
        severity=severity,
        since=since,
        until=until,
        limit=limit,
    )
    return [AuditEventOut.from_domain(e) for e in events]


# ---------------------------------------------------------------------------
# Task audit trail
# ---------------------------------------------------------------------------


@router.get(
    "/tasks/{task_id}/audit",
    response_model=list[AuditEventOut],
    summary="All audit events for a task",
)
async def task_audit(
    task_id: str,
    limit:   Annotated[int, Query(ge=1, le=500)] = 100,
    db: AsyncSession = Depends(get_db),
) -> list[AuditEventOut]:
    """Return all audit events for a specific task, oldest first."""
    row = await db.get(TaskRow, task_id)
    if not row:
        raise HTTPException(404, f"Task {task_id!r} not found.")

    svc = get_audit_service()
    events = await svc.query(task_id=task_id, limit=limit)
    # Return oldest first so the UI can render a chronological timeline
    return [AuditEventOut.from_domain(e) for e in reversed(events)]


# ---------------------------------------------------------------------------
# Decision replay
# ---------------------------------------------------------------------------


class AttemptRecordOut(BaseModel):
    attempt_number:      int
    agent_started_at:    datetime | None
    agent_completed_at:  datetime | None
    validation_result:   str | None
    composite_score:     float | None
    failure_reason:      str | None
    corrective_feedback: str | None
    llm_latency_ms:      int | None
    tokens_used:         int | None
    skills_loaded:       list[str]


class TaskReplayOut(BaseModel):
    task_id:      str
    project_id:   str | None
    final_status: str | None
    attempts:     list[AttemptRecordOut]
    event_count:  int
    events:       list[AuditEventOut]


@router.get(
    "/tasks/{task_id}/replay",
    response_model=TaskReplayOut,
    summary="Decision replay for a task",
)
async def task_replay(
    task_id: str,
    db: AsyncSession = Depends(get_db),
) -> TaskReplayOut:
    """
    Reconstruct the full decision chain for a task.

    Returns every audit event plus a structured attempt-by-attempt breakdown:
    when the agent started, what validation scored, what corrective feedback
    was injected, and what the final outcome was.

    Use this to answer 'why did this task fail?' or 'what changed between
    attempt 2 and attempt 3?'
    """
    row = await db.get(TaskRow, task_id)
    if not row:
        raise HTTPException(404, f"Task {task_id!r} not found.")

    replay: TaskReplay = await get_audit_service().replay_task(task_id)

    return TaskReplayOut(
        task_id=replay.task_id,
        project_id=replay.project_id,
        final_status=replay.final_status,
        event_count=len(replay.events),
        attempts=[
            AttemptRecordOut(
                attempt_number=a.attempt_number,
                agent_started_at=a.agent_started_at,
                agent_completed_at=a.agent_completed_at,
                validation_result=a.validation_result,
                composite_score=a.composite_score,
                failure_reason=a.failure_reason,
                corrective_feedback=a.corrective_feedback,
                llm_latency_ms=a.llm_latency_ms,
                tokens_used=a.tokens_used,
                skills_loaded=a.skills_loaded,
            )
            for a in replay.attempts
        ],
        events=[AuditEventOut.from_domain(e) for e in replay.events],
    )


# ---------------------------------------------------------------------------
# Global event search (admin / debug)
# ---------------------------------------------------------------------------


@router.get(
    "/audit/events",
    response_model=list[AuditEventOut],
    summary="Global audit event search",
)
async def global_audit(
    event_type: Annotated[list[str] | None, Query()] = None,
    severity:   str | None = Query(default=None),
    since:      datetime | None = Query(default=None),
    until:      datetime | None = Query(default=None),
    limit:      Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[AuditEventOut]:
    """
    Search audit events across all projects.
    Intended for admin / debugging use.
    """
    svc = get_audit_service()
    events = await svc.query(
        event_types=event_type or None,
        severity=severity,
        since=since,
        until=until,
        limit=limit,
    )
    return [AuditEventOut.from_domain(e) for e in events]
