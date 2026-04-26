"""
orchestrator/audit.py — Audit event emission and query service (Phase 8).

Public API
----------
emit(event_type, *, project_id, task_id, actor_type, actor_id,
     severity, before, after, **detail)          → None   (fire-and-forget)

query(project_id?, task_id?, event_types?,
      severity?, since?, until?, limit)          → list[AuditEvent]

replay_task(task_id)                             → TaskReplay
                                                   (full context + decisions)

get_audit_service()                              → AuditService (singleton)

Design decisions
----------------
- emit() is fully non-blocking: it schedules the DB write as an asyncio
  background task so the hot path (agent execution, validation) is never
  slowed by a DB round-trip.  Failures are logged but never raised.

- The query layer is deliberately simple: indexed columns only (project_id,
  task_id, event_type, severity, occurred_at).  No full-text search — that
  can be added later via a search index.

- replay_task() reconstructs the full decision chain for one task: every
  agent_started → validation_result → task_approved/failed/retry event,
  with the detail payloads intact.  This is the "decision replay" feature
  from spec §8.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import structlog
from pydantic import BaseModel
from sqlalchemy import select

from core.ids import prefixed_id
from db.engine import session_factory
from db.tables import AuditEventRow
from models.audit import AuditEvent, AuditEventType, AuditSeverity

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Task replay response model
# ---------------------------------------------------------------------------


class AttemptRecord(BaseModel):
    attempt_number: int
    agent_started_at:    datetime | None = None
    agent_completed_at:  datetime | None = None
    validation_result:   str | None = None
    composite_score:     float | None = None
    failure_reason:      str | None = None
    corrective_feedback: str | None = None
    llm_latency_ms:      int | None = None
    tokens_used:         int | None = None
    skills_loaded:       list[str] = []


class TaskReplay(BaseModel):
    task_id:    str
    project_id: str | None
    events:     list[AuditEvent]
    attempts:   list[AttemptRecord]
    final_status: str | None = None


# ---------------------------------------------------------------------------
# AuditService
# ---------------------------------------------------------------------------


class AuditService:

    # ------------------------------------------------------------------ #
    # Emit                                                                #
    # ------------------------------------------------------------------ #

    def emit(
        self,
        event_type: str,
        *,
        project_id:  str | None = None,
        task_id:     str | None = None,
        actor_type:  str = "engine",
        actor_id:    str = "",
        severity:    str = AuditSeverity.INFO,
        before:      dict[str, Any] | None = None,
        after:       dict[str, Any] | None = None,
        **detail: Any,
    ) -> None:
        """
        Fire-and-forget: schedule a DB write in the background.
        Never raises — audit failures must not interrupt the engine.
        """
        event = AuditEvent(
            project_id=project_id,
            task_id=task_id,
            actor_type=actor_type,
            actor_id=actor_id,
            event_type=event_type,
            severity=severity,
            before_state=before or {},
            after_state=after or {},
            detail=detail,
        )
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._write(event))
            else:
                # Fallback: fire from sync context (e.g. startup)
                asyncio.run(self._write(event))
        except Exception as exc:
            log.warning("audit.emit_failed", event_type=event_type, error=str(exc))

    async def emit_async(
        self,
        event_type: str,
        *,
        project_id:  str | None = None,
        task_id:     str | None = None,
        actor_type:  str = "engine",
        actor_id:    str = "",
        severity:    str = AuditSeverity.INFO,
        before:      dict[str, Any] | None = None,
        after:       dict[str, Any] | None = None,
        **detail: Any,
    ) -> None:
        """Awaitable version for use inside async functions."""
        event = AuditEvent(
            project_id=project_id,
            task_id=task_id,
            actor_type=actor_type,
            actor_id=actor_id,
            event_type=event_type,
            severity=severity,
            before_state=before or {},
            after_state=after or {},
            detail=detail,
        )
        await self._write(event)

    async def _write(self, event: AuditEvent) -> None:
        try:
            async with session_factory() as db:
                db.add(AuditEventRow(
                    event_id=event.event_id,
                    project_id=event.project_id,
                    task_id=event.task_id,
                    actor_type=event.actor_type,
                    actor_id=event.actor_id,
                    event_type=event.event_type,
                    severity=event.severity,
                    before_json=json.dumps(event.before_state, default=str),
                    after_json=json.dumps(event.after_state,  default=str),
                    detail_json=json.dumps(event.detail,      default=str),
                    occurred_at=event.occurred_at,
                ))
        except Exception as exc:
            log.warning(
                "audit.write_failed",
                event_type=event.event_type,
                error=str(exc),
            )

    # ------------------------------------------------------------------ #
    # Query                                                               #
    # ------------------------------------------------------------------ #

    async def query(
        self,
        project_id:   str | None = None,
        task_id:      str | None = None,
        event_types:  list[str] | None = None,
        severity:     str | None = None,
        since:        datetime | None = None,
        until:        datetime | None = None,
        limit:        int = 200,
    ) -> list[AuditEvent]:
        """
        Return audit events matching the given filters, newest first.
        All filters are optional and additive (AND logic).
        """
        async with session_factory() as db:
            q = select(AuditEventRow).order_by(AuditEventRow.occurred_at.desc()).limit(limit)

            if project_id:
                q = q.where(AuditEventRow.project_id == project_id)
            if task_id:
                q = q.where(AuditEventRow.task_id == task_id)
            if event_types:
                q = q.where(AuditEventRow.event_type.in_(event_types))
            if severity:
                q = q.where(AuditEventRow.severity == severity)
            if since:
                q = q.where(AuditEventRow.occurred_at >= since)
            if until:
                q = q.where(AuditEventRow.occurred_at <= until)

            rows = (await db.execute(q)).scalars().all()

        return [self._row_to_event(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Decision replay                                                     #
    # ------------------------------------------------------------------ #

    async def replay_task(self, task_id: str) -> TaskReplay:
        """
        Reconstruct the full decision chain for a task.

        Returns every audit event for the task in chronological order,
        plus a structured attempt-by-attempt breakdown with validation
        scores, failure reasons, and LLM timing.
        """
        events = await self.query(task_id=task_id, limit=500)
        events_asc = list(reversed(events))   # oldest first for replay

        project_id = events_asc[0].project_id if events_asc else None
        final_status: str | None = None

        # Walk events to build attempt records
        attempts: list[AttemptRecord] = []
        current: AttemptRecord | None = None

        for ev in events_asc:
            if ev.event_type == AuditEventType.AGENT_STARTED:
                attempt_num = ev.detail.get("attempt_number", len(attempts) + 1)
                current = AttemptRecord(
                    attempt_number=attempt_num,
                    agent_started_at=ev.occurred_at,
                    skills_loaded=ev.detail.get("skills", []),
                )

            elif ev.event_type == AuditEventType.AGENT_COMPLETED and current:
                current.agent_completed_at = ev.occurred_at
                current.llm_latency_ms = ev.detail.get("latency_ms")
                current.tokens_used    = ev.detail.get("tokens")

            elif ev.event_type == AuditEventType.VALIDATION_RESULT and current:
                current.validation_result = ev.detail.get("result")
                current.composite_score   = ev.detail.get("composite_score")
                current.failure_reason    = ev.detail.get("failure_reason")
                current.corrective_feedback = ev.detail.get("corrective_feedback")

            elif ev.event_type in (
                AuditEventType.TASK_APPROVED,
                AuditEventType.TASK_FAILED,
                AuditEventType.TASK_BLOCKED,
                AuditEventType.TASK_RETRY_QUEUED,
            ):
                if current:
                    attempts.append(current)
                    current = None
                final_status = ev.after_state.get("status")

        # Don't lose an open attempt (e.g. in-progress when replay requested)
        if current:
            attempts.append(current)

        return TaskReplay(
            task_id=task_id,
            project_id=project_id,
            events=events_asc,
            attempts=attempts,
            final_status=final_status,
        )

    # ------------------------------------------------------------------ #
    # Helper                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _row_to_event(row: AuditEventRow) -> AuditEvent:
        return AuditEvent(
            event_id=row.event_id,
            project_id=row.project_id,
            task_id=row.task_id,
            actor_type=row.actor_type,
            actor_id=row.actor_id,
            event_type=row.event_type,
            severity=row.severity,
            before_state=json.loads(row.before_json),
            after_state=json.loads(row.after_json),
            detail=json.loads(row.detail_json),
            occurred_at=row.occurred_at,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_service: AuditService | None = None


def get_audit_service() -> AuditService:
    global _service
    if _service is None:
        _service = AuditService()
    return _service


# ---------------------------------------------------------------------------
# Module-level convenience shorthands (import and call directly)
# ---------------------------------------------------------------------------


def audit(
    event_type: str,
    *,
    project_id:  str | None = None,
    task_id:     str | None = None,
    actor_type:  str = "engine",
    actor_id:    str = "",
    severity:    str = AuditSeverity.INFO,
    before:      dict[str, Any] | None = None,
    after:       dict[str, Any] | None = None,
    **detail: Any,
) -> None:
    """Single-line fire-and-forget audit emit for use anywhere in the codebase."""
    get_audit_service().emit(
        event_type,
        project_id=project_id,
        task_id=task_id,
        actor_type=actor_type,
        actor_id=actor_id,
        severity=severity,
        before=before,
        after=after,
        **detail,
    )
