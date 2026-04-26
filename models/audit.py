"""
models/audit.py — Audit event domain model (Phase 8).

Every significant engine action emits an AuditEvent.  Events are the
persisted record of what happened, when, why, and who caused it.  Unlike
structlog output (console / stdout only), audit events survive restarts,
are queryable, and power the timeline view and decision replay in the UI.

Event types (spec §17)
----------------------
Project lifecycle:
    project_created          project_status_changed
    project_frozen           project_resumed
    project_redirected

Task lifecycle:
    task_generated           task_ready
    task_dispatched          task_completed
    task_approved            task_failed
    task_blocked             task_retry_queued

Agent execution:
    agent_started            agent_completed
    agent_llm_error          agent_parse_error
    agent_skill_loaded

Validation:
    validation_result        validation_escalated

Memory:
    memory_candidate_submitted   memory_accepted
    memory_rejected              memory_promoted
    memory_locked                memory_retired

Governance (human):
    human_note_injected      human_veto
    human_approval           human_policy_change
    human_freeze             human_resume
    human_redirect

Checkpoints:
    checkpoint_saved         checkpoint_restored

Severity
--------
    debug    — routine operations (task_ready, agent_started)
    info     — significant state changes (task_approved, checkpoint_saved)
    warning  — degraded operation (agent_llm_error, validation_escalated)
    error    — failures (task_failed, engine_deadlock)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from core.ids import prefixed_id


# ---------------------------------------------------------------------------
# Event type constants  (avoids magic strings throughout the codebase)
# ---------------------------------------------------------------------------


class AuditEventType:
    # Project
    PROJECT_CREATED         = "project_created"
    PROJECT_STATUS_CHANGED  = "project_status_changed"
    PROJECT_FROZEN          = "project_frozen"
    PROJECT_RESUMED         = "project_resumed"
    PROJECT_REDIRECTED      = "project_redirected"

    # Task
    TASK_GENERATED          = "task_generated"
    TASK_READY              = "task_ready"
    TASK_DISPATCHED         = "task_dispatched"
    TASK_COMPLETED          = "task_completed"
    TASK_APPROVED           = "task_approved"
    TASK_FAILED             = "task_failed"
    TASK_BLOCKED            = "task_blocked"
    TASK_RETRY_QUEUED       = "task_retry_queued"

    # Agent
    AGENT_STARTED           = "agent_started"
    AGENT_COMPLETED         = "agent_completed"
    AGENT_LLM_ERROR         = "agent_llm_error"
    AGENT_PARSE_ERROR       = "agent_parse_error"
    AGENT_SKILL_LOADED      = "agent_skill_loaded"

    # Validation
    VALIDATION_RESULT       = "validation_result"
    VALIDATION_ESCALATED    = "validation_escalated"

    # Memory
    MEMORY_CANDIDATE_SUBMITTED = "memory_candidate_submitted"
    MEMORY_ACCEPTED         = "memory_accepted"
    MEMORY_REJECTED         = "memory_rejected"
    MEMORY_PROMOTED         = "memory_promoted"
    MEMORY_LOCKED           = "memory_locked"
    MEMORY_RETIRED          = "memory_retired"

    # Governance
    HUMAN_NOTE_INJECTED     = "human_note_injected"
    HUMAN_VETO              = "human_veto"
    HUMAN_APPROVAL          = "human_approval"
    HUMAN_POLICY_CHANGE     = "human_policy_change"
    HUMAN_FREEZE            = "human_freeze"
    HUMAN_RESUME            = "human_resume"
    HUMAN_REDIRECT          = "human_redirect"

    # Checkpoints
    CHECKPOINT_SAVED        = "checkpoint_saved"
    CHECKPOINT_RESTORED     = "checkpoint_restored"

    # Skills / workspaces
    SKILL_LOADED            = "skill_loaded"
    WORKSPACE_CREATED       = "workspace_created"
    WORKSPACE_ARCHIVED      = "workspace_archived"


class AuditSeverity:
    DEBUG   = "debug"
    INFO    = "info"
    WARNING = "warning"
    ERROR   = "error"


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


class AuditEvent(BaseModel):
    event_id:    str = Field(default_factory=lambda: prefixed_id("evt"))
    project_id:  str | None = None
    task_id:     str | None = None

    # Who caused this (engine, agent, human, system)
    actor_type:  str = "engine"    # "engine" | "agent" | "human" | "system"
    actor_id:    str = ""          # dept name, user id, etc.

    event_type:  str               # AuditEventType constant
    severity:    str = AuditSeverity.INFO

    # What changed — both optional; present when meaningful
    before_state: dict[str, Any] = Field(default_factory=dict)
    after_state:  dict[str, Any] = Field(default_factory=dict)

    # Free-form detail bag for event-specific metadata
    detail:       dict[str, Any] = Field(default_factory=dict)

    occurred_at:  datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
