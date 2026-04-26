"""
models/workspace.py — Agent Workspace domain model (Phase 3).

A workspace is the bounded, isolated environment in which a single
department does its work across all tasks assigned to it within a project.

Lifecycle:
    CREATED   → workspace row exists, no tasks dispatched yet
    ACTIVE    → at least one task is currently in_progress in this workspace
    SUSPENDED → current task completed; workspace preserved for next task
    COMPLETED → all department tasks reached terminal state
    ARCHIVED  → workspace closed; scratch_pad cleared; read-only

Spec §8 rules enforced here:
  - One workspace per (project, department) pair.
  - artifact_ids is the "artifact shelf" — only IDs explicitly added by the
    orchestrator are visible to this workspace.
  - scratch_pad is Tier-1 ephemeral content. WorkspaceManager clears it
    on archive() (spec §6 Tier 1 — never persisted long-term).
  - task_ids_processed tracks which tasks have run in this workspace so
    the orchestrator can decide when the workspace is complete.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field

from core.ids import prefixed_id


class WorkspaceStatus(StrEnum):
    CREATED   = "created"
    ACTIVE    = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    ARCHIVED  = "archived"


class Workspace(BaseModel):
    workspace_id: str = Field(default_factory=lambda: prefixed_id("wsp"))
    project_id:   str
    department:   str   # DepartmentOwner.value — kept as str to avoid circular import

    status: WorkspaceStatus = WorkspaceStatus.CREATED

    # Artifact shelf: IDs of approved artifacts passed in from upstream tasks.
    # Agents in this workspace can reference these but do not see other
    # workspaces' raw state (spec §8 isolation rule).
    artifact_ids: list[str] = Field(default_factory=list)

    # Audit trail of tasks that have run inside this workspace.
    task_ids_processed: list[str] = Field(default_factory=list)

    # Tier-1 ephemeral scratch space.
    # Populated during task execution; cleared when the workspace is archived.
    scratch_pad: str = ""

    created_at:  datetime       = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:  datetime       = Field(default_factory=lambda: datetime.now(timezone.utc))
    archived_at: datetime | None = None

    def is_terminal(self) -> bool:
        return self.status in (WorkspaceStatus.COMPLETED, WorkspaceStatus.ARCHIVED)

    def can_accept_task(self) -> bool:
        """True when the workspace is in a state that allows dispatching a task."""
        return self.status in (
            WorkspaceStatus.CREATED,
            WorkspaceStatus.SUSPENDED,
            WorkspaceStatus.ACTIVE,   # re-entrant if parallel tasks in same dept
        )
