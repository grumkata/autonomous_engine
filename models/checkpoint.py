"""
models/checkpoint.py — Checkpoint domain model (Phase 7).

A checkpoint is a complete, self-contained snapshot of a project's execution
state at a specific point in time.  It contains enough information to restore
the engine to exactly that state after any kind of crash or restart.

What is stored
--------------
project_state   — serialised Project (goals, constraints, budget, status)
task_states     — list of serialised Task objects (status, attempt_count,
                  attempts_json, etc. for every task in the graph)
edge_states     — list of serialised TaskEdge objects
workspace_states — list of serialised Workspace objects (artifact shelf,
                   task audit trail — NOT scratch_pad, that is Tier-1 ephemeral)

What is NOT stored
------------------
- In-flight LLM calls (they must be retried on resume)
- Tier-1 scratch pad content (ephemeral by spec §6.1)
- ChromaDB vector embeddings (those live in the vector store and survive restarts)
- SQLite/Postgres rows (the DB IS the source of truth; checkpoints are a
  point-in-time snapshot so we can roll back, not a DB backup)

Trigger reasons (spec §7 Phase 7)
----------------------------------
  phase_complete    — project advanced to a new lifecycle phase
  artifact_approved — a high-value artifact was approved (quality_threshold >= 0.85)
  human_intervention — operator used a governance endpoint
  instability       — validator detected score plateau or repeated failures
  shutdown          — graceful server shutdown
  manual            — operator explicitly requested via API
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from core.ids import prefixed_id


class CheckpointTrigger:
    PHASE_COMPLETE    = "phase_complete"
    ARTIFACT_APPROVED = "artifact_approved"
    HUMAN_INTERVENTION = "human_intervention"
    INSTABILITY       = "instability"
    SHUTDOWN          = "shutdown"
    MANUAL            = "manual"


class Checkpoint(BaseModel):
    checkpoint_id:  str = Field(default_factory=lambda: prefixed_id("ckp"))
    project_id:     str
    trigger_reason: str = CheckpointTrigger.MANUAL

    # Serialised domain objects — stored as dicts, not raw JSON strings,
    # so callers work with Python objects rather than double-encoded strings.
    project_state:    dict[str, Any]       = Field(default_factory=dict)
    task_states:      list[dict[str, Any]] = Field(default_factory=list)
    edge_states:      list[dict[str, Any]] = Field(default_factory=list)
    workspace_states: list[dict[str, Any]] = Field(default_factory=list)

    # Snapshot metadata
    task_count:        int = 0
    approved_count:    int = 0
    in_progress_count: int = 0
    iteration:         int = 0   # which orchestration iteration triggered this

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def summary(self) -> str:
        return (
            f"Checkpoint {self.checkpoint_id} @ {self.created_at.isoformat()} "
            f"({self.trigger_reason}) — "
            f"{self.task_count} tasks, {self.approved_count} approved"
        )
