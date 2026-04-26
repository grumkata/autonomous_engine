"""
orchestrator/checkpoint_manager.py — Checkpoint save, restore, and management (Phase 7).

Public API
----------
CheckpointManager.save(project_id, trigger, iteration)  → Checkpoint
CheckpointManager.restore(checkpoint_id)                → str  (project_id)
CheckpointManager.list_for_project(project_id)          → list[CheckpointSummary]
CheckpointManager.delete(checkpoint_id)                 → None
CheckpointManager.save_all_active(trigger)              → list[Checkpoint]
CheckpointManager.latest_for_project(project_id)        → Checkpoint | None

get_checkpoint_manager()                                → CheckpointManager (singleton)

How save() works
----------------
1. Load the full project graph from DB (project row, all task rows, all edge rows,
   all workspace rows for the project).
2. Serialise each row into its Pydantic domain model, then to dict.
3. Write a single CheckpointRow with the bundled snapshot_json.
4. Prune old checkpoints: keep only the last MAX_CHECKPOINTS_PER_PROJECT per project.

How restore() works
-------------------
1. Load the CheckpointRow; deserialise snapshot_json.
2. Deserialise task_states back into Task domain objects.
3. For each task: upsert (update if exists, insert if not) the TaskRow with the
   snapshot's field values.
4. Tasks that were IN_PROGRESS when the snapshot was taken are reset to READY
   (they were interrupted; the engine must retry them).
5. Workspace rows are upserted from workspace_states.
6. Project status is restored to EXECUTION so the engine loop re-starts.
7. Returns the project_id so the caller can re-launch run_project().

Spec §7 triggers
----------------
  phase_complete     — called by engine when project.status advances
  artifact_approved  — called by _approve_task when quality_threshold >= 0.85
  human_intervention — called by governance API endpoints
  instability        — called by validator when plateau is detected
  shutdown           — called by main.py lifespan teardown
  manual             — called via API endpoint
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import structlog
from pydantic import BaseModel, Field
from sqlalchemy import delete, select

from core.ids import prefixed_id
from db.engine import session_factory
from db.mappers import (
    checkpoint_to_row,
    row_to_checkpoint,
    row_to_edge,
    row_to_project,
    row_to_task,
    row_to_workspace,
)
from db.tables import (
    CheckpointRow,
    ProjectRow,
    TaskEdgeRow,
    TaskRow,
    WorkspaceRow,
)
from models.checkpoint import Checkpoint, CheckpointTrigger
from models.task import TaskStatus

log = structlog.get_logger(__name__)

# Keep at most this many checkpoints per project (oldest pruned first)
MAX_CHECKPOINTS_PER_PROJECT = 10


# ---------------------------------------------------------------------------
# Lightweight summary used by list endpoints
# ---------------------------------------------------------------------------


class CheckpointSummary(BaseModel):
    checkpoint_id:  str
    project_id:     str
    trigger_reason: str
    task_count:     int
    approved_count: int
    iteration:      int
    created_at:     datetime


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class CheckpointManager:

    # ------------------------------------------------------------------ #
    # Save                                                                #
    # ------------------------------------------------------------------ #

    async def save(
        self,
        project_id: str,
        trigger: str = CheckpointTrigger.MANUAL,
        iteration: int = 0,
    ) -> Checkpoint:
        """
        Snapshot the full project execution state to the DB.

        Automatically prunes old checkpoints (keeps MAX_CHECKPOINTS_PER_PROJECT).
        Non-fatal on failure — logs a warning and returns None so the engine
        can continue running.
        """
        async with session_factory() as db:
            # ── Load all rows ──────────────────────────────────────────
            proj_row = await db.get(ProjectRow, project_id)
            if not proj_row:
                raise ValueError(f"Project {project_id!r} not found for checkpointing.")

            task_rows = (
                await db.execute(
                    select(TaskRow).where(TaskRow.project_id == project_id)
                )
            ).scalars().all()

            edge_rows = (
                await db.execute(
                    select(TaskEdgeRow).where(TaskEdgeRow.project_id == project_id)
                )
            ).scalars().all()

            workspace_rows = (
                await db.execute(
                    select(WorkspaceRow).where(WorkspaceRow.project_id == project_id)
                )
            ).scalars().all()

        # ── Serialise ─────────────────────────────────────────────────
        project  = row_to_project(proj_row)
        tasks    = [row_to_task(r)     for r in task_rows]
        edges    = [row_to_edge(r)     for r in edge_rows]
        workspaces = [row_to_workspace(r) for r in workspace_rows]

        approved_count    = sum(1 for t in tasks if t.status == TaskStatus.APPROVED)
        in_progress_count = sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS)

        ckp = Checkpoint(
            project_id=project_id,
            trigger_reason=trigger,
            project_state=project.model_dump(mode="json"),
            task_states=[t.model_dump(mode="json") for t in tasks],
            edge_states=[e.model_dump(mode="json") for e in edges],
            # Exclude Tier-1 scratch_pad (ephemeral — spec §6.1)
            workspace_states=[
                {k: v for k, v in w.model_dump(mode="json").items() if k != "scratch_pad"}
                for w in workspaces
            ],
            task_count=len(tasks),
            approved_count=approved_count,
            in_progress_count=in_progress_count,
            iteration=iteration,
        )

        # ── Write row ─────────────────────────────────────────────────
        async with session_factory() as db:
            db.add(checkpoint_to_row(ckp))

        log.info(
            "checkpoint.saved",
            checkpoint_id=ckp.checkpoint_id,
            project_id=project_id,
            trigger=trigger,
            tasks=len(tasks),
            approved=approved_count,
        )

        # ── Prune old checkpoints ────────────────────────────────────
        await self._prune(project_id)

        return ckp

    # ------------------------------------------------------------------ #
    # Restore                                                             #
    # ------------------------------------------------------------------ #

    async def restore(self, checkpoint_id: str) -> str:
        """
        Restore a project to the state captured in this checkpoint.

        Returns the project_id so the caller can re-launch run_project().

        IN_PROGRESS tasks are reset to READY — they were interrupted and
        must be retried.  APPROVED / FAILED / BLOCKED tasks are left as-is.
        Tasks that no longer exist in the DB are inserted fresh.
        """
        async with session_factory() as db:
            row = await db.get(CheckpointRow, checkpoint_id)
            if not row:
                raise ValueError(f"Checkpoint {checkpoint_id!r} not found.")
            ckp = row_to_checkpoint(row)

        project_id = ckp.project_id
        log.info(
            "checkpoint.restoring",
            checkpoint_id=checkpoint_id,
            project_id=project_id,
            trigger=ckp.trigger_reason,
        )

        async with session_factory() as db:
            # ── Restore project row ───────────────────────────────────
            proj_row = await db.get(ProjectRow, project_id)
            if proj_row:
                ps = ckp.project_state
                proj_row.status      = "execution"   # re-enter execution loop
                proj_row.goals_json       = json.dumps(ps.get("goals", []), default=str)
                proj_row.constraints_json = json.dumps(ps.get("constraints", []), default=str)
                proj_row.deliverables_json = json.dumps(ps.get("deliverables", []), default=str)
                proj_row.tags_json        = json.dumps(ps.get("tags", []), default=str)
                proj_row.insight_ids_json = json.dumps(ps.get("insight_ids", []), default=str)
                proj_row.frozen      = False          # always unfreeze on restore
                proj_row.updated_at  = datetime.now(timezone.utc)

            # ── Restore task rows ─────────────────────────────────────
            for ts in ckp.task_states:
                task_id = ts["task_id"]
                # IN_PROGRESS at snapshot time = interrupted = must retry
                status = ts.get("status", "planned")
                if status == TaskStatus.IN_PROGRESS.value:
                    status = TaskStatus.READY.value

                existing = await db.get(TaskRow, task_id)
                if existing:
                    existing.status        = status
                    existing.attempt_count = ts.get("attempt_count", 0)
                    existing.attempts_json = json.dumps(
                        ts.get("attempts", []), default=str
                    )
                    existing.updated_at    = datetime.now(timezone.utc)
                    # Reset completed_at for tasks that were in-progress
                    if status == TaskStatus.READY.value:
                        existing.completed_at = None
                else:
                    # Task was deleted after the snapshot — re-insert it
                    new_row = TaskRow(
                        task_id=task_id,
                        project_id=project_id,
                        type=ts.get("type", "research"),
                        title=ts.get("title", ""),
                        description=ts.get("description", ""),
                        status=status,
                        priority=ts.get("priority", "medium"),
                        owner_department=ts.get("owner_department", "research"),
                        parent_task_id=ts.get("parent_task_id"),
                        depth_level=ts.get("depth_level", 0),
                        validation_criteria_json=json.dumps(
                            ts.get("validation_criteria", [])
                        ),
                        expected_output_types_json=json.dumps(
                            ts.get("expected_output_types", [])
                        ),
                        input_artifact_ids_json=json.dumps(
                            ts.get("input_artifact_ids", [])
                        ),
                        output_artifact_ids_json=json.dumps(
                            ts.get("output_artifact_ids", [])
                        ),
                        assigned_agent_ids_json=json.dumps(
                            ts.get("assigned_agent_ids", [])
                        ),
                        budget_json=json.dumps(ts.get("budget", {}), default=str),
                        attempts_json=json.dumps(
                            ts.get("attempts", []), default=str
                        ),
                        required_skill_ids_json=json.dumps(
                            ts.get("required_skill_ids", [])
                        ),
                        human_notes_json=json.dumps(
                            ts.get("human_notes", [])
                        ),
                        attempt_count=ts.get("attempt_count", 0),
                        quality_threshold=ts.get("quality_threshold", 0.75),
                        requires_human_approval=ts.get("requires_human_approval", False),
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                    db.add(new_row)

            # ── Restore workspace rows ────────────────────────────────
            for ws in ckp.workspace_states:
                wid = ws.get("workspace_id", "")
                if not wid:
                    continue
                existing_ws = await db.get(WorkspaceRow, wid)
                if existing_ws:
                    existing_ws.artifact_ids_json = json.dumps(
                        ws.get("artifact_ids", [])
                    )
                    existing_ws.task_ids_json = json.dumps(
                        ws.get("task_ids_processed", [])
                    )
                    # Reset active workspaces to suspended so they accept new tasks
                    if existing_ws.status == "active":
                        existing_ws.status = "suspended"
                    existing_ws.scratch_pad = ""   # always clear Tier-1 ephemeral
                    existing_ws.updated_at  = datetime.now(timezone.utc)

        log.info(
            "checkpoint.restored",
            checkpoint_id=checkpoint_id,
            project_id=project_id,
            tasks_restored=len(ckp.task_states),
        )
        return project_id

    # ------------------------------------------------------------------ #
    # List / read                                                         #
    # ------------------------------------------------------------------ #

    async def list_for_project(self, project_id: str) -> list[CheckpointSummary]:
        """Return summaries for a project, newest first."""
        async with session_factory() as db:
            result = await db.execute(
                select(CheckpointRow)
                .where(CheckpointRow.project_id == project_id)
                .order_by(CheckpointRow.created_at.desc())
            )
            rows = result.scalars().all()

        return [
            CheckpointSummary(
                checkpoint_id=r.checkpoint_id,
                project_id=r.project_id,
                trigger_reason=r.trigger_reason,
                task_count=r.task_count,
                approved_count=r.approved_count,
                iteration=r.iteration,
                created_at=r.created_at,
            )
            for r in rows
        ]

    async def latest_for_project(self, project_id: str) -> Checkpoint | None:
        """Return the most recent checkpoint for a project, or None."""
        async with session_factory() as db:
            result = await db.execute(
                select(CheckpointRow)
                .where(CheckpointRow.project_id == project_id)
                .order_by(CheckpointRow.created_at.desc())
                .limit(1)
            )
            row = result.scalars().first()
            return row_to_checkpoint(row) if row else None

    async def get(self, checkpoint_id: str) -> Checkpoint | None:
        async with session_factory() as db:
            row = await db.get(CheckpointRow, checkpoint_id)
            return row_to_checkpoint(row) if row else None

    # ------------------------------------------------------------------ #
    # Delete                                                              #
    # ------------------------------------------------------------------ #

    async def delete(self, checkpoint_id: str) -> None:
        async with session_factory() as db:
            row = await db.get(CheckpointRow, checkpoint_id)
            if not row:
                raise ValueError(f"Checkpoint {checkpoint_id!r} not found.")
            await db.delete(row)
        log.info("checkpoint.deleted", checkpoint_id=checkpoint_id)

    # ------------------------------------------------------------------ #
    # Bulk shutdown save                                                  #
    # ------------------------------------------------------------------ #

    async def save_all_active(
        self, trigger: str = CheckpointTrigger.SHUTDOWN
    ) -> list[Checkpoint]:
        """
        Save a checkpoint for every project currently in EXECUTION status.
        Called from main.py lifespan teardown on graceful shutdown.
        """
        async with session_factory() as db:
            result = await db.execute(
                select(ProjectRow).where(ProjectRow.status == "execution")
            )
            active_projects = result.scalars().all()

        saved: list[Checkpoint] = []
        for proj in active_projects:
            try:
                ckp = await self.save(proj.project_id, trigger=trigger)
                saved.append(ckp)
            except Exception as exc:
                log.error(
                    "checkpoint.shutdown_save_failed",
                    project_id=proj.project_id,
                    error=str(exc),
                )

        log.info(
            "checkpoint.shutdown_complete",
            saved=len(saved),
            projects=len(active_projects),
        )
        return saved

    # ------------------------------------------------------------------ #
    # Internal: prune old checkpoints                                     #
    # ------------------------------------------------------------------ #

    async def _prune(self, project_id: str) -> None:
        """Delete oldest checkpoints beyond MAX_CHECKPOINTS_PER_PROJECT."""
        async with session_factory() as db:
            result = await db.execute(
                select(CheckpointRow)
                .where(CheckpointRow.project_id == project_id)
                .order_by(CheckpointRow.created_at.desc())
            )
            rows = result.scalars().all()

        if len(rows) <= MAX_CHECKPOINTS_PER_PROJECT:
            return

        to_delete = rows[MAX_CHECKPOINTS_PER_PROJECT:]
        async with session_factory() as db:
            for row in to_delete:
                await db.execute(
                    delete(CheckpointRow).where(
                        CheckpointRow.checkpoint_id == row.checkpoint_id
                    )
                )

        log.debug(
            "checkpoint.pruned",
            project_id=project_id,
            deleted=len(to_delete),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: CheckpointManager | None = None


def get_checkpoint_manager() -> CheckpointManager:
    global _instance
    if _instance is None:
        _instance = CheckpointManager()
    return _instance
