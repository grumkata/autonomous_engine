"""
orchestrator/workspace_manager.py — Agent Workspace lifecycle manager (Phase 3).

This is the primary new component of Phase 3.  It owns every workspace
row in the DB and is the only layer that may create, mutate, or archive
workspace state.

Public API
----------
get_or_create(project_id, department)        → Workspace
activate(workspace_id)                       → None
suspend(workspace_id)                        → None
complete(workspace_id)                       → None
archive(workspace_id)                        → None  # clears scratch_pad
archive_all_for_project(project_id)          → None
add_artifact_ids(workspace_id, ids)          → None
record_task(workspace_id, task_id)           → None
get_scoped_memory(project_id, department,    → list[MemoryExcerpt]
                  query, task_id)
get_workspace_context(project_id,            → WorkspaceContext
                      department, task_id,
                      query)

Spec §8 contracts enforced here
--------------------------------
* One workspace per (project_id, department) — get_or_create is idempotent.
* Workspace state is always reloaded from DB at task start (no in-process cache).
* scratch_pad is cleared on archive (Tier-1 ephemeral — spec §6.1).
* Cross-workspace communication happens only through artifact_ids passed into
  the input bundle — agents never read another workspace's raw state.
* Scoped memory retrieval filters by DepartmentContextProfile.memory_categories_include
  so each department sees only its appropriate memory slice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog
from sqlalchemy import select

from core.ids import prefixed_id
from db.engine import session_factory
from db.mappers import row_to_workspace, workspace_to_row
from db.tables import WorkspaceRow
from llm.department_profiles import get_department_profile
from llm.schemas import MemoryExcerpt
from models.workspace import Workspace, WorkspaceStatus

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# WorkspaceContext — returned to the orchestrator / agent runner
# ---------------------------------------------------------------------------


@dataclass
class WorkspaceContext:
    """
    Everything a workspace contributes to an agent's context bundle.

    workspace_id        — identifies this workspace in logs and DB
    artifact_ids        — the artifact shelf: IDs visible to this agent
    scoped_memories     — department-filtered MemoryExcerpts ready to inject
    """
    workspace_id:    str
    artifact_ids:    list[str]              = field(default_factory=list)
    scoped_memories: list[MemoryExcerpt]    = field(default_factory=list)


# ---------------------------------------------------------------------------
# WorkspaceManager
# ---------------------------------------------------------------------------


class WorkspaceManager:
    """
    Singleton service that manages workspace lifecycle and scoped memory.

    All methods are async-safe and re-read state from DB on every call
    so the engine can run multiple workers without cache-coherency issues.
    """

    # ------------------------------------------------------------------ #
    # Core lifecycle                                                       #
    # ------------------------------------------------------------------ #

    async def get_or_create(self, project_id: str, department: str) -> Workspace:
        """
        Return the existing workspace for (project_id, department), or create one.

        Idempotent — safe to call before every task dispatch.
        """
        async with session_factory() as db:
            result = await db.execute(
                select(WorkspaceRow).where(
                    WorkspaceRow.project_id == project_id,
                    WorkspaceRow.department == department,
                )
            )
            row = result.scalars().first()

            if row:
                workspace = row_to_workspace(row)
                log.debug(
                    "workspace.found",
                    workspace_id=workspace.workspace_id,
                    department=department,
                    status=workspace.status.value,
                )
                return workspace

            # Create new workspace
            workspace = Workspace(
                project_id=project_id,
                department=department,
                status=WorkspaceStatus.CREATED,
            )
            new_row = workspace_to_row(workspace)
            db.add(new_row)
            await db.flush()

        log.info(
            "workspace.created",
            workspace_id=workspace.workspace_id,
            project_id=project_id,
            department=department,
        )
        return workspace

    async def activate(self, workspace_id: str) -> None:
        """Mark workspace ACTIVE when a task is dispatched into it."""
        await self._set_status(workspace_id, WorkspaceStatus.ACTIVE)
        log.debug("workspace.activated", workspace_id=workspace_id)

    async def suspend(self, workspace_id: str) -> None:
        """
        Suspend after a task completes.
        State is preserved; the workspace is ready for the next task in
        this department.
        """
        await self._set_status(workspace_id, WorkspaceStatus.SUSPENDED)
        log.debug("workspace.suspended", workspace_id=workspace_id)

    async def complete(self, workspace_id: str) -> None:
        """
        Mark COMPLETED when all department tasks for the project are terminal.
        The workspace is still readable but will not accept new tasks.
        """
        await self._set_status(workspace_id, WorkspaceStatus.COMPLETED)
        log.info("workspace.completed", workspace_id=workspace_id)

    async def archive(self, workspace_id: str) -> None:
        """
        Archive and clear Tier-1 ephemeral scratch_pad (spec §6.1).
        Called at project finalisation.
        """
        now = datetime.now(timezone.utc)
        async with session_factory() as db:
            row = await db.get(WorkspaceRow, workspace_id)
            if row:
                row.status      = WorkspaceStatus.ARCHIVED.value
                row.scratch_pad = ""     # Tier-1 ephemeral — must not persist
                row.archived_at = now
                row.updated_at  = now

        log.info("workspace.archived", workspace_id=workspace_id)

    async def archive_all_for_project(self, project_id: str) -> None:
        """
        Archive every workspace for a project.
        Called from engine_orchestrator._finalize_project().
        """
        async with session_factory() as db:
            result = await db.execute(
                select(WorkspaceRow).where(WorkspaceRow.project_id == project_id)
            )
            rows = result.scalars().all()
            now = datetime.now(timezone.utc)
            for row in rows:
                if row.status not in (
                    WorkspaceStatus.ARCHIVED.value,
                    WorkspaceStatus.COMPLETED.value,
                ):
                    row.status      = WorkspaceStatus.ARCHIVED.value
                    row.scratch_pad = ""
                    row.archived_at = now
                    row.updated_at  = now

        log.info(
            "workspace.archived_all",
            project_id=project_id,
            count=len(rows),
        )

    # ------------------------------------------------------------------ #
    # Artifact shelf                                                       #
    # ------------------------------------------------------------------ #

    async def add_artifact_ids(self, workspace_id: str, artifact_ids: list[str]) -> None:
        """
        Add artifact IDs to a workspace's shelf.

        Called by the orchestrator after a task is approved so downstream
        tasks in the same department can reference those artifacts.
        De-duplicates automatically.
        """
        if not artifact_ids:
            return

        async with session_factory() as db:
            row = await db.get(WorkspaceRow, workspace_id)
            if not row:
                log.warning("workspace.add_artifacts.not_found", workspace_id=workspace_id)
                return
            existing: list[str] = json.loads(row.artifact_ids_json)
            merged = list(dict.fromkeys(existing + artifact_ids))   # preserve order, dedupe
            row.artifact_ids_json = json.dumps(merged)
            row.updated_at = datetime.now(timezone.utc)

        log.debug(
            "workspace.artifacts_added",
            workspace_id=workspace_id,
            added=len(artifact_ids),
        )

    # ------------------------------------------------------------------ #
    # Task audit trail                                                     #
    # ------------------------------------------------------------------ #

    async def record_task(self, workspace_id: str, task_id: str) -> None:
        """
        Append task_id to the workspace's processing history.
        Idempotent — silently skips if already present.
        """
        async with session_factory() as db:
            row = await db.get(WorkspaceRow, workspace_id)
            if not row:
                return
            task_ids: list[str] = json.loads(row.task_ids_json)
            if task_id not in task_ids:
                task_ids.append(task_id)
                row.task_ids_json = json.dumps(task_ids)
                row.updated_at    = datetime.now(timezone.utc)

    # ------------------------------------------------------------------ #
    # Scoped memory retrieval  (spec §8 isolation)                        #
    # ------------------------------------------------------------------ #

    async def get_scoped_memory(
        self,
        project_id: str,
        department:  str,
        query:       str,
        task_id:     str,
        top_k:       int = 8,   # maps to n_results in MemoryManager
    ) -> list[MemoryExcerpt]:
        """
        Retrieve memory items relevant to `query`, filtered to the
        categories this department is allowed to see (spec §8 isolation).

        Falls back gracefully if the memory manager is unavailable.

        DepartmentContextProfile.memory_categories_include:
            frozenset()   → no filter (red_team sees everything)
            frozenset({…}) → only these categories returned
        """
        profile = get_department_profile(department)
        allowed_categories: frozenset[str] = profile.memory_categories_include

        try:
            from memory.manager import get_memory_manager
            mgr = get_memory_manager()
            raw_memories: list[MemoryExcerpt] = await mgr.retrieve_for_task(
                query, project_id, task_id, n_results=top_k
            )
        except Exception as exc:
            log.warning(
                "workspace.memory_retrieval_failed",
                department=department,
                project_id=project_id,
                error=str(exc),
            )
            return []

        # Apply department-level category filter
        if allowed_categories:
            raw_memories = [
                m for m in raw_memories
                if m.category in allowed_categories
            ]

        log.debug(
            "workspace.scoped_memory",
            department=department,
            total=len(raw_memories),
            filter_active=bool(allowed_categories),
        )
        return raw_memories

    # ------------------------------------------------------------------ #
    # Convenience bundle builder                                           #
    # ------------------------------------------------------------------ #

    async def get_workspace_context(
        self,
        project_id: str,
        department:  str,
        task_id:     str,
        query:       str,
    ) -> WorkspaceContext:
        """
        One-stop call: returns workspace artifact shelf + scoped memories.

        Called from engine_orchestrator._run_with_memory() before
        building the AgentInputBundle.
        """
        workspace = await self.get_or_create(project_id, department)
        scoped = await self.get_scoped_memory(
            project_id=project_id,
            department=department,
            query=query,
            task_id=task_id,
        )
        return WorkspaceContext(
            workspace_id=workspace.workspace_id,
            artifact_ids=workspace.artifact_ids,
            scoped_memories=scoped,
        )

    # ------------------------------------------------------------------ #
    # Read helpers                                                         #
    # ------------------------------------------------------------------ #

    async def list_for_project(self, project_id: str) -> list[Workspace]:
        """Return all workspaces for a project (for UI / governance API)."""
        async with session_factory() as db:
            result = await db.execute(
                select(WorkspaceRow)
                .where(WorkspaceRow.project_id == project_id)
                .order_by(WorkspaceRow.created_at)
            )
            return [row_to_workspace(r) for r in result.scalars().all()]

    async def get(self, workspace_id: str) -> Workspace | None:
        async with session_factory() as db:
            row = await db.get(WorkspaceRow, workspace_id)
            return row_to_workspace(row) if row else None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    async def _set_status(self, workspace_id: str, status: WorkspaceStatus) -> None:
        async with session_factory() as db:
            row = await db.get(WorkspaceRow, workspace_id)
            if row:
                row.status     = status.value
                row.updated_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: WorkspaceManager | None = None


def get_workspace_manager() -> WorkspaceManager:
    """Return the module-level singleton. Created on first call."""
    global _instance
    if _instance is None:
        _instance = WorkspaceManager()
    return _instance
