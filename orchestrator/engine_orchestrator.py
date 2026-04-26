"""
Orchestration engine — the autonomous execution loop.

Phase 3 changes vs previous version:
  - _run_with_memory() now calls workspace_manager.get_workspace_context()
    before building the agent context.  This provides:
      (a) The workspace artifact shelf (upstream artifacts visible to this dept).
      (b) Department-scoped memory (filtered by DepartmentContextProfile).
  - _approve_task() adds the task's created artifacts to the workspace shelf
    and records the task ID in the workspace audit trail, then suspends the
    workspace.
  - _finalize_project() calls workspace_manager.archive_all_for_project(),
    clearing Tier-1 scratch pads (spec §6.1 ephemeral rule).
  - _run_with_memory() activates the workspace before task dispatch and the
    workspace is suspended after the task result is processed.

All prior bug fixes are retained:
  1. prior_attempt_summary persisted and re-read correctly.
  2. attempt_count incremented exactly once per outcome.
  3. _row_to_task / _row_to_project replaced with db.mappers imports.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import structlog
from sqlalchemy import select

from config import Settings
from db.engine import session_factory
from db.mappers import row_to_edge, row_to_project, row_to_task
from db.tables import ProjectRow, TaskEdgeRow, TaskRow
from models.project import Project, ProjectStatus
from models.task import Task, TaskEdge, TaskStatus
from orchestrator.agent_runner import (
    RunResult,
    build_corrective_feedback,
    build_prior_attempt_summary,
    run_agent_for_task,
)
from orchestrator.scheduler import (
    compute_ready_task_ids,
    detect_deadlock,
    is_project_complete,
    select_ready_tasks,
)
from orchestrator.audit import AuditEventType, AuditSeverity, audit
from orchestrator.learning import build_episode, get_learning_service
from orchestrator.checkpoint_manager import CheckpointTrigger, get_checkpoint_manager
from orchestrator.workspace_manager import get_workspace_manager

log = structlog.get_logger(__name__)

_MAX_ITERATIONS  = 500
_POLL_INTERVAL_S = 1.5


class OrchestrationEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings      = settings
        self._max_concurrent = settings.max_concurrent_tasks
        self._running: set[str] = set()
        self._iteration: int = 0
        self._approvals_since_checkpoint: int = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run_project(self, project_id: str) -> None:
        log.info("engine.start", project_id=project_id)
        await self._set_project_status(project_id, ProjectStatus.EXECUTION)
        audit(
            AuditEventType.PROJECT_STATUS_CHANGED,
            project_id=project_id,
            before={"status": "planning"},
            after={"status": "execution"},
        )

        for iteration in range(1, _MAX_ITERATIONS + 1):
            self._iteration = iteration
            tasks, edges, project = await self._load_graph(project_id)

            if not tasks:
                log.warning("engine.no_tasks", project_id=project_id)
                break

            # ── Dependency resolution ────────────────────────────────
            newly_ready = compute_ready_task_ids(tasks, edges)
            if newly_ready:
                await self._bulk_set_status(newly_ready, TaskStatus.READY)
                tasks, edges, project = await self._load_graph(project_id)

            # ── Completion check ─────────────────────────────────────
            if is_project_complete(tasks):
                log.info("engine.complete", project_id=project_id, iterations=iteration)
                # Checkpoint before finalising — last chance to capture full approved state
                await self._save_checkpoint(project_id, CheckpointTrigger.PHASE_COMPLETE)
                await self._finalize_project(project_id, tasks)
                return

            # ── Deadlock check ───────────────────────────────────────
            if detect_deadlock(tasks, self._running):
                log.error("engine.deadlock", project_id=project_id, iteration=iteration)
                await self._save_checkpoint(project_id, CheckpointTrigger.INSTABILITY)
                await self._set_project_status(project_id, ProjectStatus.REVIEW)
                return

            # ── Task dispatch ────────────────────────────────────────
            to_dispatch = select_ready_tasks(tasks, self._running, self._max_concurrent)

            if not to_dispatch:
                await asyncio.sleep(_POLL_INTERVAL_S)
                continue

            dispatch_ids = [t.task_id for t in to_dispatch]
            await self._mark_in_progress(dispatch_ids)
            self._running.update(dispatch_ids)

            log.info(
                "engine.dispatch",
                project_id=project_id,
                iteration=iteration,
                dispatching=dispatch_ids,
            )
            for _tid in dispatch_ids:
                audit(
                    AuditEventType.TASK_DISPATCHED,
                    project_id=project_id,
                    task_id=_tid,
                    before={"status": "ready"},
                    after={"status": "in_progress"},
                    iteration=iteration,
                )

            # ── Parallel execution ───────────────────────────────────
            results = await asyncio.gather(
                *[self._run_with_memory(task, project) for task in to_dispatch],
                return_exceptions=True,
            )

            # ── Result processing ────────────────────────────────────
            for task, result in zip(to_dispatch, results):
                self._running.discard(task.task_id)

                if isinstance(result, BaseException):
                    log.error(
                        "engine.task_exception",
                        task_id=task.task_id,
                        error=str(result),
                    )
                    await self._handle_failure(task, str(result))
                else:
                    await self._handle_result(task, result)

        else:
            log.error("engine.max_iterations", project_id=project_id)
            await self._set_project_status(project_id, ProjectStatus.REVIEW)

    # ------------------------------------------------------------------
    # Memory + workspace enriched task execution  (Phase 3)
    # ------------------------------------------------------------------

    async def _run_with_memory(self, task: Task, project: Project) -> RunResult:
        """
        Phase 3:
          1. Activate this department's workspace.
          2. Retrieve scoped memory via workspace_manager (department-filtered).
          3. Retrieve failure memory (unfiltered — all depts benefit).
          4. Build corrective feedback from the last attempt record.
          5. Run the agent.
        """
        wm = get_workspace_manager()
        workspace_id     = ""
        workspace_artifact_ids: list[str] = []

        # ── Workspace activation ──────────────────────────────────────
        try:
            ctx = await wm.get_workspace_context(
                project_id=task.project_id,
                department=task.owner_department.value,
                task_id=task.task_id,
                query=f"{task.title} {task.description[:200]}",
            )
            workspace_id           = ctx.workspace_id
            workspace_artifact_ids = ctx.artifact_ids
            relevant_memories      = ctx.scoped_memories

            await wm.activate(workspace_id)
        except Exception as exc:
            log.warning(
                "engine.workspace_init_failed",
                task_id=task.task_id,
                error=str(exc),
            )
            # Graceful degradation: fall back to unscoped retrieval
            relevant_memories = []
            try:
                from memory.manager import get_memory_manager
                mgr = get_memory_manager()
                query = f"{task.title} {task.description[:200]}"
                relevant_memories = await mgr.retrieve_for_task(
                    query, task.project_id, task.task_id
                )
            except Exception as mem_exc:
                log.warning(
                    "engine.memory_fallback_failed",
                    task_id=task.task_id,
                    error=str(mem_exc),
                )

        # ── Failure memory (not filtered by dept — everyone benefits) ─
        prior_failures: list = []
        try:
            from memory.manager import get_memory_manager
            mgr = get_memory_manager()
            prior_failures = await mgr.retrieve_failures(task.project_id)
        except Exception as exc:
            log.warning(
                "engine.failure_memory_failed",
                task_id=task.task_id,
                error=str(exc),
            )

        # ── Corrective feedback from last attempt ─────────────────────
        corrective_feedback    = ""
        prior_attempt_summary  = ""
        if task.attempts:
            last = task.attempts[-1]
            corrective_feedback   = last.corrective_feedback or ""
            prior_attempt_summary = getattr(last, "prior_attempt_summary", None) or ""

        # ── Agent execution ───────────────────────────────────────────
        result = await run_agent_for_task(
            task=task,
            project=project,
            relevant_memories=relevant_memories,
            prior_failures=prior_failures,
            corrective_feedback=corrective_feedback,
            prior_attempt_summary=prior_attempt_summary,
            workspace_artifact_ids=workspace_artifact_ids,
            human_notes=task.human_notes,   # Phase 6
        )

        # ── Post-run workspace update ─────────────────────────────────
        # Suspend workspace now regardless of outcome.
        # Artifact IDs are only added on approval (in _approve_task).
        if workspace_id:
            try:
                await wm.record_task(workspace_id, task.task_id)
                await wm.suspend(workspace_id)
            except Exception as exc:
                log.warning(
                    "engine.workspace_post_run_failed",
                    task_id=task.task_id,
                    error=str(exc),
                )

        return result

    # ------------------------------------------------------------------
    # Result handlers
    # ------------------------------------------------------------------

    async def _handle_result(self, task: Task, result: RunResult) -> None:
        if result.status == "approved":
            await self._approve_task(task, result)
        elif result.status == "blocked":
            await self._block_task(task, result.failure_reason)
        else:
            async with session_factory() as db:
                row = await db.get(TaskRow, task.task_id)
                db_attempt_count = row.attempt_count if row else task.attempt_count

            if db_attempt_count < task.budget.max_retries:
                feedback = build_corrective_feedback(result)
                summary  = build_prior_attempt_summary(result)
                await self._queue_retry(task, result, feedback, summary)
            else:
                await self._handle_failure(task, result.failure_reason)

    async def _approve_task(self, task: Task, result: RunResult) -> None:
        now = datetime.now(timezone.utc)

        async with session_factory() as db:
            row = await db.get(TaskRow, task.task_id)
            if not row:
                return
            row.attempt_count = (row.attempt_count or 0) + 1
            row.status        = TaskStatus.APPROVED.value
            row.completed_at  = now
            row.updated_at    = now

            attempts = json.loads(row.attempts_json)
            attempts.append({
                "attempt_number":  row.attempt_count,
                "agent_id":        result.output.agent_role if result.output else "unknown",
                "started_at":      now.isoformat(),
                "completed_at":    now.isoformat(),
                "validation_result": "pass",
                "composite_score": result.output.confidence if result.output else 1.0,
            })
            row.attempts_json = json.dumps(attempts)

        # ── Phase 3: push approved artifacts onto the workspace shelf ─
        if result.output and result.output.artifacts_created:
            try:
                wm  = get_workspace_manager()
                wsp = await wm.get_or_create(task.project_id, task.owner_department.value)
                await wm.add_artifact_ids(wsp.workspace_id, result.output.artifacts_created)
            except Exception as exc:
                log.warning(
                    "engine.workspace_artifact_update_failed",
                    task_id=task.task_id,
                    error=str(exc),
                )

        log.info("task.approved", task_id=task.task_id, latency_ms=result.latency_ms)
        audit(
            AuditEventType.TASK_APPROVED,
            project_id=task.project_id,
            task_id=task.task_id,
            before={"status": "in_progress"},
            after={"status": "approved"},
            latency_ms=result.latency_ms,
            composite_score=result.validation_report.composite_score if result.validation_report else None,
        )

        # Phase 9: continuous learning — fire on every approval
        try:
            episode = build_episode(task, result)
            svc = get_learning_service()
            asyncio.create_task(
                svc.on_task_approved(episode),
                name=f"learn_approved_{task.task_id}",
            )
            # If this was a retry success, also capture the retry pattern
            if task.attempt_count > 1:
                scores = [
                    a.composite_score for a in task.attempts
                    if a.composite_score is not None
                ]
                if len(scores) >= 2:
                    asyncio.create_task(
                        svc.on_retry_success(episode, scores),
                        name=f"learn_retry_{task.task_id}",
                    )
        except Exception as _exc:
            log.warning("engine.learning_hook_failed", task_id=task.task_id, error=str(_exc))

        # Checkpoint when a high-value artifact is approved (quality >= 0.85)
        if result.output and task.quality_threshold >= 0.85:
            self._approvals_since_checkpoint += 1
        elif self._approvals_since_checkpoint >= 3:
            # Also checkpoint after every 3 approvals regardless of threshold
            self._approvals_since_checkpoint = 0
            await self._save_checkpoint(task.project_id, CheckpointTrigger.ARTIFACT_APPROVED)

    async def _handle_failure(self, task: Task, reason: str) -> None:
        now = datetime.now(timezone.utc)
        async with session_factory() as db:
            row = await db.get(TaskRow, task.task_id)
            if row:
                row.attempt_count = (row.attempt_count or 0) + 1
                row.status        = TaskStatus.FAILED.value
                row.completed_at  = now
                row.updated_at    = now
        log.error("task.failed", task_id=task.task_id, reason=reason)
        audit(
            AuditEventType.TASK_FAILED,
            project_id=task.project_id,
            task_id=task.task_id,
            severity=AuditSeverity.ERROR,
            before={"status": "in_progress"},
            after={"status": "failed"},
            reason=reason,
        )

        # Phase 9: learn from permanent failures immediately
        try:
            from orchestrator.learning import TaskEpisode
            episode = TaskEpisode(
                task_id=task.task_id,
                project_id=task.project_id,
                task_type=task.type.value,
                department=task.owner_department.value,
                title=task.title,
                description=task.description[:200],
                skills_used=getattr(task, "required_skill_ids", []),
                attempt_count=task.attempt_count,
                final_status="failed",
                composite_score=None,
                quality_threshold=task.quality_threshold,
                failure_reasons=[reason],
                corrective_feedbacks=[],
                validation_issues=[],
                latency_ms=0,
                tokens_used=0,
            )
            asyncio.create_task(
                get_learning_service().on_task_failed(episode),
                name=f"learn_failed_{task.task_id}",
            )
        except Exception as _exc:
            log.warning("engine.learning_failure_hook_failed", task_id=task.task_id, error=str(_exc))

    async def _block_task(self, task: Task, reason: str) -> None:
        async with session_factory() as db:
            row = await db.get(TaskRow, task.task_id)
            if row:
                row.status     = TaskStatus.BLOCKED.value
                row.updated_at = datetime.now(timezone.utc)
        log.warning("task.blocked", task_id=task.task_id, reason=reason)
        audit(
            AuditEventType.TASK_BLOCKED,
            project_id=task.project_id,
            task_id=task.task_id,
            severity=AuditSeverity.WARNING,
            before={"status": "in_progress"},
            after={"status": "blocked"},
            reason=reason,
        )

    async def _queue_retry(
        self,
        task:          Task,
        result:        RunResult,
        feedback:      str,
        prior_summary: str,
    ) -> None:
        now = datetime.now(timezone.utc)

        async with session_factory() as db:
            row = await db.get(TaskRow, task.task_id)
            if not row:
                return

            row.attempt_count = (row.attempt_count or 0) + 1
            row.status        = TaskStatus.READY.value
            row.updated_at    = now

            attempts = json.loads(row.attempts_json)
            attempts.append({
                "attempt_number":    row.attempt_count,
                "agent_id":          result.output.agent_role if result.output else "unknown",
                "started_at":        now.isoformat(),
                "completed_at":      now.isoformat(),
                "validation_result": "fail",
                "composite_score":   result.output.confidence if result.output else 0.0,
                "failure_reason":    result.failure_reason,
                "corrective_feedback":   feedback,
                "prior_attempt_summary": prior_summary,
            })
            row.attempts_json = json.dumps(attempts)

        log.info(
            "task.retry_queued",
            task_id=task.task_id,
            attempt=row.attempt_count,
            max_retries=task.budget.max_retries,
        )
        audit(
            AuditEventType.TASK_RETRY_QUEUED,
            project_id=task.project_id,
            task_id=task.task_id,
            severity=AuditSeverity.WARNING,
            after={"status": "ready"},
            attempt=row.attempt_count,
            max_retries=task.budget.max_retries,
            failure_reason=result.failure_reason,
        )

    # ------------------------------------------------------------------
    # Project finalization  (Phase 3: archive all workspaces)
    # ------------------------------------------------------------------

    async def _save_checkpoint(
        self, project_id: str, trigger: str = CheckpointTrigger.MANUAL
    ) -> None:
        """Save a checkpoint non-fatally — engine continues on error."""
        try:
            ckp = await get_checkpoint_manager().save(
                project_id, trigger=trigger, iteration=self._iteration
            )
            log.info(
                "engine.checkpoint_saved",
                checkpoint_id=ckp.checkpoint_id,
                trigger=trigger,
                project_id=project_id,
            )
            audit(
                AuditEventType.CHECKPOINT_SAVED,
                project_id=project_id,
                trigger=trigger,
                checkpoint_id=ckp.checkpoint_id,
            )
        except Exception as exc:
            log.warning(
                "engine.checkpoint_failed",
                project_id=project_id,
                trigger=trigger,
                error=str(exc),
            )

    async def _finalize_project(self, project_id: str, tasks: list[Task]) -> None:
        failed_count  = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        blocked_count = sum(1 for t in tasks if t.status == TaskStatus.BLOCKED)

        # Phase 3: clear Tier-1 scratch pads (spec §6.1 ephemeral rule)
        try:
            await get_workspace_manager().archive_all_for_project(project_id)
        except Exception as exc:
            log.warning(
                "engine.workspace_archive_failed",
                project_id=project_id,
                error=str(exc),
            )

        await self._set_project_status(project_id, ProjectStatus.REVIEW)
        log.info(
            "engine.finalized",
            project_id=project_id,
            approved=sum(1 for t in tasks if t.status == TaskStatus.APPROVED),
            failed=failed_count,
            blocked=blocked_count,
        )

        # Phase 9: run full retrospective learning synthesis
        asyncio.create_task(
            get_learning_service().run_project_retrospective(project_id),
            name=f"retrospective_{project_id}",
        )

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    async def _load_graph(
        self, project_id: str
    ) -> tuple[list[Task], list[TaskEdge], Project]:
        async with session_factory() as db:
            proj_row = await db.get(ProjectRow, project_id)
            if not proj_row:
                raise ValueError(f"Project {project_id!r} not found.")
            project = row_to_project(proj_row)

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

        return (
            [row_to_task(r) for r in task_rows],
            [row_to_edge(r) for r in edge_rows],
            project,
        )

    async def _set_project_status(
        self, project_id: str, status: ProjectStatus
    ) -> None:
        async with session_factory() as db:
            row = await db.get(ProjectRow, project_id)
            if row:
                row.status     = status.value
                row.updated_at = datetime.now(timezone.utc)

    async def _bulk_set_status(
        self, task_ids: list[str], status: TaskStatus
    ) -> None:
        now = datetime.now(timezone.utc)
        async with session_factory() as db:
            for tid in task_ids:
                row = await db.get(TaskRow, tid)
                if row and row.status == TaskStatus.PLANNED.value:
                    row.status     = status.value
                    row.updated_at = now
        log.debug("tasks.unlocked", count=len(task_ids), new_status=status)

    async def _mark_in_progress(self, task_ids: list[str]) -> None:
        """Does NOT increment attempt_count (incremented on outcome only)."""
        now = datetime.now(timezone.utc)
        async with session_factory() as db:
            for tid in task_ids:
                row = await db.get(TaskRow, tid)
                if row:
                    row.status     = TaskStatus.IN_PROGRESS.value
                    row.started_at = row.started_at or now
                    row.updated_at = now
