"""
orchestrator/agent_runner.py — Agent execution + validation.

Phase 3 change: run_agent_for_task() now accepts workspace_artifact_ids
from the WorkspaceManager and injects them into the AgentInputBundle as
input_artifact_ids so the agent knows which upstream artifacts are on its
department's shelf (spec §8 artifact shelf).

All prior Phase 4 (validation) changes are retained:
  - ValidationReport replaces the simple confidence gate.
  - Corrective feedback comes from ValidationReport.to_corrective_feedback().
  - build_corrective_feedback() and build_prior_attempt_summary() delegate
    to the report when available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from llm.client import get_client
from llm.prompts import PromptBuilder
from llm.schemas import AgentInputBundle, AgentOutput, MemoryExcerpt
from models.project import Project
from models.task import Task
from orchestrator.audit import AuditEventType, AuditSeverity, audit
from orchestrator.skill_registry import get_skill_registry
from orchestrator.validator import (
    ValidationReport,
    ValidationResult,
    clear_iteration_history,
    validate_and_build_feedback,
)

log = structlog.get_logger(__name__)

_builder = PromptBuilder()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    task_id:   str
    status:    str                          # "approved" | "failed" | "blocked"
    output:    AgentOutput | None = None
    validation_report: ValidationReport | None = None
    failure_reason:    str = ""
    latency_ms:        int = 0
    memory_ids_created: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status == "approved"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_agent_for_task(
    task:                    Task,
    project:                 Project,
    relevant_memories:       list[MemoryExcerpt] | None = None,
    prior_failures:          list[MemoryExcerpt] | None = None,
    corrective_feedback:     str = "",
    prior_attempt_summary:   str = "",
    use_llm_validation:      bool = False,
    # Phase 3: artifact IDs from the department workspace shelf
    workspace_artifact_ids:  list[str] | None = None,
    # Phase 6: human operator notes
    human_notes:             list[str] | None = None,
) -> RunResult:
    """
    Execute one task with its assigned department agent.

    workspace_artifact_ids (Phase 3):
        IDs of artifacts already on this department's workspace shelf.
        Injected into AgentInputBundle.input_artifact_ids so the agent
        knows what upstream approved outputs are available to reference.

    Disposition flow:
        output.status == "blocked"            → RunResult(status="blocked")
        ValidationReport.result == PASS       → RunResult(status="approved")
        ValidationReport.result == REVISE     → RunResult(status="failed") — engine retries
        ValidationReport.result == FAIL       → RunResult(status="failed") — permanent
    """
    t_start = time.monotonic()

    primary_goal     = project.primary_goal()
    hard_constraints = [c.description for c in project.hard_constraints()]

    # Merge workspace artifact IDs with any already on the task
    combined_input_artifacts = list(
        dict.fromkeys(
            (task.input_artifact_ids or []) + (workspace_artifact_ids or [])
        )
    )

    # Phase 4: load skill content from registry
    skill_contents: list[str] = []
    if task.required_skill_ids:
        try:
            skill_contents = await get_skill_registry().load_content_for_task(
                task.required_skill_ids
            )
        except Exception as exc:
            log.warning("agent.skill_load_failed", task_id=task.task_id, error=str(exc))

    bundle = AgentInputBundle(
        agent_role=task.owner_department.value,
        task_id=task.task_id,
        project_id=task.project_id,
        task_objective=task.title,
        task_description=task.description,
        project_title=project.title,
        primary_goal=primary_goal.statement if primary_goal else "",
        hard_constraints=hard_constraints,
        relevant_memories=relevant_memories or [],
        prior_failures=prior_failures or [],
        validation_criteria=task.validation_criteria,
        expected_output_types=task.expected_output_types,
        quality_threshold=task.quality_threshold,
        attempt_number=task.attempt_count + 1,
        corrective_feedback=corrective_feedback,
        prior_attempt_summary=prior_attempt_summary,
        max_tokens=task.budget.max_tokens,
        max_runtime_seconds=task.budget.max_runtime_seconds,
        # Phase 3: workspace artifact shelf → input_artifact_ids
        input_artifact_ids=combined_input_artifacts,
        skill_contents=skill_contents,
        human_notes=human_notes or task.human_notes,
    )

    messages = _builder.build_messages(bundle)
    client   = get_client()

    audit(
        AuditEventType.AGENT_STARTED,
        project_id=task.project_id,
        task_id=task.task_id,
        actor_type="agent",
        actor_id=task.owner_department.value,
        severity=AuditSeverity.DEBUG,
        attempt_number=bundle.attempt_number,
        skills=task.required_skill_ids,
        workspace_artifacts=len(workspace_artifact_ids or []),
    )

    log.info(
        "agent.start",
        task_id=task.task_id,
        role=task.owner_department.value,
        attempt=bundle.attempt_number,
        messages=len(messages),
        workspace_artifacts=len(workspace_artifact_ids or []),
    )

    # ── LLM call ─────────────────────────────────────────────────────
    try:
        raw_json, llm_result = await client.chat_json(
            messages,
            max_tokens=task.budget.max_tokens,
            retry_limit=2,
        )
    except Exception as exc:
        latency_ms = int((time.monotonic() - t_start) * 1000)
        log.error("agent.llm_error", task_id=task.task_id, error=str(exc))
        audit(
            AuditEventType.AGENT_LLM_ERROR,
            project_id=task.project_id,
            task_id=task.task_id,
            actor_type="agent",
            actor_id=task.owner_department.value,
            severity=AuditSeverity.ERROR,
            error=str(exc),
            latency_ms=latency_ms,
        )
        return RunResult(
            task_id=task.task_id,
            status="failed",
            failure_reason=f"LLM call failed: {exc}",
            latency_ms=latency_ms,
        )

    latency_ms = int((time.monotonic() - t_start) * 1000)

    # ── Parse output ─────────────────────────────────────────────────
    try:
        output = _builder.parse_agent_output(raw_json)
        output.raw_llm_content = llm_result.content
    except Exception as exc:
        log.error(
            "agent.parse_error",
            task_id=task.task_id,
            error=str(exc),
            raw=str(raw_json)[:300],
        )
        audit(
            AuditEventType.AGENT_PARSE_ERROR,
            project_id=task.project_id,
            task_id=task.task_id,
            actor_type="agent",
            actor_id=task.owner_department.value,
            severity=AuditSeverity.ERROR,
            error=str(exc),
            raw_preview=str(raw_json)[:200],
        )
        return RunResult(
            task_id=task.task_id,
            status="failed",
            failure_reason=f"Output schema validation failed: {exc}",
            latency_ms=latency_ms,
        )

    # ── Memory candidates (non-fatal if fails) ────────────────────────
    memory_ids: list[str] = []
    if output.memory_candidates:
        try:
            from memory.manager import get_memory_manager
            mgr = get_memory_manager()
            memory_ids = await mgr.submit_candidates(
                candidates=output.memory_candidates,
                project_id=task.project_id,
                task_id=task.task_id,
                agent_role=task.owner_department.value,
            )
        except Exception as exc:
            log.warning(
                "agent.memory_submit_failed",
                task_id=task.task_id,
                error=str(exc),
            )

    # ── Blocked short-circuit ─────────────────────────────────────────
    if output.status == "blocked":
        reason = output.blocked_reason or "Agent reported blocked status."
        log.info("agent.blocked", task_id=task.task_id, reason=reason[:200])
        return RunResult(
            task_id=task.task_id,
            status="blocked",
            output=output,
            failure_reason=reason,
            latency_ms=latency_ms,
            memory_ids_created=memory_ids,
        )

    # ── Validation ────────────────────────────────────────────────────
    validation_report: ValidationReport | None = None
    try:
        validation_report, _ = await validate_and_build_feedback(
            output=output,
            task=task,
            hard_constraints=hard_constraints,
            use_llm=use_llm_validation,
        )
    except Exception as exc:
        log.warning("agent.validation_error", task_id=task.task_id, error=str(exc))

    # ── Disposition ───────────────────────────────────────────────────
    if validation_report is not None:
        if validation_report.result == ValidationResult.PASS:
            final_status   = "approved"
            failure_reason = ""
        elif validation_report.result == ValidationResult.FAIL:
            final_status   = "failed"
            failure_reason = (
                f"Validation FAIL — {validation_report.composite_score:.0%} "
                f"vs threshold {validation_report.threshold:.0%}. "
                + (validation_report.issues[0] if validation_report.issues else "")
            )
        else:  # REVISE
            final_status   = "failed"
            failure_reason = (
                f"Validation REVISE — {validation_report.composite_score:.0%} "
                f"vs threshold {validation_report.threshold:.0%}."
            )
    else:
        # Fallback: simple confidence gate if validator raised
        if output.confidence >= task.quality_threshold:
            final_status   = "approved"
            failure_reason = ""
        else:
            final_status   = "failed"
            failure_reason = (
                f"Confidence {output.confidence:.0%} below threshold "
                f"{task.quality_threshold:.0%}."
            )

    if final_status == "approved":
        clear_iteration_history(task.task_id)

    log.info(
        "agent.complete",
        task_id=task.task_id,
        final_status=final_status,
        confidence=output.confidence,
        composite=validation_report.composite_score if validation_report else None,
        threshold=task.quality_threshold,
        memories=len(memory_ids),
        latency_ms=latency_ms,
        tokens=llm_result.usage.total_tokens,
    )
    audit(
        AuditEventType.AGENT_COMPLETED,
        project_id=task.project_id,
        task_id=task.task_id,
        actor_type="agent",
        actor_id=task.owner_department.value,
        severity=AuditSeverity.DEBUG,
        latency_ms=latency_ms,
        tokens=llm_result.usage.total_tokens,
        confidence=output.confidence,
        final_status=final_status,
    )
    if validation_report:
        audit(
            AuditEventType.VALIDATION_RESULT,
            project_id=task.project_id,
            task_id=task.task_id,
            severity=(
                AuditSeverity.INFO if final_status == "approved"
                else AuditSeverity.WARNING
            ),
            result=validation_report.result.value,
            composite_score=validation_report.composite_score,
            threshold=validation_report.threshold,
            department=task.owner_department.value,
            issues=validation_report.issues[:3],
            corrective_feedback=validation_report.to_corrective_feedback()[:400]
                if final_status != "approved" else "",
        )

    # Phase 9: if this was a first-attempt approval with skills, record effectiveness
    # immediately as a memory candidate from within the agent runner.
    # (Richer learning fires from engine hooks; this is a lightweight fast path.)
    if (
        final_status == "approved"
        and task.required_skill_ids
        and bundle.attempt_number == 1
        and validation_report
        and validation_report.composite_score >= 0.80
    ):
        try:
            from memory.manager import get_memory_manager
            _mgr = get_memory_manager()
            await _mgr.submit_candidates(
                candidates=[{
                    "category": "insight",
                    "title": (
                        f"Skill fit: {'+'.join(task.required_skill_ids[:2])} "
                        f"for {task.type.value}/{task.owner_department.value}"
                    ),
                    "content": (
                        f"Skills [{', '.join(task.required_skill_ids)}] achieved "
                        f"{validation_report.composite_score:.0%} on first attempt "
                        f"for task type '{task.type.value}' in '{task.owner_department.value}'."
                    ),
                    "confidence": min(0.88, validation_report.composite_score),
                    "tags": ["skill_fit", "first_attempt"] + task.required_skill_ids[:3],
                }],
                project_id=task.project_id,
                task_id=task.task_id,
                agent_role="learning",
            )
        except Exception:
            pass   # never block the return

    return RunResult(
        task_id=task.task_id,
        status=final_status,
        output=output,
        validation_report=validation_report,
        failure_reason=failure_reason,
        latency_ms=latency_ms,
        memory_ids_created=memory_ids,
    )


# ---------------------------------------------------------------------------
# Corrective feedback helpers
# ---------------------------------------------------------------------------


def build_corrective_feedback(result: RunResult) -> str:
    if result.validation_report is not None:
        return result.validation_report.to_corrective_feedback()

    # Legacy fallback
    if not result.output:
        return result.failure_reason or "Previous attempt produced no parseable output."
    parts: list[str] = []
    if result.failure_reason:
        parts.append(f"Failure reason: {result.failure_reason}")
    if result.output.risks:
        parts.append(
            "Risks to address:\n" + "\n".join(f"  - {r}" for r in result.output.risks)
        )
    if result.output.open_questions:
        parts.append(
            "Unresolved questions:\n"
            + "\n".join(f"  - {q}" for q in result.output.open_questions)
        )
    if not parts:
        parts.append("Output did not meet quality threshold. Improve depth and specificity.")
    return "\n\n".join(parts)


def build_prior_attempt_summary(result: RunResult) -> str:
    if not result.output:
        return ""
    out = result.output
    preview   = "; ".join(out.findings[:3]) if out.findings else "none"
    score_note = ""
    if result.validation_report:
        r          = result.validation_report
        score_note = f"Validation score: {r.composite_score:.0%} (threshold {r.threshold:.0%}). "
    return (
        f"Prior attempt: {out.summary[:300]} "
        f"Key findings: {preview}. "
        f"Confidence: {out.confidence:.0%}. {score_note}"
    )
