"""
orchestrator/planner.py — The Planning Agent (Phase 1 of rebuild spec)

This is the ignition switch.  It takes a Project whose raw_intake has been
filled in and uses the LLM to decompose it into a complete task graph, then
writes that graph atomically to the database.

Without this module the system cannot run autonomously — a human would have to
manually POST every task before triggering /run.

Design contract
---------------
* One public coroutine: decompose_project(project) → list[Task]
* Retries the LLM call up to MAX_PLAN_RETRIES times with increasingly explicit
  instructions on each failure.
* If all retries fail the project is set to REVIEW and nothing is written.
* The full task graph is committed in a single transaction — partial graphs
  are not allowed.
* After a successful commit it automatically starts the OrchestrationEngine
  in the same background task so no further human action is needed.

Task graph rules enforced here
-------------------------------
* Every IMPLEMENTATION / DESIGN task gets a downstream CRITIQUE task
  (red_team or qa) and a downstream REVISION task.
* Graph must be acyclic (checked with TaskGraphSnapshot.has_cycle()).
* Max 200 tasks, max depth 6.
* Tasks with no upstream dependencies start as READY; others as PLANNED.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import select

from config import get_settings
from core.ids import prefixed_id
from db.engine import session_factory
from db.mappers import project_to_row, task_to_row
from db.tables import ProjectRow, TaskEdgeRow, TaskRow
from llm.client import get_client_for
from llm.schemas import Message
from models.project import Project, ProjectStatus
from models.task import (
    DepartmentOwner,
    Task,
    TaskBudget,
    TaskEdge,
    TaskGraphSnapshot,
    TaskPriority,
    TaskStatus,
    TaskType,
)

from orchestrator.audit import AuditEventType, AuditSeverity, audit
from orchestrator.skill_registry import get_skill_registry

log = structlog.get_logger(__name__)

MAX_PLAN_RETRIES = 3
MAX_TASKS = 200
MAX_DEPTH = 6

# ---------------------------------------------------------------------------
# Domain-specific planning hints (spec §1.5)
# ---------------------------------------------------------------------------

_DOMAIN_HINTS: dict[str, str] = {
    "software": (
        "Recommended flow: requirements_analysis → architecture_design → "
        "module_specification tasks (parallel) → implementation tasks (parallel, gated on specs) "
        "→ unit_test tasks (parallel with implementation) → integration_test → documentation → deployment_planning. "
        "Every implementation task must be followed by a unit_test task and a critique task."
    ),
    "writing": (
        "Recommended flow: research → outline → draft_by_chapter (parallel chapters) → "
        "critique_per_chapter → revision_per_chapter → integration_merge → proofreading → final. "
        "Include a world_and_character_bible task first if the project is fiction. "
        "Include a continuity_check task that validates chapter-to-chapter consistency."
    ),
    "research": (
        "Recommended flow: literature_review → hypothesis_formation → methodology_design → "
        "data_analysis → findings_synthesis → peer_critique → revision → final_report. "
        "All analysis tasks should be followed by a critique from the red_team department."
    ),
    "analysis": (
        "Recommended flow: problem_framing → data_gathering → analysis_methodology → "
        "findings → validation → final_report. "
        "Include a critique task after findings before final report."
    ),
    "design": (
        "Recommended flow: requirements_capture → concept_exploration (parallel) → "
        "design_selection → detailed_design → critique → revision → specification_document. "
        "Red team should review the detailed design before specification is written."
    ),
    "planning": (
        "Recommended flow: goal_clarification → constraint_analysis → option_generation (parallel) → "
        "option_critique → plan_synthesis → risk_review → final_plan. "
        "Include a red_team critique of the final plan before marking it complete."
    ),
    "mixed": (
        "Decompose into clear domains first, then apply domain-appropriate sub-flows. "
        "Ensure critique and revision cycles are present for every major output."
    ),
}

# ---------------------------------------------------------------------------
# Valid enum values for the LLM to reference
# ---------------------------------------------------------------------------

_VALID_TASK_TYPES = [t.value for t in TaskType if t != TaskType.APPROVAL]
_VALID_DEPARTMENTS = [d.value for d in DepartmentOwner]
_VALID_PRIORITIES = [p.value for p in TaskPriority]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_plan_prompt(project: Project, attempt: int) -> list[Message]:
    """Return the [system, user] message list for the planning LLM call."""

    domain_hint = _DOMAIN_HINTS.get(project.type.value, _DOMAIN_HINTS["mixed"])

    strictness = ""
    if attempt == 2:
        strictness = (
            "\n\nIMPORTANT: Your previous response was invalid JSON or had schema errors. "
            "Return ONLY a valid JSON object. No markdown, no preamble, no backticks."
        )
    elif attempt >= 3:
        strictness = (
            "\n\nCRITICAL: Previous attempts failed validation. "
            "Keep the plan simple — 5 to 15 tasks only. "
            "Return ONLY raw JSON, nothing else."
        )

    goals_text = "\n".join(
        f"  {i+1}. {g.statement}" for i, g in enumerate(project.goals)
    ) or "  (no explicit goals — infer from raw_intake)"

    constraints_text = "\n".join(
        f"  - [{c.type.value}] {c.description}" for c in project.constraints
    ) or "  (none specified)"

    system_prompt = textwrap.dedent(f"""
        You are the orchestration planning agent for an autonomous AI engine.
        Your job is to decompose a project into a directed acyclic task graph
        that a team of specialized AI agents can execute.

        RULES:
        1. Return ONLY a single JSON object — no markdown, no explanation, no backticks.
        2. Every IMPLEMENTATION or DESIGN task MUST have at least one downstream
           critique task (owner_department: "red_team" or "qa") and a downstream
           revision task (owner_department: "implementation" or "design").
        3. The graph must be acyclic — no circular dependencies.
        4. Maximum {MAX_TASKS} tasks, maximum depth {MAX_DEPTH}.
        5. Task titles must be unique within the plan.
        6. Edge references use the exact task title string.
        7. Tasks with no upstream deps are READY immediately; others are PLANNED.

        VALID task types: {_VALID_TASK_TYPES}
        VALID departments: {_VALID_DEPARTMENTS}
        VALID priorities: {_VALID_PRIORITIES}

        DOMAIN GUIDANCE for project type "{project.type.value}":
        {domain_hint}

        REQUIRED JSON SCHEMA:
        {{
          "tasks": [
            {{
              "title": "<unique string, max 150 chars>",
              "description": "<detailed string, min 20 chars>",
              "type": "<one of {_VALID_TASK_TYPES}>",
              "owner_department": "<one of {_VALID_DEPARTMENTS}>",
              "priority": "<one of {_VALID_PRIORITIES}>",
              "depth_level": <int 0-{MAX_DEPTH}>,
              "validation_criteria": ["<string>"],
              "expected_output_types": ["<string>"],
              "quality_threshold": <float 0.6-0.95>,
              "budget": {{
                "max_tokens": <int 512-8192>,
                "max_runtime_seconds": <int 60-600>,
                "max_retries": <int 1-5>
              }}
            }}
          ],
          "edges": [
            {{
              "upstream_title": "<exact title of upstream task>",
              "downstream_title": "<exact title of downstream task>"
            }}
          ]
        }}
        {strictness}
    """).strip()

    user_prompt = textwrap.dedent(f"""
        PROJECT: {project.title}
        TYPE: {project.type.value}
        DESCRIPTION: {project.description or '(none)'}

        GOALS:
        {goals_text}

        CONSTRAINTS:
        {constraints_text}

        RAW INTAKE:
        {project.raw_intake[:3000]}

        Decompose this project into a complete task graph following the rules above.
        Ensure every major output goes through: produce → critique → revise → verify.
        Return only the JSON object.
    """).strip()

    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]


# ---------------------------------------------------------------------------
# Graph validation
# ---------------------------------------------------------------------------

def _validate_plan(
    raw_tasks: list[dict],
    raw_edges: list[dict],
) -> tuple[bool, str]:
    """
    Validate the LLM's raw plan before writing to DB.
    Returns (is_valid, error_message).
    """
    if not raw_tasks:
        return False, "No tasks returned."

    if len(raw_tasks) > MAX_TASKS:
        return False, f"Too many tasks: {len(raw_tasks)} > {MAX_TASKS}."

    titles: set[str] = set()
    for i, t in enumerate(raw_tasks):
        title = t.get("title", "")
        if not title:
            return False, f"Task {i} missing title."
        if title in titles:
            return False, f"Duplicate task title: {title!r}."
        titles.add(title)
        if t.get("depth_level", 0) > MAX_DEPTH:
            return False, f"Task {title!r} exceeds max depth {MAX_DEPTH}."

    for e in raw_edges:
        up = e.get("upstream_title", "")
        down = e.get("downstream_title", "")
        if up not in titles:
            return False, f"Edge references unknown upstream task: {up!r}."
        if down not in titles:
            return False, f"Edge references unknown downstream task: {down!r}."
        if up == down:
            return False, f"Self-loop on task: {up!r}."

    return True, ""


def _enforce_critique_chains(
    raw_tasks: list[dict],
    raw_edges: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Ensure every IMPLEMENTATION / DESIGN task has a downstream critique task.
    Automatically injects missing critique + revision tasks rather than failing.
    """
    needs_chain = {"implementation", "design"}
    downstream_titles: set[str] = {e["downstream_title"] for e in raw_edges}

    # Find implementation/design tasks that have NO downstream critique
    task_map = {t["title"]: t for t in raw_tasks}
    critique_upstream: set[str] = set()

    for edge in raw_edges:
        down = task_map.get(edge["downstream_title"], {})
        if down.get("owner_department") in ("red_team", "qa"):
            critique_upstream.add(edge["upstream_title"])

    injected_tasks: list[dict] = []
    injected_edges: list[dict] = []

    for t in raw_tasks:
        if t.get("type") not in needs_chain:
            continue
        if t["title"] in critique_upstream:
            continue  # already has a critique

        critique_title = f"Critique: {t['title'][:80]}"
        revision_title = f"Revision: {t['title'][:80]}"

        if critique_title not in task_map:
            dept = "red_team" if t.get("quality_threshold", 0.75) >= 0.85 else "qa"
            injected_tasks.append({
                "title": critique_title,
                "description": (
                    f"Adversarially review the output of '{t['title']}'. "
                    "Identify gaps, inconsistencies, and weaknesses."
                ),
                "type": "critique",
                "owner_department": dept,
                "priority": t.get("priority", "medium"),
                "depth_level": min(t.get("depth_level", 0) + 1, MAX_DEPTH),
                "validation_criteria": [
                    "Identifies at least 3 specific weaknesses or gaps",
                    "Provides actionable improvement recommendations",
                    "Does not simply restate the original output",
                ],
                "expected_output_types": ["critique_report"],
                "quality_threshold": 0.75,
                "budget": {"max_tokens": 2048, "max_runtime_seconds": 180, "max_retries": 2},
            })
            injected_edges.append({
                "upstream_title": t["title"],
                "downstream_title": critique_title,
            })

        if revision_title not in task_map:
            injected_tasks.append({
                "title": revision_title,
                "description": (
                    f"Revise the output of '{t['title']}' based on critique feedback. "
                    "Address every identified weakness."
                ),
                "type": "refinement",
                "owner_department": t.get("owner_department", "implementation"),
                "priority": t.get("priority", "medium"),
                "depth_level": min(t.get("depth_level", 0) + 2, MAX_DEPTH),
                "validation_criteria": [
                    "All critique points have been addressed",
                    "Quality is improved over the prior attempt",
                    "No new problems introduced during revision",
                ],
                "expected_output_types": t.get("expected_output_types", []),
                "quality_threshold": t.get("quality_threshold", 0.80),
                "budget": t.get("budget", {"max_tokens": 4096, "max_runtime_seconds": 300, "max_retries": 3}),
            })
            injected_edges.append({
                "upstream_title": critique_title,
                "downstream_title": revision_title,
            })

    return raw_tasks + injected_tasks, raw_edges + injected_edges


def _build_domain_objects(
    project: Project,
    raw_tasks: list[dict],
    raw_edges: list[dict],
) -> tuple[list[Task], list[TaskEdge]]:
    """Convert raw dicts from LLM into domain objects."""

    title_to_id: dict[str, str] = {}
    tasks: list[Task] = []

    # Determine which titles have upstream edges (so we set PLANNED vs READY)
    has_upstream: set[str] = {e["downstream_title"] for e in raw_edges}

    for raw in raw_tasks:
        title = raw["title"]
        task_id = prefixed_id("tsk")
        title_to_id[title] = task_id

        initial_status = (
            TaskStatus.PLANNED if title in has_upstream else TaskStatus.READY
        )

        budget_raw = raw.get("budget", {})
        budget = TaskBudget(
            max_tokens=min(max(budget_raw.get("max_tokens", 4096), 512), 8192),
            max_runtime_seconds=min(max(budget_raw.get("max_runtime_seconds", 300), 60), 600),
            max_retries=min(max(budget_raw.get("max_retries", 3), 1), 5),
        )

        # Map type string → TaskType, fallback to RESEARCH
        try:
            task_type = TaskType(raw.get("type", "research"))
        except ValueError:
            task_type = TaskType.RESEARCH

        # Map department string → DepartmentOwner, fallback to ORCHESTRATION
        try:
            dept = DepartmentOwner(raw.get("owner_department", "orchestration"))
        except ValueError:
            dept = DepartmentOwner.ORCHESTRATION

        # Map priority string → TaskPriority, fallback to MEDIUM
        try:
            priority = TaskPriority(raw.get("priority", "medium"))
        except ValueError:
            priority = TaskPriority.MEDIUM

        task = Task(
            task_id=task_id,
            project_id=project.project_id,
            type=task_type,
            title=title,
            description=raw.get("description", title),
            status=initial_status,
            priority=priority,
            owner_department=dept,
            depth_level=max(0, min(raw.get("depth_level", 1), MAX_DEPTH)),
            validation_criteria=raw.get("validation_criteria", []),
            expected_output_types=raw.get("expected_output_types", []),
            quality_threshold=max(0.5, min(raw.get("quality_threshold", 0.75), 0.99)),
            budget=budget,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        # Phase 4: annotate task with suggested skills
        task.required_skill_ids = get_skill_registry().suggest_for_task(
            task.type.value, task.owner_department.value
        )

        tasks.append(task)

    edges: list[TaskEdge] = []
    for raw_edge in raw_edges:
        up_id = title_to_id.get(raw_edge["upstream_title"])
        down_id = title_to_id.get(raw_edge["downstream_title"])
        if up_id and down_id:
            edges.append(
                TaskEdge(
                    project_id=project.project_id,
                    upstream_task_id=up_id,
                    downstream_task_id=down_id,
                    created_at=datetime.now(timezone.utc),
                )
            )

    return tasks, edges


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

async def _write_graph_to_db(
    project: Project,
    tasks: list[Task],
    edges: list[TaskEdge],
) -> None:
    """Write the full task graph in a single transaction. All or nothing."""
    from db.mappers import task_to_row

    async with session_factory() as db:
        # Update project status to PLANNING
        proj_row = await db.get(ProjectRow, project.project_id)
        if proj_row:
            proj_row.status = ProjectStatus.PLANNING.value
            proj_row.updated_at = datetime.now(timezone.utc)

        for task in tasks:
            db.add(task_to_row(task))

        for edge in edges:
            db.add(
                TaskEdgeRow(
                    edge_id=edge.edge_id,
                    project_id=edge.project_id,
                    upstream_task_id=edge.upstream_task_id,
                    downstream_task_id=edge.downstream_task_id,
                    edge_type=edge.edge_type,
                    created_at=edge.created_at,
                )
            )
    # session_factory auto-commits on clean exit
        # Phase 8: emit task_generated for every task written
        for task in tasks:
            audit(
                AuditEventType.TASK_GENERATED,
                project_id=project.project_id,
                task_id=task.task_id,
                severity=AuditSeverity.DEBUG,
                task_type=task.type.value,
                department=task.owner_department.value,
                depth_level=task.depth_level,
                skills=task.required_skill_ids,
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def decompose_project(project: Project) -> list[Task]:
    """
    Decompose a project into a task graph and write it to the database.

    Steps:
      1. Build planning prompt (with domain hints)
      2. Call LLM (up to MAX_PLAN_RETRIES times)
      3. Parse + validate the JSON graph
      4. Enforce critique chains
      5. Check for cycles
      6. Write atomically to DB
      7. Update project status → PLANNING

    Returns the list of Task objects on success.
    Raises RuntimeError if all retries fail.
    """
    log.info("planner.start", project_id=project.project_id, title=project.title)

    client = get_client_for("planner", "planning")
    settings = get_settings()
    last_error = "Unknown error"

    for attempt in range(1, MAX_PLAN_RETRIES + 1):
        log.info("planner.attempt", attempt=attempt, project_id=project.project_id)

        messages = _build_plan_prompt(project, attempt)

        try:
            raw_json, llm_result = await client.chat_json(
                messages,
                max_tokens=4096,
                retry_limit=1,
            )
        except Exception as exc:
            last_error = f"LLM call failed: {exc}"
            log.warning("planner.llm_error", attempt=attempt, error=last_error)
            continue

        # ── Parse ─────────────────────────────────────────────────────────
        try:
            raw_tasks: list[dict] = raw_json.get("tasks", [])
            raw_edges: list[dict] = raw_json.get("edges", [])
        except Exception as exc:
            last_error = f"JSON parse error: {exc}"
            log.warning("planner.parse_error", attempt=attempt, error=last_error)
            continue

        # ── Validate ──────────────────────────────────────────────────────
        valid, err = _validate_plan(raw_tasks, raw_edges)
        if not valid:
            last_error = f"Validation failed: {err}"
            log.warning("planner.validation_error", attempt=attempt, error=last_error)
            continue

        # ── Enforce critique chains ────────────────────────────────────────
        raw_tasks, raw_edges = _enforce_critique_chains(raw_tasks, raw_edges)

        # Re-validate after injection
        valid, err = _validate_plan(raw_tasks, raw_edges)
        if not valid:
            last_error = f"Post-injection validation failed: {err}"
            log.warning("planner.post_inject_error", attempt=attempt, error=last_error)
            continue

        # ── Build domain objects ───────────────────────────────────────────
        tasks, edges = _build_domain_objects(project, raw_tasks, raw_edges)

        # ── Cycle check ───────────────────────────────────────────────────
        snapshot = TaskGraphSnapshot(
            project_id=project.project_id,
            tasks=tasks,
            edges=edges,
        )
        if snapshot.has_cycle():
            last_error = "LLM produced a cyclic task graph — retrying."
            log.warning("planner.cycle_detected", attempt=attempt)
            continue

        # ── All good — write to DB ─────────────────────────────────────────
        try:
            await _write_graph_to_db(project, tasks, edges)
        except Exception as exc:
            # DB failures are not LLM failures — retrying the LLM won't help.
            # Break immediately so the retry loop exits cleanly and the
            # "all retries exhausted" block below can set the project to REVIEW.
            last_error = f"DB write failed: {exc}"
            log.error(
                "planner.db_write_error",
                project_id=project.project_id,
                attempt=attempt,
                error=last_error,
            )
            break

        log.info(
            "planner.success",
            project_id=project.project_id,
            tasks=len(tasks),
            edges=len(edges),
            attempt=attempt,
            tokens=llm_result.usage.total_tokens,
        )
        return tasks

    # ── All retries exhausted ──────────────────────────────────────────────
    log.error("planner.failed", project_id=project.project_id, reason=last_error)

    async with session_factory() as db:
        proj_row = await db.get(ProjectRow, project.project_id)
        if proj_row:
            proj_row.status = ProjectStatus.REVIEW.value
            proj_row.updated_at = datetime.now(timezone.utc)

    raise RuntimeError(
        f"Planning failed for project {project.project_id!r} after "
        f"{MAX_PLAN_RETRIES} attempts. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Plan + run pipeline (called from api/projects.py background task)
# ---------------------------------------------------------------------------

async def plan_and_run(project: Project) -> None:
    """
    Full pipeline: decompose the project, then immediately start the engine.
    Designed to be called as a FastAPI BackgroundTask.
    """
    try:
        await decompose_project(project)
    except Exception as exc:
        log.error(
            "planner.pipeline_failed",
            project_id=project.project_id,
            error=str(exc),
        )
        # Safety net: decompose_project sets REVIEW before raising, but if that
        # write itself failed the project could still be stuck at INTAKE forever.
        # Re-apply REVIEW here so the project is always recoverable.
        try:
            async with session_factory() as db:
                proj_row = await db.get(ProjectRow, project.project_id)
                if proj_row and proj_row.status not in (
                    ProjectStatus.REVIEW.value,
                    "closure",
                    "archived",
                ):
                    proj_row.status = ProjectStatus.REVIEW.value
                    proj_row.updated_at = datetime.now(timezone.utc)
        except Exception as db_exc:
            log.error(
                "planner.failsafe_status_update_failed",
                project_id=project.project_id,
                error=str(db_exc),
            )
        return

    # Auto-start the engine — no human /run call needed
    try:
        from orchestrator.engine_orchestrator import OrchestrationEngine
        settings = get_settings()
        engine = OrchestrationEngine(settings)
        await engine.run_project(project.project_id)
    except Exception as exc:
        log.error(
            "engine.unhandled_exception",
            project_id=project.project_id,
            error=str(exc),
        )
