"""
Task graph schema.

A project's work is represented as a directed acyclic graph (DAG) of tasks.
This module defines the domain models. The execution engine (Phase 2) imports
these and builds the live graph using networkx.

Key design decisions:
  - TaskEdge is explicit: no implicit "the next task in a list" ordering.
  - validation_criteria are strings here; the validation framework (Phase 4)
    reads them and computes scores.
  - retry_limit is per-task so complex tasks can have more attempts than
    simple ones.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from core.ids import prefixed_id


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(StrEnum):
    PLANNED = "planned"         # created, dependencies not yet met
    READY = "ready"             # all dependencies satisfied, waiting for agent slot
    IN_PROGRESS = "in_progress" # agent is currently working on it
    BLOCKED = "blocked"         # waiting on external dependency or human gate
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"           # exceeded retry_limit or unrecoverable error
    APPROVED = "approved"       # passed validation, output promoted
    ARCHIVED = "archived"       # done, moved to artifact memory


class TaskType(StrEnum):
    ROOT_GOAL = "root_goal"         # the single top-level decomposition task
    PLANNING = "planning"
    RESEARCH = "research"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    CRITIQUE = "critique"
    VALIDATION = "validation"
    MERGE = "merge"
    REFINEMENT = "refinement"
    APPROVAL = "approval"           # waits for human intervention
    ARCHIVE = "archive"


class TaskPriority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def as_int(self) -> int:
        return {"low": 3, "medium": 2, "high": 1, "critical": 0}[self.value]


class DepartmentOwner(StrEnum):
    RESEARCH = "research"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    QA = "qa"
    RED_TEAM = "red_team"
    INTEGRATION = "integration"
    DOCUMENTATION = "documentation"
    GOVERNANCE = "governance"
    ORCHESTRATION = "orchestration"   # internal — used by planner tasks


# ---------------------------------------------------------------------------
# Task attempt record (one entry per retry)
# ---------------------------------------------------------------------------


class TaskAttempt(BaseModel):
    attempt_number: int
    agent_id: str
    started_at: datetime
    completed_at: datetime | None = None
    output_artifact_id: str | None = None
    validation_result: str | None = None   # "pass" | "fail" | "revise"
    composite_score: float | None = None
    failure_reason: str | None = None
    corrective_feedback: str | None = None  # injected on retry


# ---------------------------------------------------------------------------
# Task budget (per-task resource limits)
# ---------------------------------------------------------------------------


class TaskBudget(BaseModel):
    max_tokens: int = Field(default=4096, ge=256)
    max_runtime_seconds: int = Field(default=300, ge=10)
    max_retries: int = Field(default=3, ge=0)


# ---------------------------------------------------------------------------
# Core task model
# ---------------------------------------------------------------------------


class Task(BaseModel):
    """
    A single unit of work in the project task graph.

    Lifecycle:
        PLANNED → READY (when deps complete) → IN_PROGRESS (agent picked up)
             → NEEDS_REVIEW → APPROVED (validation pass)
                           → FAILED    (retry limit hit)
             → BLOCKED (human gate or missing resource)

    The orchestration engine (Phase 2) manages all status transitions.
    This model is the source of truth for a task's current state.
    """

    task_id: str = Field(default_factory=lambda: prefixed_id("tsk"))
    project_id: str
    type: TaskType
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10, max_length=5000)
    status: TaskStatus = TaskStatus.PLANNED
    priority: TaskPriority = TaskPriority.MEDIUM

    # Ownership & assignment
    owner_department: DepartmentOwner
    assigned_agent_ids: list[str] = Field(default_factory=list)

    # Graph structure — populated by planning agent in Phase 2
    parent_task_id: str | None = None   # for hierarchical task trees
    depth_level: int = Field(
        default=0,
        description="0=root, 1=milestone, 2=task, 3=subtask. Maps to spec §25 L0-L3.",
    )

    # I/O
    input_artifact_ids: list[str] = Field(default_factory=list)
    expected_output_types: list[str] = Field(
        default_factory=list,
        description="e.g. ['research_note', 'design_doc', 'code_artifact']",
    )
    output_artifact_ids: list[str] = Field(default_factory=list)

    # Validation
    validation_criteria: list[str] = Field(
        default_factory=list,
        description="Plain-language criteria. Validation framework (Phase 4) scores against these.",
    )
    quality_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum composite score to auto-approve.",
    )
    requires_human_approval: bool = False

    # Phase 4: skill names to load for this task (e.g. ["code/python", "qa/testing"])
    required_skill_ids: list[str] = Field(
        default_factory=list,
        description="Skill names from the registry to inject into this task's context.",
    )

    # Phase 6: human operator notes injected into agent context on next run
    human_notes: list[str] = Field(
        default_factory=list,
        description="Notes added by the human operator via the governance API.",
    )

    # Execution tracking
    budget: TaskBudget = Field(default_factory=TaskBudget)
    attempt_count: int = 0
    attempts: list[TaskAttempt] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Extra context (injected by orchestrator, not persisted long-term)
    context_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Ad-hoc context additions for this task invocation.",
        exclude=True,  # excluded from serialization to DB
    )

    @model_validator(mode="after")
    def approval_tasks_require_human_flag(self) -> "Task":
        if self.type == TaskType.APPROVAL and not self.requires_human_approval:
            raise ValueError("Tasks of type APPROVAL must have requires_human_approval=True.")
        return self

    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.APPROVED, TaskStatus.FAILED, TaskStatus.ARCHIVED)

    def is_retryable(self) -> bool:
        return (
            self.status == TaskStatus.FAILED
            and self.attempt_count < self.budget.max_retries
        )

    def last_attempt(self) -> TaskAttempt | None:
        return self.attempts[-1] if self.attempts else None

    def record_attempt_start(self, agent_id: str) -> TaskAttempt:
        attempt = TaskAttempt(
            attempt_number=self.attempt_count + 1,
            agent_id=agent_id,
            started_at=datetime.now(timezone.utc),
        )
        self.attempts.append(attempt)
        self.attempt_count += 1
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = self.started_at or attempt.started_at
        self.updated_at = datetime.now(timezone.utc)
        return attempt


# ---------------------------------------------------------------------------
# Task graph edge (explicit dependency)
# ---------------------------------------------------------------------------


class TaskEdge(BaseModel):
    """
    Directed edge: upstream_task_id → downstream_task_id.

    The downstream task cannot move to READY until the upstream task
    reaches APPROVED status.
    """

    edge_id: str = Field(default_factory=lambda: prefixed_id("edg"))
    project_id: str
    upstream_task_id: str
    downstream_task_id: str

    # Edge type — used by merge logic in Phase 2
    edge_type: str = Field(
        default="requires",
        description="'requires' = hard dependency. 'informs' = soft (downstream gets upstream output but can start earlier).",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def no_self_loop(self) -> "TaskEdge":
        if self.upstream_task_id == self.downstream_task_id:
            raise ValueError("A task cannot depend on itself.")
        return self


# ---------------------------------------------------------------------------
# Task graph snapshot (used by orchestrator and checkpoints)
# ---------------------------------------------------------------------------


class TaskGraphSnapshot(BaseModel):
    """
    Complete graph state for a project at a point in time.
    Serialized into checkpoint records (Phase 6).
    """

    project_id: str
    tasks: list[Task]
    edges: list[TaskEdge]
    snapshotted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def ready_tasks(self) -> list[Task]:
        """Tasks where all upstream dependencies are APPROVED."""
        approved_ids = {t.task_id for t in self.tasks if t.status == TaskStatus.APPROVED}
        downstream_map: dict[str, list[str]] = {}
        for edge in self.edges:
            downstream_map.setdefault(edge.downstream_task_id, []).append(edge.upstream_task_id)

        result = []
        for task in self.tasks:
            if task.status != TaskStatus.PLANNED:
                continue
            required_upstreams = downstream_map.get(task.task_id, [])
            if all(uid in approved_ids for uid in required_upstreams):
                result.append(task)
        return result

    def has_cycle(self) -> bool:
        """
        Detect cycles via DFS. The graph must be a DAG.
        Called during task graph creation to catch bad decompositions.
        """
        adjacency: dict[str, list[str]] = {}
        for edge in self.edges:
            adjacency.setdefault(edge.upstream_task_id, []).append(edge.downstream_task_id)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor in adjacency.get(node, []):
                if color.get(neighbor) == GRAY:
                    return True  # back edge → cycle
                if color.get(neighbor, WHITE) == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        all_nodes = {t.task_id for t in self.tasks}
        for node in all_nodes:
            if color.get(node, WHITE) == WHITE:
                if dfs(node):
                    return True
        return False


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateTaskRequest(BaseModel):
    project_id: str
    type: TaskType
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10, max_length=5000)
    owner_department: DepartmentOwner
    priority: TaskPriority = TaskPriority.MEDIUM
    validation_criteria: list[str] = Field(default_factory=list)
    dependency_task_ids: list[str] = Field(
        default_factory=list,
        description="These task IDs will become upstream nodes (edges created automatically).",
    )
    requires_human_approval: bool = False


class TaskStatusUpdate(BaseModel):
    status: TaskStatus
    reason: str = Field(default="", max_length=1000)
