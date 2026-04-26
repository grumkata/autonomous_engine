"""
Project state model — the root object that all other entities belong to.

Design note: These are Pydantic *domain models* (what the application works
with). The SQLAlchemy ORM *table models* live in db/tables.py and mirror
these shapes. Keeping them separate lets you change persistence details
without touching business logic.
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


class ProjectStatus(StrEnum):
    INTAKE = "intake"
    STRUCTURING = "structuring"
    PLANNING = "planning"
    EXECUTION = "execution"
    REVIEW = "review"
    REVISION = "revision"
    CONSOLIDATION = "consolidation"
    CLOSURE = "closure"
    RETROSPECTIVE = "retrospective"
    ARCHIVED = "archived"


class ProjectType(StrEnum):
    SOFTWARE = "software"
    RESEARCH = "research"
    WRITING = "writing"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    DESIGN = "design"
    MIXED = "mixed"


class ConstraintType(StrEnum):
    SCOPE = "scope"
    TIME = "time"
    BUDGET = "budget"
    QUALITY = "quality"
    TECHNICAL = "technical"
    ETHICAL = "ethical"
    DOMAIN = "domain"


class DeliverableType(StrEnum):
    CODE = "code"
    REPORT = "report"
    DESIGN_DOC = "design_doc"
    RESEARCH_BRIEF = "research_brief"
    ANALYSIS = "analysis"
    PLAN = "plan"
    OUTLINE = "outline"
    MIXED = "mixed"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Goal(BaseModel):
    """
    A single objective the project is trying to achieve.
    Projects may have multiple goals, ordered by priority.
    """

    goal_id: str = Field(default_factory=lambda: prefixed_id("goal"))
    statement: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Clear, unambiguous statement of the goal.",
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Priority 1 = highest. Used to rank competing goals.",
    )
    success_criteria: list[str] = Field(
        default_factory=list,
        description="Measurable conditions that mean this goal is achieved.",
    )
    is_locked: bool = Field(
        default=False,
        description="Locked goals cannot be modified without a governance override.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Constraint(BaseModel):
    """
    A limit or requirement the project must operate within.
    Constraints are checked by the validation framework.
    """

    constraint_id: str = Field(default_factory=lambda: prefixed_id("cst"))
    type: ConstraintType
    description: str = Field(..., min_length=5, max_length=1000)
    is_hard: bool = Field(
        default=True,
        description="Hard constraints must not be violated. Soft constraints are targets.",
    )
    source: str = Field(
        default="user",
        description="Who specified this constraint: 'user', 'system', 'governance'.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Deliverable(BaseModel):
    """
    A concrete output the project must produce.
    Multiple deliverables map to different departments and task types.
    """

    deliverable_id: str = Field(default_factory=lambda: prefixed_id("dlv"))
    type: DeliverableType
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(default="", max_length=2000)
    quality_bar: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum composite validation score required before this deliverable is considered done.",
    )
    is_completed: bool = False
    completed_at: datetime | None = None
    artifact_ids: list[str] = Field(
        default_factory=list,
        description="IDs of memory artifacts that constitute this deliverable.",
    )


class ProjectBudget(BaseModel):
    """Resource limits for the project execution engine."""

    max_tasks: int = Field(default=500, ge=1)
    max_agent_calls: int = Field(default=2000, ge=1)
    max_iterations_per_task: int = Field(default=10, ge=1)
    max_runtime_seconds: int | None = Field(
        default=None,
        description="Wall-clock limit. None = no limit.",
    )


# ---------------------------------------------------------------------------
# Root project model
# ---------------------------------------------------------------------------


class Project(BaseModel):
    """
    Root domain model. All other entities reference project_id.

    Creation flow:
        1. Intake → project created with status=INTAKE
        2. Structuring → goals parsed, constraints set, deliverables defined
        3. Planning → task graph built (Phase 2), status=PLANNING
        4. Execution → orchestration begins (Phase 2)
    """

    project_id: str = Field(default_factory=lambda: prefixed_id("prj"))
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(default="", max_length=5000)
    type: ProjectType = ProjectType.MIXED
    status: ProjectStatus = ProjectStatus.INTAKE

    goals: list[Goal] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    deliverables: list[Deliverable] = Field(default_factory=list)
    budget: ProjectBudget = Field(default_factory=ProjectBudget)

    # Metadata
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Cross-project references (populated by learning pipeline in Phase 9)
    parent_project_id: str | None = None
    insight_ids: list[str] = Field(default_factory=list)

    # Raw intake text — preserved so decomposition agents can refer back to it
    raw_intake: str = Field(default="", max_length=50_000)

    @model_validator(mode="after")
    def at_least_one_goal(self) -> "Project":
        # Allowed to be empty at INTAKE stage; required before PLANNING
        if self.status not in (ProjectStatus.INTAKE, ProjectStatus.STRUCTURING):
            if not self.goals:
                raise ValueError("A project must have at least one goal before planning begins.")
        return self

    def primary_goal(self) -> Goal | None:
        """Return the highest-priority goal."""
        if not self.goals:
            return None
        return min(self.goals, key=lambda g: g.priority)

    def hard_constraints(self) -> list[Constraint]:
        return [c for c in self.constraints if c.is_hard]

    def is_over_budget(self, agent_calls: int, tasks: int) -> bool:
        return agent_calls > self.budget.max_agent_calls or tasks > self.budget.max_tasks


# ---------------------------------------------------------------------------
# Request / Response models (used by FastAPI endpoints)
# ---------------------------------------------------------------------------


class CreateProjectRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(default="", max_length=5000)
    type: ProjectType = ProjectType.MIXED
    raw_intake: str = Field(..., min_length=10, max_length=50_000)
    initial_goals: list[str] = Field(
        default_factory=list,
        description="Optional plain-text goals. Will be parsed into Goal objects.",
    )
    initial_constraints: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ProjectSummary(BaseModel):
    """Lightweight response — used in list endpoints."""

    project_id: str
    title: str
    type: ProjectType
    status: ProjectStatus
    goal_count: int
    deliverable_count: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_project(cls, p: Project) -> "ProjectSummary":
        return cls(
            project_id=p.project_id,
            title=p.title,
            type=p.type,
            status=p.status,
            goal_count=len(p.goals),
            deliverable_count=len(p.deliverables),
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
