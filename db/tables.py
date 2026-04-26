"""
SQLAlchemy 2.0 ORM table definitions.

Phase 3: WorkspaceRow
Phase 4: SkillRow + required_skill_ids_json on TaskRow
Phase 6: frozen + notes_json on ProjectRow; human_notes_json on TaskRow
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


class ProjectRow(Base):
    __tablename__ = "projects"

    project_id:  Mapped[str] = mapped_column(String(50), primary_key=True)
    title:       Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    type:        Mapped[str] = mapped_column(String(30), nullable=False)
    status:      Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    raw_intake:  Mapped[str] = mapped_column(Text, default="")

    goals_json:        Mapped[str] = mapped_column(Text, default="[]")
    constraints_json:  Mapped[str] = mapped_column(Text, default="[]")
    deliverables_json: Mapped[str] = mapped_column(Text, default="[]")
    budget_json:       Mapped[str] = mapped_column(Text, default="{}")
    tags_json:         Mapped[str] = mapped_column(Text, default="[]")
    insight_ids_json:  Mapped[str] = mapped_column(Text, default="[]")

    parent_project_id: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Phase 6: freeze flag + human notes on project
    frozen:     Mapped[bool] = mapped_column(Boolean, default=False)
    notes_json: Mapped[str]  = mapped_column(Text, default="[]")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    tasks:          Mapped[list["TaskRow"]]         = relationship(
        "TaskRow", back_populates="project", cascade="all, delete-orphan"
    )
    memory_objects: Mapped[list["MemoryObjectRow"]] = relationship(
        "MemoryObjectRow", back_populates="project", cascade="all, delete-orphan"
    )
    workspaces:     Mapped[list["WorkspaceRow"]]    = relationship(
        "WorkspaceRow", back_populates="project", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_projects_status_created", "status", "created_at"),
    )


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class TaskRow(Base):
    __tablename__ = "tasks"

    task_id:    Mapped[str] = mapped_column(String(50), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False
    )
    type:             Mapped[str]  = mapped_column(String(30), nullable=False)
    title:            Mapped[str]  = mapped_column(String(200), nullable=False)
    description:      Mapped[str]  = mapped_column(Text, nullable=False)
    status:           Mapped[str]  = mapped_column(String(30), nullable=False, index=True)
    priority:         Mapped[str]  = mapped_column(String(20), nullable=False, default="medium")
    owner_department: Mapped[str]  = mapped_column(String(30), nullable=False)

    parent_task_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    depth_level:    Mapped[int]        = mapped_column(Integer, default=0)

    assigned_agent_ids_json:    Mapped[str] = mapped_column(Text, default="[]")
    input_artifact_ids_json:    Mapped[str] = mapped_column(Text, default="[]")
    expected_output_types_json: Mapped[str] = mapped_column(Text, default="[]")
    output_artifact_ids_json:   Mapped[str] = mapped_column(Text, default="[]")
    validation_criteria_json:   Mapped[str] = mapped_column(Text, default="[]")
    budget_json:                Mapped[str] = mapped_column(Text, default="{}")
    attempts_json:              Mapped[str] = mapped_column(Text, default="[]")

    # Phase 4: skill names to load for this task
    required_skill_ids_json: Mapped[str] = mapped_column(Text, default="[]")

    # Phase 6: human operator notes injected into agent context on next run
    human_notes_json: Mapped[str] = mapped_column(Text, default="[]")

    attempt_count:           Mapped[int]   = mapped_column(Integer, default=0)
    quality_threshold:       Mapped[float] = mapped_column(Float, default=0.75)
    requires_human_approval: Mapped[bool]  = mapped_column(Boolean, default=False)

    created_at:   Mapped[datetime]       = mapped_column(DateTime(timezone=True), default=_now)
    updated_at:   Mapped[datetime]       = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    started_at:   Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project:          Mapped["ProjectRow"]       = relationship("ProjectRow", back_populates="tasks")
    upstream_edges:   Mapped[list["TaskEdgeRow"]] = relationship(
        "TaskEdgeRow",
        foreign_keys="TaskEdgeRow.downstream_task_id",
        back_populates="downstream_task",
        cascade="all, delete-orphan",
    )
    downstream_edges: Mapped[list["TaskEdgeRow"]] = relationship(
        "TaskEdgeRow",
        foreign_keys="TaskEdgeRow.upstream_task_id",
        back_populates="upstream_task",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_tasks_project_status", "project_id", "status"),
        Index("ix_tasks_project_type",   "project_id", "type"),
        Index("ix_tasks_department",     "owner_department"),
    )


# ---------------------------------------------------------------------------
# Task edges
# ---------------------------------------------------------------------------


class TaskEdgeRow(Base):
    __tablename__ = "task_edges"

    edge_id:            Mapped[str] = mapped_column(String(50), primary_key=True)
    project_id:         Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    upstream_task_id:   Mapped[str] = mapped_column(
        String(50), ForeignKey("tasks.task_id", ondelete="CASCADE"), nullable=False
    )
    downstream_task_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tasks.task_id", ondelete="CASCADE"), nullable=False
    )
    edge_type:  Mapped[str]      = mapped_column(String(20), default="requires")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    upstream_task:   Mapped["TaskRow"] = relationship(
        "TaskRow", foreign_keys=[upstream_task_id], back_populates="downstream_edges"
    )
    downstream_task: Mapped["TaskRow"] = relationship(
        "TaskRow", foreign_keys=[downstream_task_id], back_populates="upstream_edges"
    )

    __table_args__ = (
        Index("ix_edges_upstream",   "upstream_task_id"),
        Index("ix_edges_downstream", "downstream_task_id"),
    )


# ---------------------------------------------------------------------------
# Workspaces  (Phase 3)
# ---------------------------------------------------------------------------


class WorkspaceRow(Base):
    __tablename__ = "workspaces"

    workspace_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    project_id:   Mapped[str] = mapped_column(
        String(50), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False
    )
    department:  Mapped[str] = mapped_column(String(30), nullable=False)
    status:      Mapped[str] = mapped_column(String(20), nullable=False, default="created", index=True)

    artifact_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    task_ids_json:     Mapped[str] = mapped_column(Text, default="[]")
    scratch_pad:       Mapped[str] = mapped_column(Text, default="")

    created_at:  Mapped[datetime]       = mapped_column(DateTime(timezone=True), default=_now)
    updated_at:  Mapped[datetime]       = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped["ProjectRow"] = relationship("ProjectRow", back_populates="workspaces")

    __table_args__ = (
        Index("ix_workspaces_project_dept",   "project_id", "department"),
        Index("ix_workspaces_project_status", "project_id", "status"),
    )


# ---------------------------------------------------------------------------
# Skills  (Phase 4)
# ---------------------------------------------------------------------------


class SkillRow(Base):
    __tablename__ = "skills"

    skill_id:         Mapped[str] = mapped_column(String(50), primary_key=True)
    name:             Mapped[str] = mapped_column(String(100), nullable=False)
    version:          Mapped[str] = mapped_column(String(20),  nullable=False, default="1.0")
    description:      Mapped[str] = mapped_column(String(300), default="")
    departments_json: Mapped[str] = mapped_column(Text, default="[]")
    content_md:       Mapped[str] = mapped_column(Text, default="")
    tool_ids_json:    Mapped[str] = mapped_column(Text, default="[]")
    source:           Mapped[str] = mapped_column(String(20),  default="builtin")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    __table_args__ = (
        Index("ix_skills_name_version", "name", "version"),
        Index("ix_skills_source",       "source"),
    )




# ---------------------------------------------------------------------------
# Checkpoints  (Phase 7)
# ---------------------------------------------------------------------------


class CheckpointRow(Base):
    """
    Point-in-time snapshot of a project's full execution state.

    snapshot_json stores the complete serialised Checkpoint domain object
    (project, tasks, edges, workspaces) as a single JSON blob.  This is
    intentionally denormalised — a checkpoint must be self-contained so it
    can be restored without depending on any other table rows that may have
    changed since the snapshot was taken.
    """

    __tablename__ = "checkpoints"

    checkpoint_id:  Mapped[str] = mapped_column(String(50), primary_key=True)
    project_id:     Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    trigger_reason: Mapped[str] = mapped_column(String(40), nullable=False, default="manual")
    snapshot_json:  Mapped[str] = mapped_column(Text, nullable=False)

    # Denormalised counts for fast listing (avoids deserialising the blob)
    task_count:        Mapped[int] = mapped_column(Integer, default=0)
    approved_count:    Mapped[int] = mapped_column(Integer, default=0)
    in_progress_count: Mapped[int] = mapped_column(Integer, default=0)
    iteration:         Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        Index("ix_checkpoints_project", "project_id"),
        Index("ix_checkpoints_created", "project_id", "created_at"),
    )


# ---------------------------------------------------------------------------
# Audit events  (Phase 8)
# ---------------------------------------------------------------------------


class AuditEventRow(Base):
    """
    Persisted record of every significant engine action.

    detail_json holds event-specific metadata (validation scores,
    failure reasons, memory IDs, etc.) as a JSON blob so the schema
    stays stable regardless of how many new event types we add.

    before_json / after_json are optional state snapshots — they are
    only populated when the state change is meaningful to replay
    (e.g. task status PLANNED → APPROVED, project status changes).
    """

    __tablename__ = "audit_events"

    event_id:   Mapped[str] = mapped_column(String(50), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    task_id:    Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)

    actor_type: Mapped[str] = mapped_column(String(20), nullable=False, default="engine")
    actor_id:   Mapped[str] = mapped_column(String(100), nullable=False, default="")

    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    severity:   Mapped[str] = mapped_column(String(10), nullable=False, default="info", index=True)

    before_json: Mapped[str] = mapped_column(Text, default="{}")
    after_json:  Mapped[str] = mapped_column(Text, default="{}")
    detail_json: Mapped[str] = mapped_column(Text, default="{}")

    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    __table_args__ = (
        Index("ix_audit_project_type",     "project_id", "event_type"),
        Index("ix_audit_project_time",     "project_id", "occurred_at"),
        Index("ix_audit_task",             "task_id"),
        Index("ix_audit_severity",         "severity", "occurred_at"),
    )

# ---------------------------------------------------------------------------
# Memory objects
# ---------------------------------------------------------------------------


class MemoryObjectRow(Base):
    __tablename__ = "memory_objects"

    memory_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    tier:      Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    category:  Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    scope:     Mapped[str] = mapped_column(String(20), nullable=False, default="project")
    status:    Mapped[str] = mapped_column(String(20), nullable=False, default="draft", index=True)

    project_id:    Mapped[str | None] = mapped_column(
        String(50), ForeignKey("projects.project_id", ondelete="SET NULL"), nullable=True
    )
    task_id:       Mapped[str | None] = mapped_column(String(50), nullable=True)
    department_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    agent_id:      Mapped[str | None] = mapped_column(String(50), nullable=True)

    title:            Mapped[str] = mapped_column(String(300), nullable=False)
    content:          Mapped[str] = mapped_column(Text, nullable=False)
    tags_json:        Mapped[str] = mapped_column(Text, default="[]")
    source_refs_json: Mapped[str] = mapped_column(Text, default="[]")

    confidence:          Mapped[float]      = mapped_column(Float, default=0.5)
    relevance:           Mapped[float]      = mapped_column(Float, default=0.5)
    version:             Mapped[int]        = mapped_column(Integer, default=1)
    previous_version_id: Mapped[str | None] = mapped_column(String(50), nullable=True)

    visibility:   Mapped[str] = mapped_column(String(20), default="project")
    write_policy: Mapped[str] = mapped_column(String(20), default="open")

    tier_data_json:  Mapped[str] = mapped_column(Text, default="{}")
    provenance_json: Mapped[str] = mapped_column(Text, default="{}")

    created_at: Mapped[datetime]       = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime]       = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped["ProjectRow | None"] = relationship(
        "ProjectRow", back_populates="memory_objects"
    )

    __table_args__ = (
        Index("ix_memory_project_tier",     "project_id", "tier"),
        Index("ix_memory_project_category", "project_id", "category"),
        Index("ix_memory_project_status",   "project_id", "status"),
        Index("ix_memory_confidence",       "confidence"),
        Index("ix_memory_scope",            "scope"),
    )


# ---------------------------------------------------------------------------
# Memory candidates
# ---------------------------------------------------------------------------


class MemoryCandidateRow(Base):
    __tablename__ = "memory_candidates"

    candidate_id:        Mapped[str]        = mapped_column(String(50), primary_key=True)
    submitting_agent_id: Mapped[str]        = mapped_column(String(50), nullable=False)
    task_id:             Mapped[str]        = mapped_column(String(50), nullable=False)
    project_id:          Mapped[str]        = mapped_column(String(50), nullable=False, index=True)
    proposed_category:   Mapped[str]        = mapped_column(String(40), nullable=False)
    proposed_tier:       Mapped[str | None] = mapped_column(String(30), nullable=True)
    title:               Mapped[str]        = mapped_column(String(300), nullable=False)
    content:             Mapped[str]        = mapped_column(Text, nullable=False)
    tags_json:           Mapped[str]        = mapped_column(Text, default="[]")
    source_refs_json:    Mapped[str]        = mapped_column(Text, default="[]")
    confidence:          Mapped[float]      = mapped_column(Float, default=0.5)
    submitted_at:        Mapped[datetime]   = mapped_column(DateTime(timezone=True), default=_now)
    disposition:         Mapped[str | None] = mapped_column(String(20), nullable=True)
    resulting_memory_id: Mapped[str | None] = mapped_column(String(50), nullable=True)

    __table_args__ = (
        Index("ix_candidates_project_disposition", "project_id", "disposition"),
    )
