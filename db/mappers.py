"""
db/mappers.py — single source of truth for ORM ↔ domain-model conversion.

Phase 3: workspace mappers
Phase 4: task mapper handles required_skill_ids
Phase 6: task mapper handles human_notes; project mapper handles frozen/notes
"""

from __future__ import annotations

import json

from db.tables import CheckpointRow, ProjectRow, SkillRow, TaskEdgeRow, TaskRow, WorkspaceRow
from models.project import Constraint, Deliverable, Goal, Project, ProjectBudget
from models.checkpoint import Checkpoint
from models.skill import Skill
from models.task import Task, TaskAttempt, TaskBudget, TaskEdge
from models.workspace import Workspace, WorkspaceStatus


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


def row_to_project(row: ProjectRow) -> Project:
    return Project(
        project_id=row.project_id,
        title=row.title,
        description=row.description,
        type=row.type,
        status=row.status,
        raw_intake=row.raw_intake,
        goals=[Goal(**g) for g in json.loads(row.goals_json)],
        constraints=[Constraint(**c) for c in json.loads(row.constraints_json)],
        deliverables=[Deliverable(**d) for d in json.loads(row.deliverables_json)],
        budget=ProjectBudget(**json.loads(row.budget_json)),
        tags=json.loads(row.tags_json),
        insight_ids=json.loads(row.insight_ids_json),
        parent_project_id=row.parent_project_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def project_to_row(project: Project) -> ProjectRow:
    return ProjectRow(
        project_id=project.project_id,
        title=project.title,
        description=project.description,
        type=project.type.value,
        status=project.status.value,
        raw_intake=project.raw_intake,
        goals_json=json.dumps([g.model_dump(mode="json") for g in project.goals]),
        constraints_json=json.dumps(
            [c.model_dump(mode="json") for c in project.constraints]
        ),
        deliverables_json=json.dumps(
            [d.model_dump(mode="json") for d in project.deliverables]
        ),
        budget_json=json.dumps(project.budget.model_dump(mode="json")),
        tags_json=json.dumps(project.tags),
        insight_ids_json=json.dumps(project.insight_ids),
        parent_project_id=project.parent_project_id,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def row_to_task(row: TaskRow) -> Task:
    return Task(
        task_id=row.task_id,
        project_id=row.project_id,
        type=row.type,
        title=row.title,
        description=row.description,
        status=row.status,
        priority=row.priority,
        owner_department=row.owner_department,
        parent_task_id=row.parent_task_id,
        depth_level=row.depth_level,
        assigned_agent_ids=json.loads(row.assigned_agent_ids_json),
        input_artifact_ids=json.loads(row.input_artifact_ids_json),
        expected_output_types=json.loads(row.expected_output_types_json),
        output_artifact_ids=json.loads(row.output_artifact_ids_json),
        validation_criteria=json.loads(row.validation_criteria_json),
        budget=TaskBudget(**json.loads(row.budget_json)),
        attempts=[TaskAttempt(**a) for a in json.loads(row.attempts_json)],
        attempt_count=row.attempt_count,
        quality_threshold=row.quality_threshold,
        requires_human_approval=row.requires_human_approval,
        # Phase 4
        required_skill_ids=json.loads(row.required_skill_ids_json),
        # Phase 6
        human_notes=json.loads(row.human_notes_json),
        created_at=row.created_at,
        updated_at=row.updated_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
    )


def task_to_row(task: Task) -> TaskRow:
    return TaskRow(
        task_id=task.task_id,
        project_id=task.project_id,
        type=task.type.value,
        title=task.title,
        description=task.description,
        status=task.status.value,
        priority=task.priority.value,
        owner_department=task.owner_department.value,
        parent_task_id=task.parent_task_id,
        depth_level=task.depth_level,
        validation_criteria_json=json.dumps(task.validation_criteria),
        expected_output_types_json=json.dumps(task.expected_output_types),
        input_artifact_ids_json=json.dumps(task.input_artifact_ids),
        output_artifact_ids_json=json.dumps(task.output_artifact_ids),
        assigned_agent_ids_json=json.dumps(task.assigned_agent_ids),
        budget_json=json.dumps(task.budget.model_dump(mode="json")),
        attempts_json=json.dumps([a.model_dump(mode="json") for a in task.attempts]),
        attempt_count=task.attempt_count,
        quality_threshold=task.quality_threshold,
        requires_human_approval=task.requires_human_approval,
        # Phase 4
        required_skill_ids_json=json.dumps(task.required_skill_ids),
        # Phase 6
        human_notes_json=json.dumps(task.human_notes),
        created_at=task.created_at,
        updated_at=task.updated_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
    )


# ---------------------------------------------------------------------------
# Task edge
# ---------------------------------------------------------------------------


def row_to_edge(row: TaskEdgeRow) -> TaskEdge:
    return TaskEdge(
        edge_id=row.edge_id,
        project_id=row.project_id,
        upstream_task_id=row.upstream_task_id,
        downstream_task_id=row.downstream_task_id,
        edge_type=row.edge_type,
        created_at=row.created_at,
    )


# ---------------------------------------------------------------------------
# Workspace  (Phase 3)
# ---------------------------------------------------------------------------


def row_to_workspace(row: WorkspaceRow) -> Workspace:
    return Workspace(
        workspace_id=row.workspace_id,
        project_id=row.project_id,
        department=row.department,
        status=WorkspaceStatus(row.status),
        artifact_ids=json.loads(row.artifact_ids_json),
        task_ids_processed=json.loads(row.task_ids_json),
        scratch_pad=row.scratch_pad or "",
        created_at=row.created_at,
        updated_at=row.updated_at,
        archived_at=row.archived_at,
    )


def workspace_to_row(workspace: Workspace) -> WorkspaceRow:
    return WorkspaceRow(
        workspace_id=workspace.workspace_id,
        project_id=workspace.project_id,
        department=workspace.department,
        status=workspace.status.value,
        artifact_ids_json=json.dumps(workspace.artifact_ids),
        task_ids_json=json.dumps(workspace.task_ids_processed),
        scratch_pad=workspace.scratch_pad,
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
        archived_at=workspace.archived_at,
    )


# ---------------------------------------------------------------------------
# Skill  (Phase 4)
# ---------------------------------------------------------------------------


def row_to_skill(row: SkillRow) -> Skill:
    return Skill(
        skill_id=row.skill_id,
        name=row.name,
        version=row.version,
        description=row.description,
        departments=json.loads(row.departments_json),
        content_md=row.content_md,
        tool_ids=json.loads(row.tool_ids_json),
        source=row.source,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def skill_to_row(skill: Skill) -> SkillRow:
    return SkillRow(
        skill_id=skill.skill_id,
        name=skill.name,
        version=skill.version,
        description=skill.description,
        departments_json=json.dumps(skill.departments),
        content_md=skill.content_md,
        tool_ids_json=json.dumps(skill.tool_ids),
        source=skill.source,
        created_at=skill.created_at,
        updated_at=skill.updated_at,
    )


# ---------------------------------------------------------------------------
# Checkpoint  (Phase 7)
# ---------------------------------------------------------------------------


def row_to_checkpoint(row: CheckpointRow) -> Checkpoint:
    data = json.loads(row.snapshot_json)
    return Checkpoint(
        checkpoint_id=row.checkpoint_id,
        project_id=row.project_id,
        trigger_reason=row.trigger_reason,
        project_state=data.get("project_state", {}),
        task_states=data.get("task_states", []),
        edge_states=data.get("edge_states", []),
        workspace_states=data.get("workspace_states", []),
        task_count=row.task_count,
        approved_count=row.approved_count,
        in_progress_count=row.in_progress_count,
        iteration=row.iteration,
        created_at=row.created_at,
    )


def checkpoint_to_row(ckp: Checkpoint) -> CheckpointRow:
    snapshot = {
        "project_state":    ckp.project_state,
        "task_states":      ckp.task_states,
        "edge_states":      ckp.edge_states,
        "workspace_states": ckp.workspace_states,
    }
    return CheckpointRow(
        checkpoint_id=ckp.checkpoint_id,
        project_id=ckp.project_id,
        trigger_reason=ckp.trigger_reason,
        snapshot_json=json.dumps(snapshot, default=str),
        task_count=ckp.task_count,
        approved_count=ckp.approved_count,
        in_progress_count=ckp.in_progress_count,
        iteration=ckp.iteration,
        created_at=ckp.created_at,
    )
