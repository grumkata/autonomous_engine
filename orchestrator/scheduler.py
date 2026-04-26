"""
Task scheduler — selects the next batch of READY tasks to execute.

Priority order: critical → high → medium → low
Tiebreakers:    depth_level ASC (shallower tasks first), created_at ASC

The scheduler is purely functional — it reads state and returns decisions,
never writing to the DB directly. State transitions are the engine's job.
"""

from __future__ import annotations

from models.task import Task, TaskEdge, TaskStatus


# ---------------------------------------------------------------------------
# Ready-task selection
# ---------------------------------------------------------------------------


def select_ready_tasks(
    all_tasks: list[Task],
    running_task_ids: set[str],
    max_slots: int,
) -> list[Task]:
    """
    Return up to (max_slots - len(running_task_ids)) tasks to dispatch.

    Only picks tasks that are:
    - status == READY
    - not already in running_task_ids

    Sorted by priority (critical=0, low=3), then depth_level, then created_at.
    """
    available_slots = max_slots - len(running_task_ids)
    if available_slots <= 0:
        return []

    candidates = [
        t for t in all_tasks
        if t.status == TaskStatus.READY and t.task_id not in running_task_ids
    ]

    candidates.sort(key=lambda t: (
        t.priority.as_int(),   # 0=critical, 3=low
        t.depth_level,         # shallower (L0/L1) first
        t.created_at,          # older tasks first within same priority
    ))

    return candidates[:available_slots]


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


def compute_ready_task_ids(
    all_tasks: list[Task],
    edges: list[TaskEdge],
) -> list[str]:
    """
    Return task IDs that should transition PLANNED → READY.

    A PLANNED task becomes READY when every upstream task in `edges`
    has status == APPROVED.

    Called by the engine after each result batch to unlock downstream work.
    """
    approved_ids: set[str] = {
        t.task_id for t in all_tasks if t.status == TaskStatus.APPROVED
    }

    # Map downstream_id → [list of upstream_ids it needs]
    upstream_map: dict[str, list[str]] = {}
    for edge in edges:
        upstream_map.setdefault(edge.downstream_task_id, []).append(edge.upstream_task_id)

    newly_ready: list[str] = []
    for task in all_tasks:
        if task.status != TaskStatus.PLANNED:
            continue
        required_upstreams = upstream_map.get(task.task_id, [])
        # A task with no dependencies is immediately eligible
        if all(uid in approved_ids for uid in required_upstreams):
            newly_ready.append(task.task_id)

    return newly_ready


# ---------------------------------------------------------------------------
# Graph health checks
# ---------------------------------------------------------------------------


def is_project_complete(tasks: list[Task]) -> bool:
    """True when every task is in a terminal state."""
    if not tasks:
        return False
    terminal = {TaskStatus.APPROVED, TaskStatus.FAILED, TaskStatus.ARCHIVED, TaskStatus.BLOCKED}
    return all(t.status in terminal for t in tasks)


def detect_deadlock(
    tasks: list[Task],
    running_task_ids: set[str],
) -> bool:
    """
    Returns True if there are non-terminal tasks but nothing is running or READY.
    This indicates a stuck graph (cycle, missing edge, or all-blocked scenario).
    """
    terminal = {TaskStatus.APPROVED, TaskStatus.FAILED, TaskStatus.ARCHIVED, TaskStatus.BLOCKED}
    non_terminal = [t for t in tasks if t.status not in terminal]
    if not non_terminal:
        return False  # project is actually done

    ready_or_running = any(
        t.status == TaskStatus.READY or t.task_id in running_task_ids
        for t in non_terminal
    )
    return not ready_or_running
