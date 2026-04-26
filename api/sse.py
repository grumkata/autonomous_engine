"""
api/sse.py — Server-Sent Events for real-time project monitoring.

GET /projects/{project_id}/stream

The UI subscribes to this endpoint and receives task-status snapshots
every 2 seconds while the orchestration engine is running. The client
uses these events to update the DAG and task list without full page reloads.

Event format:
    data: {"project_id": "...", "status": "...", "tasks": [...], "ts": "..."}
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from db.engine import session_factory
from db.mappers import row_to_task
from db.tables import ProjectRow, TaskRow

router = APIRouter(tags=["sse"])

_POLL_INTERVAL_S = 2.0


@router.get(
    "/projects/{project_id}/stream",
    summary="Live task-status stream (SSE)",
)
async def project_stream(project_id: str):
    """
    Subscribe to live status updates for a project.

    Yields a snapshot every 2 s. The stream closes automatically
    when the project reaches a terminal state (review, archived, closure).
    """

    async def _generator():
        terminal_statuses = {"review", "closure", "archived", "retrospective"}
        consecutive_errors = 0

        while True:
            try:
                async with session_factory() as db:
                    proj_row = await db.get(ProjectRow, project_id)
                    if not proj_row:
                        yield {
                            "event": "error",
                            "data": json.dumps({"detail": "project not found"}),
                        }
                        return

                    task_rows = (
                        await db.execute(
                            select(TaskRow).where(TaskRow.project_id == project_id)
                        )
                    ).scalars().all()

                tasks = [row_to_task(r) for r in task_rows]

                # Build lightweight task snapshots for the UI
                task_snapshots = [
                    {
                        "task_id": t.task_id,
                        "title": t.title,
                        "status": t.status,
                        "priority": t.priority,
                        "owner_department": t.owner_department,
                        "depth_level": t.depth_level,
                        "attempt_count": t.attempt_count,
                        "type": t.type,
                        "parent_task_id": t.parent_task_id,
                    }
                    for t in tasks
                ]

                status_counts: dict[str, int] = {}
                for t in tasks:
                    status_counts[t.status] = status_counts.get(t.status, 0) + 1

                payload = {
                    "project_id": project_id,
                    "project_status": proj_row.status,
                    "tasks": task_snapshots,
                    "status_counts": status_counts,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }

                yield {"event": "snapshot", "data": json.dumps(payload)}
                consecutive_errors = 0

                # Stop streaming once the engine has finished
                if proj_row.status in terminal_statuses:
                    yield {
                        "event": "done",
                        "data": json.dumps({"final_status": proj_row.status}),
                    }
                    return

            except Exception as exc:
                consecutive_errors += 1
                yield {
                    "event": "error",
                    "data": json.dumps({"detail": str(exc)}),
                }
                if consecutive_errors >= 5:
                    return

            await asyncio.sleep(_POLL_INTERVAL_S)

    return EventSourceResponse(_generator())
