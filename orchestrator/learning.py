"""
orchestrator/learning.py — Continuous Learning Pipeline (Phase 9).

Learning is continuous, not a post-mortem.  An insight is extracted the
moment something works well or fails badly — not only when a project closes.

When learning fires
-------------------
  on_task_approved()        — every approved task (heuristic + async LLM)
  on_task_failed()          — every permanently failed task (heuristic)
  on_retry_success()        — when attempt N>1 passes (captures "what fixed it")
  run_project_retrospective() — full synthesis at project closure
  promote_insights()        — elevate draft insights across projects

Three extraction modes
----------------------
  Heuristic (zero latency, always runs)
      Pattern-match on validation scores, skill combos, dept/type pairs,
      failure reasons.  Produces candidates immediately on the hot path.

  LLM narrative (async background, free unlimited model)
      Synthesises richer patterns in natural language.  Runs as a background
      asyncio task so it never slows the engine.

  Retrospective (full LLM synthesis, triggered at project close)
      Looks at the whole episode: what was planned vs what happened, where
      quality was high vs low, what failure patterns repeated, what the
      successful retry strategies were.

Promotion rules (spec §9 step 4-7)
-----------------------------------
  draft    → verified  : confidence >= 0.75 AND not flagged as overgeneralised
  verified → canonical : support_count >= 3 (same pattern in 3+ projects)

All insights flow through submit_candidates() on the MemoryManager so
deduplication, ChromaDB indexing, and tier routing are handled centrally.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from orchestrator.audit import AuditEventType, AuditSeverity, audit

log = structlog.get_logger(__name__)

# Minimum score delta between attempts to consider a retry "instructive"
_RETRY_MIN_GAIN = 0.08

# Validation score threshold considered "high quality"
_HIGH_QUALITY_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Episode record — what happened on a single task
# ---------------------------------------------------------------------------


@dataclass
class TaskEpisode:
    """Structured summary of a task's execution history — input to pattern extraction."""
    task_id:         str
    project_id:      str
    task_type:       str
    department:      str
    title:           str
    description:     str
    skills_used:     list[str]
    attempt_count:   int
    final_status:    str          # "approved" | "failed" | "blocked"
    composite_score: float | None
    quality_threshold: float
    failure_reasons: list[str]    # one per failed attempt
    corrective_feedbacks: list[str]
    validation_issues: list[str]
    latency_ms:      int
    tokens_used:     int


# ---------------------------------------------------------------------------
# LearningService
# ---------------------------------------------------------------------------


class LearningService:
    """
    Singleton that extracts and promotes insights from task outcomes.

    All public methods are non-fatal — learning failures never interrupt
    the orchestration engine.
    """

    # ------------------------------------------------------------------ #
    # Continuous hooks — called from engine on every outcome              #
    # ------------------------------------------------------------------ #

    async def on_task_approved(
        self,
        episode: TaskEpisode,
    ) -> None:
        """
        Extract learning from a successful task.
        Runs heuristic extraction immediately, LLM extraction in background.
        """
        try:
            candidates = self._heuristic_success(episode)
            if candidates:
                await self._submit(candidates, episode.project_id, episode.task_id, "learning")

            # LLM extraction in background — never awaited on the hot path
            if episode.composite_score and episode.composite_score >= _HIGH_QUALITY_THRESHOLD:
                asyncio.create_task(
                    self._llm_success_extract(episode),
                    name=f"learn_success_{episode.task_id}",
                )
        except Exception as exc:
            log.warning("learning.on_approved_failed", task_id=episode.task_id, error=str(exc))

    async def on_task_failed(
        self,
        episode: TaskEpisode,
    ) -> None:
        """
        Extract failure patterns from a permanently failed task.
        Heuristic only — LLM synthesis happens in the retrospective.
        """
        try:
            candidates = self._heuristic_failure(episode)
            if candidates:
                await self._submit(candidates, episode.project_id, episode.task_id, "learning")

            audit(
                AuditEventType.MEMORY_ACCEPTED,
                project_id=episode.project_id,
                task_id=episode.task_id,
                actor_type="learning",
                severity=AuditSeverity.DEBUG,
                pattern_type="failure",
                candidates_extracted=len(candidates),
            )
        except Exception as exc:
            log.warning("learning.on_failed_failed", task_id=episode.task_id, error=str(exc))

    async def on_retry_success(
        self,
        episode: TaskEpisode,
        scores_by_attempt: list[float],
    ) -> None:
        """
        Capture what changed between a failing attempt and a passing one.
        This is some of the richest signal in the system.
        """
        if len(scores_by_attempt) < 2:
            return
        try:
            gain = scores_by_attempt[-1] - scores_by_attempt[-2]
            if gain < _RETRY_MIN_GAIN:
                return  # Not instructive enough to record

            correction_that_worked = (
                episode.corrective_feedbacks[-1]
                if episode.corrective_feedbacks else ""
            )
            candidates = [{
                "category": "insight",
                "title": (
                    f"Retry success: {episode.department}/{episode.task_type} "
                    f"+{gain:.0%} after correction"
                ),
                "content": (
                    f"Task type: {episode.task_type} | Department: {episode.department}\n"
                    f"Skills: {', '.join(episode.skills_used) or 'none'}\n"
                    f"Attempt {len(scores_by_attempt)-1} score: {scores_by_attempt[-2]:.0%} "
                    f"→ Attempt {len(scores_by_attempt)} score: {scores_by_attempt[-1]:.0%}\n"
                    f"Correction that worked:\n{correction_that_worked[:600]}\n"
                    f"Prior failure reasons: {'; '.join(episode.failure_reasons[:-1])[:300]}"
                ),
                "confidence": min(0.9, 0.6 + gain),
                "tags": [
                    "retry_success", episode.department, episode.task_type,
                    *episode.skills_used[:2],
                ],
            }]
            await self._submit(candidates, episode.project_id, episode.task_id, "learning")
            log.info(
                "learning.retry_pattern",
                task_id=episode.task_id,
                gain=f"{gain:.0%}",
                correction_preview=correction_that_worked[:80],
            )
        except Exception as exc:
            log.warning("learning.retry_failed", task_id=episode.task_id, error=str(exc))

    # ------------------------------------------------------------------ #
    # Project retrospective — full synthesis at project closure           #
    # ------------------------------------------------------------------ #

    async def run_project_retrospective(self, project_id: str) -> dict[str, Any]:
        """
        Full learning synthesis for a completed project.

        Steps (spec §9):
          1. Collect all task episodes for the project
          2. Group by outcome, department, task type
          3. LLM narrative synthesis across all episodes
          4. Submit insights with cross-project scope
          5. Return summary for the API response

        Returns a summary dict with counts and top insights.
        """
        try:
            episodes = await self._collect_episodes(project_id)
            if not episodes:
                return {"project_id": project_id, "episodes": 0, "insights_extracted": 0}

            # Group episodes
            approved = [e for e in episodes if e.final_status == "approved"]
            failed   = [e for e in episodes if e.final_status == "failed"]
            retried  = [e for e in episodes if e.attempt_count > 1 and e.final_status == "approved"]

            # Heuristic cross-episode patterns
            candidates: list[dict] = []
            candidates.extend(self._cross_episode_patterns(approved, failed, project_id))

            # LLM synthesis (awaited here — retrospective is not on the hot path)
            llm_candidates = await self._llm_retrospective(project_id, episodes)
            candidates.extend(llm_candidates)

            # Submit all
            submitted_ids: list[str] = []
            if candidates:
                submitted_ids = await self._submit(candidates, project_id, "retrospective", "learning")

            # Promote any eligible draft insights
            promoted = await self._promote_insights()

            summary = {
                "project_id":        project_id,
                "episodes":          len(episodes),
                "approved":          len(approved),
                "failed":            len(failed),
                "retried_to_success": len(retried),
                "insights_extracted": len(submitted_ids),
                "insights_promoted":  promoted,
            }

            audit(
                "retrospective_completed",
                project_id=project_id,
                actor_type="learning",
                **summary,
            )
            log.info("learning.retrospective_complete", **summary)
            return summary

        except Exception as exc:
            log.error("learning.retrospective_failed", project_id=project_id, error=str(exc))
            return {"project_id": project_id, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Heuristic extractors                                                #
    # ------------------------------------------------------------------ #

    def _heuristic_success(self, ep: TaskEpisode) -> list[dict]:
        """Zero-latency pattern extraction from a single approved task."""
        candidates: list[dict] = []
        score = ep.composite_score or 0.0

        # Pattern 1: High-quality first attempt (no retries needed)
        if ep.attempt_count == 1 and score >= _HIGH_QUALITY_THRESHOLD:
            candidates.append({
                "category": "insight",
                "title": f"First-attempt success: {ep.department}/{ep.task_type}",
                "content": (
                    f"Task type '{ep.task_type}' handled by '{ep.department}' "
                    f"succeeded on first attempt with score {score:.0%}.\n"
                    f"Skills: {', '.join(ep.skills_used) or 'none'}.\n"
                    f"Threshold: {ep.quality_threshold:.0%}."
                ),
                "confidence": min(0.88, score),
                "tags": ["first_attempt_success", ep.department, ep.task_type,
                         *ep.skills_used[:2]],
            })

        # Pattern 2: Skill combination that worked well
        if ep.skills_used and score >= 0.78:
            candidates.append({
                "category": "insight",
                "title": (
                    f"Effective skills for {ep.task_type}: "
                    f"{', '.join(ep.skills_used[:2])}"
                ),
                "content": (
                    f"Skills [{', '.join(ep.skills_used)}] produced score {score:.0%} "
                    f"on task type '{ep.task_type}' in department '{ep.department}'.\n"
                    f"Attempts needed: {ep.attempt_count}."
                ),
                "confidence": 0.70 + (score - 0.78) * 0.5,
                "tags": ["skill_effectiveness", ep.task_type, *ep.skills_used],
            })

        return candidates

    def _heuristic_failure(self, ep: TaskEpisode) -> list[dict]:
        """Extract failure patterns for permanent task failures."""
        candidates: list[dict] = []
        if not ep.failure_reasons:
            return candidates

        # Deduplicate failure reasons — repeated patterns are stronger signal
        unique_reasons = list(dict.fromkeys(ep.failure_reasons))

        candidates.append({
            "category": "failure_reason",
            "title": f"Failure pattern: {ep.department}/{ep.task_type} after {ep.attempt_count} attempts",
            "content": (
                f"Task type '{ep.task_type}' in department '{ep.department}' "
                f"failed after {ep.attempt_count} attempts.\n"
                f"Skills tried: {', '.join(ep.skills_used) or 'none'}.\n"
                f"Failure reasons:\n"
                + "\n".join(f"  [{i+1}] {r[:200]}" for i, r in enumerate(unique_reasons))
                + f"\n\nValidation issues: {'; '.join(ep.validation_issues[:3])}"
            ),
            "confidence": 0.80,
            "tags": ["permanent_failure", ep.department, ep.task_type, *ep.skills_used[:2]],
        })

        return candidates

    def _cross_episode_patterns(
        self,
        approved: list[TaskEpisode],
        failed: list[TaskEpisode],
        project_id: str,
    ) -> list[dict]:
        """Patterns that emerge from multiple episodes in the same project."""
        candidates: list[dict] = []

        # Dept performance summary
        dept_scores: dict[str, list[float]] = {}
        for ep in approved:
            if ep.composite_score:
                dept_scores.setdefault(ep.department, []).append(ep.composite_score)

        for dept, scores in dept_scores.items():
            if len(scores) >= 3:
                avg = sum(scores) / len(scores)
                candidates.append({
                    "category": "insight",
                    "title": f"Department performance: {dept} avg {avg:.0%} ({len(scores)} tasks)",
                    "content": (
                        f"In project {project_id}, department '{dept}' completed "
                        f"{len(scores)} tasks with average validation score {avg:.0%}. "
                        f"Range: {min(scores):.0%} – {max(scores):.0%}."
                    ),
                    "confidence": 0.72 if avg >= 0.80 else 0.65,
                    "tags": ["dept_performance", dept, project_id],
                })

        # Task types that consistently failed
        failed_types: dict[str, int] = {}
        for ep in failed:
            failed_types[ep.task_type] = failed_types.get(ep.task_type, 0) + 1

        for task_type, count in failed_types.items():
            if count >= 2:
                candidates.append({
                    "category": "insight",
                    "title": f"Repeated failure type in project: {task_type} ({count}x)",
                    "content": (
                        f"Task type '{task_type}' failed {count} times in project {project_id}. "
                        f"Consider: higher quality thresholds, different department assignment, "
                        f"or additional skill packs for this type."
                    ),
                    "confidence": 0.78,
                    "tags": ["repeated_failure_type", task_type, project_id],
                })

        return candidates

    # ------------------------------------------------------------------ #
    # LLM extractors (background, never block engine)                     #
    # ------------------------------------------------------------------ #

    async def _llm_success_extract(self, ep: TaskEpisode) -> None:
        """
        Background LLM call to extract richer narrative insight from a
        high-quality approval. Uses free model — non-fatal on error.
        """
        try:
            from llm.client import get_client
            from llm.schemas import Message
            client = get_client()

            prompt = (
                f"A task just passed validation with a high quality score.\n\n"
                f"Task type: {ep.task_type}\n"
                f"Department: {ep.department}\n"
                f"Skills used: {', '.join(ep.skills_used) or 'none'}\n"
                f"Attempts needed: {ep.attempt_count}\n"
                f"Score: {ep.composite_score:.0%} (threshold: {ep.quality_threshold:.0%})\n"
                f"Task title: {ep.title}\n\n"
                f"In 2-3 sentences, identify ONE generalizable insight about what made this "
                f"task succeed. Focus on actionable patterns (department fit, skill match, "
                f"task decomposition quality). Do NOT say 'the task succeeded because it passed'.\n\n"
                f"Respond ONLY with JSON: "
                f'{{\"title\": \"<10 word insight title>\", \"content\": \"<2-3 sentences>\", '
                f'"confidence\": <0.0-1.0>, \"tags\": [\"<tag1>\", \"<tag2>\"]}}'
            )

            raw, _ = await client.chat_json(
                [Message(role="user", content=prompt)],
                max_tokens=300,
                retry_limit=1,
            )

            candidate = {
                "category": "insight",
                "title": str(raw.get("title", ""))[:150],
                "content": str(raw.get("content", "")),
                "confidence": max(0.5, min(0.95, float(raw.get("confidence", 0.72)))),
                "tags": raw.get("tags", [ep.department, ep.task_type]) + ["llm_extracted"],
            }

            if candidate["title"] and candidate["content"]:
                await self._submit([candidate], ep.project_id, ep.task_id, "learning_llm")
                log.debug("learning.llm_insight_extracted", task_id=ep.task_id)

        except Exception as exc:
            log.debug("learning.llm_extract_skipped", task_id=ep.task_id, reason=str(exc)[:80])

    async def _llm_retrospective(
        self,
        project_id: str,
        episodes: list[TaskEpisode],
    ) -> list[dict]:
        """
        Full LLM synthesis across all project episodes.
        Returns a list of insight candidates.
        """
        if len(episodes) < 3:
            return []

        try:
            from llm.client import get_client
            from llm.schemas import Message
            client = get_client()

            approved_count = sum(1 for e in episodes if e.final_status == "approved")
            failed_count   = sum(1 for e in episodes if e.final_status == "failed")
            avg_score = (
                sum(e.composite_score for e in episodes if e.composite_score)
                / max(1, sum(1 for e in episodes if e.composite_score))
            )
            dept_counts: dict[str, int] = {}
            for e in episodes:
                dept_counts[e.department] = dept_counts.get(e.department, 0) + 1

            episode_summary = (
                f"Project: {project_id}\n"
                f"Total tasks: {len(episodes)}\n"
                f"Approved: {approved_count} | Failed: {failed_count}\n"
                f"Average validation score: {avg_score:.0%}\n"
                f"Department distribution: {json.dumps(dept_counts)}\n\n"
                f"Failed task types: "
                + ", ".join(e.task_type for e in episodes if e.final_status == "failed")[:300]
                + f"\n\nHigh-scoring tasks (score >= 85%): "
                + ", ".join(
                    f"{e.task_type}/{e.department}"
                    for e in episodes
                    if e.composite_score and e.composite_score >= 0.85
                )[:300]
                + f"\n\nSkills that appeared in approved tasks: "
                + ", ".join(
                    set(s for e in episodes for s in e.skills_used
                        if e.final_status == "approved")
                )[:200]
            )

            prompt = (
                f"Analyse this project's task execution data and extract 3 reusable insights "
                f"that would help FUTURE projects of similar types.\n\n"
                f"{episode_summary}\n\n"
                f"Return ONLY JSON with this exact structure:\n"
                f'{{"insights": [{{'
                f'"title": "<concise insight title>", '
                f'"content": "<2-4 sentences, actionable>", '
                f'"confidence": <0.6-0.95>, '
                f'"tags": ["<tag>"]'
                f'}}]}}'
            )

            raw, _ = await client.chat_json(
                [Message(role="user", content=prompt)],
                max_tokens=800,
                retry_limit=2,
            )

            candidates: list[dict] = []
            for item in raw.get("insights", []):
                title = str(item.get("title", ""))[:150]
                content = str(item.get("content", ""))
                if title and content and len(content) > 30:
                    candidates.append({
                        "category": "insight",
                        "title": title,
                        "content": content,
                        "confidence": max(0.60, min(0.95, float(item.get("confidence", 0.72)))),
                        "tags": item.get("tags", []) + ["retrospective", "cross_project"],
                    })

            log.info(
                "learning.retrospective_llm",
                project_id=project_id,
                insights=len(candidates),
            )
            return candidates

        except Exception as exc:
            log.warning("learning.retrospective_llm_failed", project_id=project_id, error=str(exc))
            return []

    # ------------------------------------------------------------------ #
    # Insight promotion                                                   #
    # ------------------------------------------------------------------ #

    async def _promote_insights(self) -> int:
        """
        Promote insights that have accumulated support across projects.

        draft    → verified  : confidence >= 0.75
        verified → canonical : support_count >= 3 (title seen in 3+ projects)

        Returns the number of promotions made.
        """
        from sqlalchemy import func, select, update
        from db.engine import session_factory
        from db.tables import MemoryObjectRow

        promoted = 0
        try:
            async with session_factory() as db:
                # Promote draft → verified
                draft_rows = (
                    await db.execute(
                        select(MemoryObjectRow).where(
                            MemoryObjectRow.tier == "insight",
                            MemoryObjectRow.status == "draft",
                            MemoryObjectRow.confidence >= 0.75,
                        )
                    )
                ).scalars().all()

                for row in draft_rows:
                    row.status = "verified"
                    row.updated_at = datetime.now(timezone.utc)
                    promoted += 1
                    log.debug("learning.promoted_to_verified", memory_id=row.memory_id)

                # Promote verified → canonical if seen 3+ times (by title similarity)
                # Use exact title matching as a proxy for "same pattern"
                verified_rows = (
                    await db.execute(
                        select(MemoryObjectRow).where(
                            MemoryObjectRow.tier == "insight",
                            MemoryObjectRow.status == "verified",
                        )
                    )
                ).scalars().all()

                title_counts: dict[str, list[str]] = {}
                for row in verified_rows:
                    key = row.title[:60].lower().strip()
                    title_counts.setdefault(key, []).append(row.memory_id)

                for title_key, ids in title_counts.items():
                    if len(ids) >= 3:
                        for mid in ids:
                            row = await db.get(MemoryObjectRow, mid)
                            if row and row.status == "verified":
                                row.status = "canonical"
                                row.scope  = "global"
                                row.write_policy = "restricted"
                                row.updated_at = datetime.now(timezone.utc)
                                promoted += 1
                                log.info(
                                    "learning.promoted_to_canonical",
                                    memory_id=mid,
                                    title_key=title_key,
                                )
        except Exception as exc:
            log.warning("learning.promote_failed", error=str(exc))

        return promoted

    # ------------------------------------------------------------------ #
    # Episode collection                                                  #
    # ------------------------------------------------------------------ #

    async def _collect_episodes(self, project_id: str) -> list[TaskEpisode]:
        """Load all task history for a project into TaskEpisode structs."""
        from sqlalchemy import select
        from db.engine import session_factory
        from db.tables import TaskRow
        import json

        async with session_factory() as db:
            rows = (
                await db.execute(
                    select(TaskRow).where(TaskRow.project_id == project_id)
                )
            ).scalars().all()

        episodes: list[TaskEpisode] = []
        for row in rows:
            attempts = json.loads(row.attempts_json)
            failure_reasons = [
                a.get("failure_reason", "")
                for a in attempts
                if a.get("failure_reason")
            ]
            corrective = [
                a.get("corrective_feedback", "")
                for a in attempts
                if a.get("corrective_feedback")
            ]
            scores = [
                a.get("composite_score")
                for a in attempts
                if a.get("composite_score") is not None
            ]
            final_score = scores[-1] if scores else None

            episodes.append(TaskEpisode(
                task_id=row.task_id,
                project_id=project_id,
                task_type=row.type,
                department=row.owner_department,
                title=row.title,
                description=row.description[:200],
                skills_used=json.loads(row.required_skill_ids_json),
                attempt_count=row.attempt_count,
                final_status=row.status,
                composite_score=final_score,
                quality_threshold=row.quality_threshold,
                failure_reasons=failure_reasons,
                corrective_feedbacks=corrective,
                validation_issues=[],
                latency_ms=0,
                tokens_used=0,
            ))

        return episodes

    # ------------------------------------------------------------------ #
    # Internal submit wrapper                                             #
    # ------------------------------------------------------------------ #

    async def _submit(
        self,
        candidates: list[dict],
        project_id: str,
        task_id: str,
        agent_role: str,
    ) -> list[str]:
        """Submit candidates through the MemoryManager — handles dedup + indexing."""
        try:
            from memory.manager import get_memory_manager
            mgr = get_memory_manager()
            ids = await mgr.submit_candidates(
                candidates=candidates,
                project_id=project_id,
                task_id=task_id,
                agent_role=agent_role,
            )
            if ids:
                log.info(
                    "learning.insights_submitted",
                    count=len(ids),
                    project_id=project_id,
                    task_id=task_id,
                )
            return ids
        except Exception as exc:
            log.warning("learning.submit_failed", error=str(exc))
            return []


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_service: LearningService | None = None


def get_learning_service() -> LearningService:
    global _service
    if _service is None:
        _service = LearningService()
    return _service


# ---------------------------------------------------------------------------
# Convenience builder — turns engine data into a TaskEpisode
# ---------------------------------------------------------------------------


def build_episode(
    task: Any,
    result: Any,
) -> TaskEpisode:
    """
    Build a TaskEpisode from an engine Task + RunResult.
    Called from engine_orchestrator after every terminal outcome.
    """
    attempts = task.attempts or []
    failure_reasons = [
        a.corrective_feedback or a.validation_result or ""
        for a in attempts
        if a.validation_result in ("fail", "revise")
    ]
    corrective = [
        a.corrective_feedback or ""
        for a in attempts
        if a.corrective_feedback
    ]

    vr = result.validation_report
    return TaskEpisode(
        task_id=task.task_id,
        project_id=task.project_id,
        task_type=task.type.value,
        department=task.owner_department.value,
        title=task.title,
        description=task.description[:200],
        skills_used=getattr(task, "required_skill_ids", []),
        attempt_count=task.attempt_count,
        final_status=result.status,
        composite_score=vr.composite_score if vr else None,
        quality_threshold=task.quality_threshold,
        failure_reasons=failure_reasons,
        corrective_feedbacks=corrective,
        validation_issues=vr.issues[:5] if vr else [],
        latency_ms=result.latency_ms,
        tokens_used=0,
    )
