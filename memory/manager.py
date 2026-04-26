"""
Memory Manager — gatekeeper between agents and persistent memory.

BUG FIXES vs original:
  1. Cross-project insights are now retrievable by new projects. The original
     code filtered insight_memory with where={"project_id": project_id}, which
     silently excluded all cross-project insights (scope=CROSS_PROJECT,
     project_id=None). Insights are now queried without a project_id filter.
  2. Basic deduplication: before accepting a candidate, we check if a memory
     with a similar title already exists for this project/task. This prevents
     retry runs from flooding the store with repeated near-identical submissions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import structlog

from core.ids import prefixed_id
from db.engine import session_factory
from db.tables import MemoryCandidateRow, MemoryObjectRow
from llm.schemas import MemoryExcerpt
from models.memory import (
    MemoryCategory,
    MemoryScope,
    MemoryStatus,
    MemoryTier,
    Visibility,
    WritePolicy,
)

from orchestrator.audit import AuditEventType, AuditSeverity, audit

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tier + collection mapping tables
# ---------------------------------------------------------------------------

_CATEGORY_TO_TIER: dict[MemoryCategory, MemoryTier] = {
    MemoryCategory.GOAL:               MemoryTier.PROJECT_WORKING,
    MemoryCategory.CONSTRAINT:         MemoryTier.PROJECT_WORKING,
    MemoryCategory.ASSUMPTION:         MemoryTier.PROJECT_WORKING,
    MemoryCategory.RISK:               MemoryTier.PROJECT_WORKING,
    MemoryCategory.RESEARCH_FINDING:   MemoryTier.PROJECT_ARTIFACT,
    MemoryCategory.DECISION:           MemoryTier.PROJECT_ARTIFACT,
    MemoryCategory.DESIGN_RATIONALE:   MemoryTier.PROJECT_ARTIFACT,
    MemoryCategory.ARTIFACT_REFERENCE: MemoryTier.PROJECT_ARTIFACT,
    MemoryCategory.TEST_RESULT:        MemoryTier.PROJECT_ARTIFACT,
    MemoryCategory.MILESTONE:          MemoryTier.PROJECT_ARTIFACT,
    MemoryCategory.FAILURE_REASON:     MemoryTier.FAILURE,
    MemoryCategory.REJECTED_OPTION:    MemoryTier.FAILURE,
    MemoryCategory.INSIGHT:            MemoryTier.INSIGHT,
    MemoryCategory.HUMAN_OVERRIDE:     MemoryTier.CANONICAL,
}

_TIER_TO_COLLECTION: dict[str, str] = {
    MemoryTier.PROJECT_WORKING.value:  "project_working",
    MemoryTier.PROJECT_ARTIFACT.value: "project_artifacts",
    MemoryTier.FAILURE.value:          "failure_memory",
    MemoryTier.INSIGHT.value:          "insight_memory",
    MemoryTier.CANONICAL.value:        "canonical_knowledge",
    MemoryTier.EPHEMERAL.value:        "project_working",
}


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class MemoryManager:
    def __init__(self) -> None:
        self._chroma = None

    def _get_chroma(self):
        if self._chroma is not None:
            return self._chroma
        try:
            import chromadb
            from config import get_settings
            settings = get_settings()
            self._chroma = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        except Exception as exc:
            log.warning("memory.chroma_unavailable", error=str(exc))
        return self._chroma

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    async def submit_candidates(
        self,
        candidates: list[dict[str, Any]],
        project_id: str,
        task_id: str,
        agent_role: str,
    ) -> list[str]:
        accepted_ids: list[str] = []

        for raw in candidates:
            try:
                accepted_id = await self._process_one(raw, project_id, task_id, agent_role)
                if accepted_id:
                    accepted_ids.append(accepted_id)
            except Exception as exc:
                log.warning(
                    "memory.candidate_skipped",
                    error=str(exc),
                    title=str(raw.get("title", ""))[:80],
                )

        log.info(
            "memory.batch_complete",
            project_id=project_id,
            submitted=len(candidates),
            accepted=len(accepted_ids),
        )
        if accepted_ids:
            audit(
                AuditEventType.MEMORY_ACCEPTED,
                project_id=project_id,
                task_id=task_id,
                actor_type="agent",
                actor_id=agent_role,
                severity=AuditSeverity.DEBUG,
                memory_ids=accepted_ids,
                submitted=len(candidates),
                accepted=len(accepted_ids),
            )
        return accepted_ids

    async def _is_duplicate(self, title: str, project_id: str, category: str) -> bool:
        """
        BUG FIX: basic deduplication.
        Returns True if a memory with an identical title already exists for
        this project+category combination. Prevents retry runs from filling
        the store with repeated near-identical items.
        """
        from sqlalchemy import select
        async with session_factory() as db:
            rows = (
                await db.execute(
                    select(MemoryObjectRow)
                    .where(
                        MemoryObjectRow.project_id == project_id,
                        MemoryObjectRow.category == category,
                        MemoryObjectRow.title == title[:300],
                    )
                    .limit(1)
                )
            ).scalars().all()
        return len(rows) > 0

    async def _process_one(
        self,
        raw: dict[str, Any],
        project_id: str,
        task_id: str,
        agent_role: str,
    ) -> str | None:
        try:
            category = MemoryCategory(raw["category"])
        except (KeyError, ValueError):
            log.warning("memory.bad_category", raw_category=raw.get("category"))
            return None

        title = str(raw.get("title", "Untitled"))[:300]
        content = raw.get("content", "")
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
        tags = [str(t).lower().strip() for t in raw.get("tags", [])]

        if not content:
            return None

        # BUG FIX: skip exact-title duplicates for this project
        if await self._is_duplicate(title, project_id, category.value):
            log.debug("memory.duplicate_skipped", title=title[:60], project_id=project_id)
            return None

        tier = _CATEGORY_TO_TIER.get(category, MemoryTier.PROJECT_WORKING)
        scope = (
            MemoryScope.CROSS_PROJECT
            if (category == MemoryCategory.INSIGHT and confidence >= 0.85)
            else MemoryScope.PROJECT
        )

        candidate_id = prefixed_id("cnd")
        memory_id = prefixed_id("mem")
        content_str = content if isinstance(content, str) else json.dumps(content)
        now = datetime.now(timezone.utc)

        # Persist candidate + memory object
        async with session_factory() as db:
            db.add(MemoryCandidateRow(
                candidate_id=candidate_id,
                submitting_agent_id=agent_role,
                task_id=task_id,
                project_id=project_id,
                proposed_category=category.value,
                proposed_tier=tier.value,
                title=title,
                content=content_str,
                tags_json=json.dumps(tags),
                confidence=confidence,
                submitted_at=now,
                disposition="accepted",
                resulting_memory_id=memory_id,
            ))
            db.add(MemoryObjectRow(
                memory_id=memory_id,
                tier=tier.value,
                category=category.value,
                scope=scope.value,
                status=MemoryStatus.DRAFT.value,
                project_id=project_id if scope != MemoryScope.CROSS_PROJECT else None,
                task_id=task_id,
                agent_id=agent_role,
                title=title,
                content=content_str,
                tags_json=json.dumps(tags),
                confidence=confidence,
                relevance=0.5,
                provenance_json=json.dumps({
                    "created_by": agent_role,
                    "derived_from": [task_id],
                    "validation_history": [],
                }),
                visibility=Visibility.PROJECT.value,
                write_policy=WritePolicy.OPEN.value,
                created_at=now,
                updated_at=now,
            ))

        await self._index(memory_id, title, category.value, content_str, tier, project_id, scope, tags, confidence)

        log.debug("memory.accepted", memory_id=memory_id, tier=tier.value, category=category.value)
        return memory_id

    async def _index(
        self,
        memory_id: str,
        title: str,
        category: str,
        content: str,
        tier: MemoryTier,
        project_id: str,
        scope: MemoryScope,
        tags: list[str],
        confidence: float,
    ) -> None:
        chroma = self._get_chroma()
        if not chroma:
            return
        try:
            collection_name = _TIER_TO_COLLECTION.get(tier.value, "project_working")
            collection = chroma.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            text = f"{title}\n{category}\n{content}"[:2000]
            metadata: dict[str, str] = {
                "category": category,
                "confidence": str(confidence),
                "tags": ",".join(tags[:10]),
                "scope": scope.value,
            }
            # Only tag with project_id if it's a project-scoped item
            # BUG FIX: cross-project insights stored WITHOUT project_id so
            # they can be retrieved by any future project.
            if scope == MemoryScope.PROJECT:
                metadata["project_id"] = project_id

            collection.upsert(documents=[text], ids=[memory_id], metadatas=[metadata])
        except Exception as exc:
            log.warning("memory.index_failed", memory_id=memory_id, error=str(exc))

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    async def retrieve_for_task(
        self,
        query: str,
        project_id: str,
        task_id: str,
        n_results: int = 8,
    ) -> list[MemoryExcerpt]:
        try:
            results = await self._chroma_query(query, project_id, n_results)
            if results:
                return results
        except Exception as exc:
            log.warning("memory.chroma_query_failed", error=str(exc))
        return await self._db_fallback(project_id, n_results)

    async def retrieve_failures(self, project_id: str, n_results: int = 4) -> list[MemoryExcerpt]:
        from sqlalchemy import select
        async with session_factory() as db:
            rows = (
                await db.execute(
                    select(MemoryObjectRow)
                    .where(
                        MemoryObjectRow.project_id == project_id,
                        MemoryObjectRow.tier == MemoryTier.FAILURE.value,
                    )
                    .order_by(MemoryObjectRow.created_at.desc())
                    .limit(n_results)
                )
            ).scalars().all()
        return [_row_to_excerpt(r) for r in rows]

    async def _chroma_query(
        self, query: str, project_id: str, n_results: int
    ) -> list[MemoryExcerpt]:
        chroma = self._get_chroma()
        if not chroma:
            return []

        excerpts: list[MemoryExcerpt] = []

        # Project-scoped collections: filter by project_id
        for coll_name in ("project_working", "project_artifacts"):
            try:
                coll = chroma.get_or_create_collection(coll_name)
                if coll.count() == 0:
                    continue
                results = coll.query(
                    query_texts=[query],
                    n_results=min(4, n_results, coll.count()),
                    where={"project_id": project_id},  # project-scoped
                )
                excerpts.extend(_parse_chroma_results(results))
            except Exception:
                continue

        # BUG FIX: insight_memory is NOT filtered by project_id because
        # cross-project insights have no project_id in their metadata.
        # Query without a filter to retrieve global learnings.
        try:
            insight_coll = chroma.get_or_create_collection("insight_memory")
            if insight_coll.count() > 0:
                results = insight_coll.query(
                    query_texts=[query],
                    n_results=min(3, insight_coll.count()),
                    # No where filter — cross-project insights have no project_id
                )
                excerpts.extend(_parse_chroma_results(results))
        except Exception:
            pass

        excerpts.sort(key=lambda e: e.relevance, reverse=True)
        return excerpts[:n_results]

    async def _db_fallback(self, project_id: str, n_results: int) -> list[MemoryExcerpt]:
        from sqlalchemy import select
        async with session_factory() as db:
            rows = (
                await db.execute(
                    select(MemoryObjectRow)
                    .where(
                        MemoryObjectRow.project_id == project_id,
                        MemoryObjectRow.tier != MemoryTier.FAILURE.value,
                    )
                    .order_by(MemoryObjectRow.confidence.desc())
                    .limit(n_results)
                )
            ).scalars().all()
        return [_row_to_excerpt(r) for r in rows]

    async def promote_to_canonical(self, memory_id: str, approved_by: str) -> None:
        now = datetime.now(timezone.utc)
        async with session_factory() as db:
            row = await db.get(MemoryObjectRow, memory_id)
            if not row:
                raise ValueError(f"Memory {memory_id!r} not found.")
            row.status = MemoryStatus.CANONICAL.value
            row.tier = MemoryTier.CANONICAL.value
            row.write_policy = WritePolicy.RESTRICTED.value
            row.visibility = Visibility.GLOBAL.value
            row.scope = MemoryScope.GLOBAL.value
            prov = json.loads(row.provenance_json)
            prov["last_human_actor"] = approved_by
            prov["last_human_touch"] = now.isoformat()
            row.provenance_json = json.dumps(prov)
            row.updated_at = now
        log.info("memory.promoted_canonical", memory_id=memory_id, by=approved_by)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_excerpt(row: MemoryObjectRow) -> MemoryExcerpt:
    return MemoryExcerpt(
        memory_id=row.memory_id,
        category=row.category,
        title=row.title,
        content=row.content[:600],
        confidence=row.confidence,
        relevance=row.relevance,
        tags=json.loads(row.tags_json) if row.tags_json else [],
    )


def _parse_chroma_results(results: dict) -> list[MemoryExcerpt]:
    excerpts = []
    try:
        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            relevance = max(0.0, 1.0 - float(dist))
            lines = doc.split("\n")
            excerpts.append(MemoryExcerpt(
                memory_id=results["ids"][0][i],
                category=meta.get("category", "unknown"),
                title=lines[0][:100] if lines else "",
                content="\n".join(lines[2:])[:600] if len(lines) > 2 else doc[:600],
                confidence=float(meta.get("confidence", "0.5")),
                relevance=relevance,
                tags=meta.get("tags", "").split(",") if meta.get("tags") else [],
            ))
    except Exception:
        pass
    return excerpts


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager
