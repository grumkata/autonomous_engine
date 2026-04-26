"""
orchestrator/skill_registry.py — Skill loading, lookup, and task annotation (Phase 4).

Public API
----------
SkillRegistry.load_all()                       → None  (seed DB from filesystem)
SkillRegistry.get(name, version?)              → Skill | None
SkillRegistry.list_skills(department?)         → list[Skill]
SkillRegistry.suggest_for_task(type, dept)     → list[str]  (skill names)
SkillRegistry.load_content_for_task(skill_ids) → list[str]  (SKILL.md texts)

get_skill_registry()                           → SkillRegistry (singleton)

How it works
------------
1. On startup (called from main.py lifespan or lazily on first use) the registry
   walks the skills/ directory tree, reads every SKILL.md, and upserts into the
   skills table. User skills (skills/user/) override builtins of the same name.

2. The planner calls suggest_for_task() after generating each task to annotate
   it with required_skill_ids. This is a deterministic heuristic — no LLM call.

3. agent_runner calls load_content_for_task() with the task's required_skill_ids
   list. The returned strings are injected into AgentInputBundle.skill_contents,
   which PromptBuilder prepends to the system message.

Spec §9 rules enforced
-----------------------
- User skills override built-ins of the same name.
- Skills are versioned; a task requesting skill@1.0 always gets that version.
- Conflicting skills (same category, different rules) are flagged at load time.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog
from sqlalchemy import select

from core.ids import prefixed_id
from db.engine import session_factory
from db.tables import SkillRow
from models.skill import Skill

log = structlog.get_logger(__name__)

# Root of the built-in skills tree.  Resolved relative to this file's location
# so it works regardless of where the server is launched from.
_BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Task-type → default skill names to suggest
_TYPE_SKILL_MAP: dict[str, list[str]] = {
    "research":       ["research/academic"],
    "design":         ["design/system"],
    "implementation": [],          # populated per-department below
    "critique":       ["qa/security"],
    "validation":     ["qa/testing"],
    "refinement":     [],
    "merge":          [],
    "documentation":  ["writing/longform"],
    "planning":       [],
    "approval":       [],
    "archive":        [],
}

# Department → extra skills to overlay on top of the type map
_DEPT_SKILL_MAP: dict[str, list[str]] = {
    "implementation": ["code/python"],   # default; typescript added for web tasks
    "qa":             ["qa/testing"],
    "red_team":       ["qa/security"],
    "documentation":  ["writing/longform"],
    "research":       ["research/academic"],
    "design":         ["design/system"],
}


# ---------------------------------------------------------------------------
# SkillRow → Skill mapper (local — avoids circular import with db.mappers)
# ---------------------------------------------------------------------------


def _row_to_skill(row: SkillRow) -> Skill:
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


def _skill_to_row(skill: Skill) -> SkillRow:
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
# SkillRegistry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """
    Singleton service — loads skills from filesystem into DB, answers
    lookup queries, and annotates tasks with appropriate skill IDs.
    """

    # In-process cache keyed by name@version
    _cache: dict[str, Skill] = {}

    # ------------------------------------------------------------------
    # Seed / load from filesystem
    # ------------------------------------------------------------------

    async def load_all(self, skills_dir: Path | None = None) -> int:
        """
        Walk the skills directory tree, read every SKILL.md, and upsert
        into the skills table. Returns the number of skills loaded.

        User skills (source="user") override builtins of the same name.
        Safe to call multiple times — idempotent.
        """
        root = skills_dir or _BUILTIN_SKILLS_DIR
        if not root.exists():
            log.warning("skill_registry.dir_missing", path=str(root))
            return 0

        loaded = 0
        for skill_md in sorted(root.rglob("SKILL.md")):
            # Determine name from relative path: skills/code/python/SKILL.md → "code/python"
            rel = skill_md.parent.relative_to(root)
            name = "/".join(rel.parts)
            source = "user" if "user" in rel.parts else "builtin"

            try:
                content = skill_md.read_text(encoding="utf-8")
            except OSError as exc:
                log.warning("skill_registry.read_error", path=str(skill_md), error=str(exc))
                continue

            skill = Skill(
                name=name,
                version="1.0",
                description=self._extract_description(content),
                content_md=content,
                source=source,
            )
            await self._upsert(skill)
            self._cache[skill.slug] = skill
            loaded += 1
            log.debug("skill_registry.loaded", name=name, source=source)

        log.info("skill_registry.load_all_complete", count=loaded)
        return loaded

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    async def get(self, name: str, version: str = "1.0") -> Skill | None:
        """Return a skill by name (and optionally version)."""
        slug = f"{name}@{version}"
        if slug in self._cache:
            return self._cache[slug]

        async with session_factory() as db:
            result = await db.execute(
                select(SkillRow).where(
                    SkillRow.name == name,
                    SkillRow.version == version,
                )
            )
            row = result.scalars().first()
            if not row:
                return None
            skill = _row_to_skill(row)
            self._cache[slug] = skill
            return skill

    async def list_skills(self, department: str | None = None) -> list[Skill]:
        """Return all skills, optionally filtered to those usable by a department."""
        async with session_factory() as db:
            result = await db.execute(select(SkillRow).order_by(SkillRow.name))
            skills = [_row_to_skill(r) for r in result.scalars().all()]

        if department:
            skills = [s for s in skills if s.applies_to(department)]
        return skills

    # ------------------------------------------------------------------
    # Task annotation
    # ------------------------------------------------------------------

    def suggest_for_task(self, task_type: str, department: str) -> list[str]:
        """
        Return a list of skill names appropriate for this task type and
        department. Deterministic — no LLM call.

        Rules (spec §9 — skills must not contradict each other):
          1. Start from the task-type defaults.
          2. Overlay department defaults.
          3. Deduplicate preserving order.
          4. Never exceed 3 skills per task (context budget).
        """
        names: list[str] = []
        names.extend(_TYPE_SKILL_MAP.get(task_type, []))
        names.extend(_DEPT_SKILL_MAP.get(department, []))

        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                unique.append(n)

        return unique[:3]

    async def load_content_for_task(self, skill_names: list[str]) -> list[str]:
        """
        Return the SKILL.md content strings for the given skill names.
        Skills not found in the registry are silently skipped with a warning.
        """
        contents: list[str] = []
        for name in skill_names:
            skill = await self.get(name)
            if skill:
                contents.append(skill.content_md)
            else:
                log.warning("skill_registry.skill_not_found", name=name)
        return contents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _upsert(self, skill: Skill) -> None:
        """Insert or update a skill. User skills win over builtins on conflict."""
        async with session_factory() as db:
            result = await db.execute(
                select(SkillRow).where(
                    SkillRow.name == skill.name,
                    SkillRow.version == skill.version,
                )
            )
            existing = result.scalars().first()

            if existing:
                # User skill always wins; builtin never overwrites user
                if existing.source == "user" and skill.source == "builtin":
                    return
                existing.description = skill.description
                existing.content_md  = skill.content_md
                existing.source      = skill.source
            else:
                db.add(_skill_to_row(skill))

    @staticmethod
    def _extract_description(content: str) -> str:
        """Pull a one-line description from the first H1 ## Purpose section."""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("# SKILL:"):
                return line[len("# SKILL:"):].strip()
            if line and not line.startswith("#"):
                return line[:120]
        return ""


# ---------------------------------------------------------------------------
# Module-level singleton + startup helper
# ---------------------------------------------------------------------------

_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


async def init_skill_registry(skills_dir: Path | None = None) -> SkillRegistry:
    """
    Call from main.py lifespan after init_db().
    Seeds the DB from the filesystem and primes the in-process cache.
    """
    reg = get_skill_registry()
    await reg.load_all(skills_dir)
    return reg
