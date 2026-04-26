"""
models/skill.py — Skill domain model (Phase 4).

A Skill is a versioned, named capability pack that extends what an agent
can produce. It consists of:
  - content_md: the SKILL.md instruction document injected into system prompts
  - departments: which departments may load this skill (empty = any)
  - tool_ids: tools registered by this skill (reserved for future dispatch)

Skills live in the DB and are also seeded from the built-in skills/
filesystem tree. User skills (source="user") override builtins of the
same name (spec §9 custom skill override).

Naming: slash-separated path — "code/python", "writing/longform", etc.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from core.ids import prefixed_id


class Skill(BaseModel):
    skill_id:    str  = Field(default_factory=lambda: prefixed_id("skl"))
    name:        str  = Field(..., description="Slash-separated path: 'code/python'")
    version:     str  = Field(default="1.0")
    description: str  = Field(default="")

    # Empty list means usable by any department
    departments: list[str] = Field(default_factory=list)

    # Full SKILL.md content — injected into agent system prompt
    content_md:  str       = Field(default="")

    # Tool IDs registered by this skill (future use)
    tool_ids:    list[str] = Field(default_factory=list)

    # "builtin" | "user" — user overrides builtin of same name
    source:      str       = Field(default="builtin")

    created_at:  datetime  = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:  datetime  = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def slug(self) -> str:
        return f"{self.name}@{self.version}"

    def applies_to(self, department: str) -> bool:
        return not self.departments or department in self.departments
