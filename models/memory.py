"""
Memory schema — all six tiers from spec §4.

Tier hierarchy (trust level, low → high):
  1. EphemeralContext    — scratch space, per-task, never persisted
  2. ProjectWorking      — mutable project state
  3. ProjectArtifact     — versioned deliverables and research outputs
  4. Failure             — unsuccessful attempts and why they failed
  5. Insight             — validated heuristics, cross-project patterns
  6. CanonicalKnowledge  — source of truth, write-restricted

Design:
  - All tiers share MemoryObject as a base (spec §4.3 schema).
  - Tier-specific fields are added via subclassing.
  - The MemoryManager (Phase 3) decides which tier an item belongs to
    and enforces write policies.
  - Agents submit *candidate* MemoryItems and never write directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from core.ids import prefixed_id


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MemoryTier(StrEnum):
    EPHEMERAL = "ephemeral"
    PROJECT_WORKING = "project_working"
    PROJECT_ARTIFACT = "project_artifact"
    FAILURE = "failure"
    INSIGHT = "insight"
    CANONICAL = "canonical"


class MemoryStatus(StrEnum):
    DRAFT = "draft"
    VERIFIED = "verified"
    REJECTED = "rejected"
    CANONICAL = "canonical"
    RETIRED = "retired"         # demoted canonical item


class MemoryCategory(StrEnum):
    """Fine-grained type for retrieval precision (spec §4.4)."""
    GOAL = "goal"
    CONSTRAINT = "constraint"
    ASSUMPTION = "assumption"
    RESEARCH_FINDING = "research_finding"
    DECISION = "decision"
    REJECTED_OPTION = "rejected_option"
    FAILURE_REASON = "failure_reason"
    DESIGN_RATIONALE = "design_rationale"
    ARTIFACT_REFERENCE = "artifact_reference"
    TEST_RESULT = "test_result"
    INSIGHT = "insight"
    RISK = "risk"
    MILESTONE = "milestone"
    HUMAN_OVERRIDE = "human_override"


class MemoryScope(StrEnum):
    PROJECT = "project"         # visible only within this project
    CROSS_PROJECT = "cross_project"  # promoted by learning pipeline
    GLOBAL = "global"           # system-wide policy / schema


class WritePolicy(StrEnum):
    OPEN = "open"               # any agent can update
    GATED = "gated"             # requires validation pass
    RESTRICTED = "restricted"   # governance pathway only


class Visibility(StrEnum):
    PRIVATE = "private"         # only the creating agent
    DEPARTMENT = "department"   # agents in the same workspace
    PROJECT = "project"         # all agents in the project
    GLOBAL = "global"           # all projects


# ---------------------------------------------------------------------------
# Provenance record — every meaningful item tracks its origin
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    created_by: str = Field(description="Agent ID or 'system' or 'human'.")
    derived_from: list[str] = Field(
        default_factory=list,
        description="Memory object IDs or artifact IDs this item was derived from.",
    )
    validation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of validation events: {validator, result, score, timestamp}.",
    )
    last_human_touch: datetime | None = None
    last_human_actor: str | None = None


# ---------------------------------------------------------------------------
# Base memory object (spec §4.3)
# ---------------------------------------------------------------------------


class MemoryObject(BaseModel):
    """
    Base for all memory items. Every field maps directly to spec §4.3.

    Never instantiated directly — use one of the tier subclasses.
    The 'tier' discriminator field enables Pydantic's tagged union
    so you can deserialize a list[MemoryObject] from the DB without
    knowing the tier up front.
    """

    memory_id: str = Field(default_factory=lambda: prefixed_id("mem"))
    tier: MemoryTier
    category: MemoryCategory
    scope: MemoryScope = MemoryScope.PROJECT
    status: MemoryStatus = MemoryStatus.DRAFT

    # Linking
    project_id: str | None = None
    task_id: str | None = None
    department_id: str | None = None
    agent_id: str | None = None

    # Content
    title: str = Field(..., min_length=1, max_length=300)
    content: str | dict[str, Any] = Field(
        ...,
        description="Plain text for notes/findings, structured dict for decisions/schemas.",
    )
    tags: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(
        default_factory=list,
        description="External URLs, document names, or system artifact IDs.",
    )

    # Quality signals (used by retrieval ranking in Phase 3)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)

    # Versioning
    version: int = Field(default=1, ge=1)
    previous_version_id: str | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = Field(
        default=None,
        description="For ephemeral items — TTL after which the memory manager purges them.",
    )

    # Access control
    visibility: Visibility = Visibility.PROJECT
    write_policy: WritePolicy = WritePolicy.OPEN
    provenance: Provenance = Field(default_factory=lambda: Provenance(created_by="system"))

    # Embedding (populated by vector store on write; not stored in relational DB)
    embedding_vector: list[float] | None = Field(default=None, exclude=True)

    @field_validator("tags")
    @classmethod
    def lowercase_tags(cls, v: list[str]) -> list[str]:
        return [tag.lower().strip() for tag in v]

    def bump_version(self) -> "MemoryObject":
        """Return a copy with version incremented and previous pointer set."""
        data = self.model_dump()
        data["previous_version_id"] = self.memory_id
        data["memory_id"] = prefixed_id("mem")
        data["version"] = self.version + 1
        data["updated_at"] = datetime.now(timezone.utc)
        return self.__class__(**data)

    def text_for_embedding(self) -> str:
        """Canonical string passed to the embedder (ChromaDB / sentence-transformers)."""
        content_str = self.content if isinstance(self.content, str) else str(self.content)
        return f"{self.title}\n{self.category.value}\n{content_str}"


# ---------------------------------------------------------------------------
# Tier 1 — Ephemeral context (scratch, never written to DB)
# ---------------------------------------------------------------------------


class EphemeralContext(MemoryObject):
    """
    Short-lived scratch space within a single task execution window.
    Lives in agent memory only. The MemoryManager must never persist this.
    """

    tier: Literal[MemoryTier.EPHEMERAL] = MemoryTier.EPHEMERAL
    write_policy: WritePolicy = WritePolicy.OPEN
    visibility: Visibility = Visibility.PRIVATE

    # Ephemeral fields
    task_window_id: str = Field(description="Groups all scratch items from one task run.")
    scratch_note: str = Field(default="", description="Free-form working notes.")


# ---------------------------------------------------------------------------
# Tier 2 — Project working memory (mutable project state)
# ---------------------------------------------------------------------------


class ProjectWorkingMemory(MemoryObject):
    """
    Mutable project state: current assumptions, drafts, active concerns.
    Survives task handoff and workspace restart.
    Updated frequently. Not versioned beyond the base version field.
    """

    tier: Literal[MemoryTier.PROJECT_WORKING] = MemoryTier.PROJECT_WORKING
    write_policy: WritePolicy = WritePolicy.OPEN

    is_assumption: bool = False
    is_concern: bool = False
    is_open_question: bool = False
    draft_artifact_id: str | None = None  # link to in-progress artifact


# ---------------------------------------------------------------------------
# Tier 3 — Project artifact memory (versioned deliverables)
# ---------------------------------------------------------------------------


class ArtifactType(StrEnum):
    RESEARCH_DOC = "research_doc"
    DESIGN_DRAFT = "design_draft"
    CODE = "code"
    TABLE = "table"
    DIAGRAM = "diagram"
    TEST_RESULT = "test_result"
    VALIDATED_SUMMARY = "validated_summary"
    MILESTONE_OUTPUT = "milestone_output"


class ProjectArtifact(MemoryObject):
    """
    Canonical store of produced artifacts.
    - Never silently overwritten — always bumped to new version.
    - Traceable to originating task and agent.
    - Can only be retired through governance (Phase 8).
    """

    tier: Literal[MemoryTier.PROJECT_ARTIFACT] = MemoryTier.PROJECT_ARTIFACT
    write_policy: WritePolicy = WritePolicy.GATED
    visibility: Visibility = Visibility.PROJECT

    artifact_type: ArtifactType
    file_path: str | None = Field(
        default=None,
        description="Relative path if the artifact is stored on disk (e.g. code files).",
    )
    mime_type: str = Field(default="text/plain")
    byte_size: int | None = None
    validation_score: float | None = None
    is_final: bool = False  # True once promoted to deliverable


# ---------------------------------------------------------------------------
# Tier 4 — Failure memory (unsuccessful attempts, permanently retained)
# ---------------------------------------------------------------------------


class FailureReason(StrEnum):
    VALIDATION_FAIL = "validation_fail"
    AGENT_ERROR = "agent_error"
    CONTEXT_OVERLOAD = "context_overload"
    INCORRECT_ASSUMPTION = "incorrect_assumption"
    DEAD_END = "dead_end"
    CONSTRAINT_VIOLATION = "constraint_violation"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class FailureMemory(MemoryObject):
    """
    Records what failed and why. Never purged (unless explicitly archived
    by governance). Searchable by failure pattern for future prevention.
    """

    tier: Literal[MemoryTier.FAILURE] = MemoryTier.FAILURE
    write_policy: WritePolicy = WritePolicy.GATED  # MemoryManager writes, not agents
    visibility: Visibility = Visibility.PROJECT

    failure_reason: FailureReason
    failed_output_summary: str = Field(
        description="Brief summary of what the failed output contained.",
        max_length=2000,
    )
    validation_result: dict[str, Any] = Field(
        default_factory=dict,
        description="Full validation report that caused the rejection.",
    )
    rejected_assumptions: list[str] = Field(default_factory=list)
    revised_direction: str = Field(
        default="",
        description="What was tried instead after this failure.",
        max_length=2000,
    )
    is_archived: bool = False  # soft-delete equivalent for old failures


# ---------------------------------------------------------------------------
# Tier 5 — Insight memory (validated heuristics, cross-project patterns)
# ---------------------------------------------------------------------------


class InsightStatus(StrEnum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    CANONICAL = "canonical"
    RETIRED = "retired"


class Insight(MemoryObject):
    """
    High-value abstraction layer. Only populated when a pattern has been
    validated across multiple episodes. Carries explicit confidence and
    applicability bounds so it is not over-generalized.
    """

    tier: Literal[MemoryTier.INSIGHT] = MemoryTier.INSIGHT
    write_policy: WritePolicy = WritePolicy.GATED
    scope: MemoryScope = MemoryScope.CROSS_PROJECT  # insights default to cross-project

    insight_status: InsightStatus = InsightStatus.CANDIDATE
    statement: str = Field(..., min_length=10, max_length=2000)
    conditions: list[str] = Field(
        default_factory=list,
        description="When this insight applies.",
    )
    exceptions: list[str] = Field(
        default_factory=list,
        description="When this insight does NOT apply (anti-overfitting).",
    )
    origin_task_ids: list[str] = Field(default_factory=list)
    evidence_artifact_ids: list[str] = Field(default_factory=list)
    supporting_project_ids: list[str] = Field(
        default_factory=list,
        description="Projects where this pattern was observed.",
    )
    support_count: int = Field(
        default=1,
        ge=1,
        description="Number of independent observations. Promotes to VERIFIED at threshold.",
    )


# ---------------------------------------------------------------------------
# Tier 6 — Canonical knowledge (source of truth, write-restricted)
# ---------------------------------------------------------------------------


class CanonicalKnowledge(MemoryObject):
    """
    System source of truth. Changed only through a governance pathway (Phase 8).
    Includes: approved goals, locked requirements, validated findings, final decisions.
    """

    tier: Literal[MemoryTier.CANONICAL] = MemoryTier.CANONICAL
    write_policy: WritePolicy = WritePolicy.RESTRICTED
    visibility: Visibility = Visibility.GLOBAL
    scope: MemoryScope = MemoryScope.GLOBAL

    is_policy: bool = False  # True for system-wide behavioral rules
    governance_approval_id: str | None = Field(
        default=None,
        description="ID of the governance record that authorized this item.",
    )
    locked_at: datetime | None = None
    locked_by: str | None = None


# ---------------------------------------------------------------------------
# Union type for DB deserialization
# ---------------------------------------------------------------------------

AnyMemoryObject = (
    EphemeralContext
    | ProjectWorkingMemory
    | ProjectArtifact
    | FailureMemory
    | Insight
    | CanonicalKnowledge
)


# ---------------------------------------------------------------------------
# Memory candidate — what agents submit to the MemoryManager
# ---------------------------------------------------------------------------


class MemoryCandidate(BaseModel):
    """
    Agents never write to the memory store directly.
    They submit MemoryCandidate objects and the MemoryManager decides
    whether to persist, which tier to assign, and what write policy applies.
    """

    candidate_id: str = Field(default_factory=lambda: prefixed_id("cnd"))
    submitting_agent_id: str
    task_id: str
    project_id: str

    proposed_category: MemoryCategory
    proposed_tier: MemoryTier | None = Field(
        default=None,
        description="Agent's suggestion. MemoryManager may override.",
    )
    title: str = Field(..., min_length=1, max_length=300)
    content: str | dict[str, Any]
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_refs: list[str] = Field(default_factory=list)

    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    disposition: str | None = Field(
        default=None,
        description="Set by MemoryManager: 'accepted', 'rejected', 'duplicate'.",
    )
    resulting_memory_id: str | None = None
