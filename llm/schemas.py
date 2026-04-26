"""
LLM schemas — Pydantic models for the Ollama wire format and
the agent input/output contracts defined in spec §6.3–6.5.

Wire format mirrors the Ollama REST API exactly so the client can
serialize/deserialize without manual mapping.

Agent schemas (AgentInputBundle, AgentOutput) are what the orchestration
engine (Phase 2) passes to agents, and what agents must return.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Wire format — mirrors Ollama /api/chat and /api/generate
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in a conversation (system / user / assistant)."""

    role: Literal["system", "user", "assistant"]
    content: str

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"Message(role={self.role!r}, content={preview!r}{'…' if len(self.content) > 60 else ''})"


class OllamaOptions(BaseModel):
    """
    Subset of Ollama model options passed in the 'options' field.
    All optional — Ollama uses its defaults when omitted.
    """

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    num_predict: int | None = Field(
        default=None,
        description="Max tokens to generate. Maps to max_tokens in OpenAI terminology.",
    )
    stop: list[str] | None = None
    repeat_penalty: float | None = Field(default=None, ge=0.0)
    seed: int | None = None

    def to_ollama_dict(self) -> dict[str, Any]:
        """Drop None values — Ollama ignores unset options but keeping nulls is noisy."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ChatRequest(BaseModel):
    """Request body for POST /api/chat."""

    model: str
    messages: list[Message]
    stream: bool = False
    options: OllamaOptions = Field(default_factory=OllamaOptions)
    keep_alive: str = "5m"  # how long to keep model in VRAM after response

    def to_ollama_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [m.model_dump() for m in self.messages],
            "stream": self.stream,
            "options": self.options.to_ollama_dict(),
            "keep_alive": self.keep_alive,
        }


class GenerateRequest(BaseModel):
    """Request body for POST /api/generate (single-turn, no message history)."""

    model: str
    prompt: str
    system: str = ""
    stream: bool = False
    options: OllamaOptions = Field(default_factory=OllamaOptions)

    def to_ollama_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model,
            "prompt": self.prompt,
            "stream": self.stream,
            "options": self.options.to_ollama_dict(),
        }
        if self.system:
            d["system"] = self.system
        return d


class UsageStats(BaseModel):
    """Token counts extracted from the Ollama response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_ollama_response(cls, data: dict[str, Any]) -> "UsageStats":
        prompt = data.get("prompt_eval_count", 0) or 0
        completion = data.get("eval_count", 0) or 0
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )


class ChatResponse(BaseModel):
    """Parsed response from POST /api/chat (non-streaming)."""

    model: str
    message: Message
    done: bool = True
    done_reason: str = "stop"
    usage: UsageStats = Field(default_factory=UsageStats)
    total_duration_ms: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def content(self) -> str:
        """Convenience accessor for the assistant message text."""
        return self.message.content

    @classmethod
    def from_ollama_dict(cls, data: dict[str, Any]) -> "ChatResponse":
        total_ns = data.get("total_duration")
        return cls(
            model=data["model"],
            message=Message(**data["message"]),
            done=data.get("done", True),
            done_reason=data.get("done_reason", "stop"),
            usage=UsageStats.from_ollama_response(data),
            total_duration_ms=int(total_ns / 1_000_000) if total_ns else None,
        )


class StreamChunk(BaseModel):
    """
    A single chunk from a streaming /api/chat response.
    When done=True this is the final chunk and carries usage stats.
    """

    model: str
    message: Message
    done: bool
    usage: UsageStats | None = None

    @classmethod
    def from_ollama_dict(cls, data: dict[str, Any]) -> "StreamChunk":
        done = data.get("done", False)
        return cls(
            model=data["model"],
            message=Message(**data["message"]),
            done=done,
            usage=UsageStats.from_ollama_response(data) if done else None,
        )


class ModelInfo(BaseModel):
    """Entry in the GET /api/tags response."""

    name: str
    modified_at: str | None = None
    size: int | None = None  # bytes
    digest: str | None = None

    @property
    def size_gb(self) -> float | None:
        return round(self.size / 1e9, 2) if self.size else None


class AvailableModels(BaseModel):
    models: list[ModelInfo] = Field(default_factory=list)

    def has_model(self, name: str) -> bool:
        """Check by exact name or base name (ignores tag suffix)."""
        base = name.split(":")[0]
        for m in self.models:
            if m.name == name or m.name.split(":")[0] == base:
                return True
        return False


# ---------------------------------------------------------------------------
# LLM call result — wraps ChatResponse with retry / timing metadata
# ---------------------------------------------------------------------------


class LLMResult(BaseModel):
    """
    What the client returns to callers — includes the response plus
    execution metadata useful for the audit trail (Phase 7).
    """

    response: ChatResponse
    attempt_number: int = 1
    total_latency_ms: int = 0
    model_used: str = ""
    was_retried: bool = False

    @property
    def content(self) -> str:
        return self.response.content

    @property
    def usage(self) -> UsageStats:
        return self.response.usage


# ---------------------------------------------------------------------------
# Agent I/O contracts (spec §6.3–6.5)
# ---------------------------------------------------------------------------


class MemoryExcerpt(BaseModel):
    """
    A single retrieved memory item passed into an agent's input bundle.
    Stripped down from the full MemoryObject — agents see only what they need.
    """

    memory_id: str
    category: str
    title: str
    content: str
    confidence: float
    relevance: float
    tags: list[str] = Field(default_factory=list)


class AgentInputBundle(BaseModel):
    """
    The full context package delivered to an agent before it runs (spec §6.4).
    Built by PromptBuilder.assemble_bundle() using task + memory retrieval.

    Principle: an agent receives ONLY the context required for its role.
    This bundle is what becomes the prompt messages list.
    """

    # Identity
    agent_role: str  # DepartmentOwner value
    task_id: str
    project_id: str

    # Core task
    task_objective: str = Field(description="Clear statement of what this task must produce.")
    task_description: str = Field(default="")

    # Project context (subset only — not the full project state)
    project_title: str = ""
    primary_goal: str = ""
    hard_constraints: list[str] = Field(default_factory=list)

    # Memory context (retrieved, ranked, de-duplicated by Phase 3)
    relevant_memories: list[MemoryExcerpt] = Field(default_factory=list)
    prior_failures: list[MemoryExcerpt] = Field(default_factory=list)

    # Validation & output expectations
    validation_criteria: list[str] = Field(default_factory=list)
    expected_output_types: list[str] = Field(default_factory=list)
    quality_threshold: float = 0.75

    # Phase 3: workspace artifact shelf
    input_artifact_ids: list[str] = Field(
        default_factory=list,
        description="IDs of approved upstream artifacts on this workspace's shelf.",
    )

    # Phase 4: skill instruction content — prepended to system prompt
    skill_contents: list[str] = Field(
        default_factory=list,
        description="Loaded SKILL.md texts for skills required by this task.",
    )

    # Retry context — injected on second+ attempt
    attempt_number: int = 1
    corrective_feedback: str = Field(
        default="",
        description="Specific feedback from the previous failed attempt. Empty on first try.",
    )
    prior_attempt_summary: str = Field(
        default="",
        description="Brief summary of what the prior attempt produced. Empty on first try.",
    )

    # Phase 6: human operator notes
    human_notes: list[str] = Field(
        default_factory=list,
        description="Notes injected by the human operator via the governance API.",
    )

    # Budget
    max_tokens: int = 4096
    max_runtime_seconds: int = 300


class AgentOutput(BaseModel):
    """
    Structured output every agent must return (spec §6.5).

    The orchestration engine reads this to decide:
    - Whether to promote outputs to memory
    - Whether to send to validation
    - Whether to extract memory candidates
    - What to pass to downstream tasks
    """

    task_id: str
    agent_role: str

    status: Literal["complete", "blocked", "needs_clarification"] = "complete"

    # Core content
    summary: str = Field(..., min_length=10, description="One-paragraph summary of what was produced.")
    findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    # Quality signals
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Memory IDs, artifact IDs, or source URLs supporting key claims.",
    )

    # For the orchestrator and next tasks
    open_questions: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)

    # Memory candidates the agent wants to persist (MemoryManager decides whether to accept)
    memory_candidates: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Each dict has: category, title, content, confidence, tags.",
    )

    # IDs of new artifacts created
    artifacts_created: list[str] = Field(default_factory=list)

    # Blocking info (populated when status='blocked')
    blocked_reason: str = ""
    blocked_waiting_for: str = ""

    # Raw LLM output preserved for debugging / audit trail
    raw_llm_content: str = Field(default="", exclude=True)

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 3)
