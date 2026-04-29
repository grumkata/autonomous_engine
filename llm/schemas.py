"""
llm/schemas.py — All data contracts for the LLM layer.

Key upgrade: Message now supports multimodal content (text + images + audio)
using the OpenAI vision content format, which is compatible with:
  - All OpenAI-compat providers (Groq, Cerebras, Gemini, etc.)
  - Anthropic (via anthropic_provider.py adapter)
  - Ollama vision models (llava, bakllava, moondream, etc.)

Content formats:
  Simple text:    Message(role="user", content="hello")
  Multimodal:     Message(role="user", content=[
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                  ])
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Message — supports text-only and multimodal content
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """
    A single message. content can be a plain string (text-only)
    or a list of content blocks (multimodal — text + images + audio).

    Multimodal block types (OpenAI vision format):
        {"type": "text",      "text": "..."}
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
    """

    role: Literal["system", "user", "assistant"]
    content: str | list[dict[str, Any]]

    def __repr__(self) -> str:
        if isinstance(self.content, str):
            preview = self.content[:60].replace("\n", " ")
            return f"Message(role={self.role!r}, content={preview!r}{'…' if len(self.content) > 60 else ''})"
        return f"Message(role={self.role!r}, content=[{len(self.content)} blocks])"

    @property
    def text_content(self) -> str:
        """Return the text portion of the content regardless of format."""
        if isinstance(self.content, str):
            return self.content
        return " ".join(
            block.get("text", "")
            for block in self.content
            if block.get("type") == "text"
        )

    @property
    def has_images(self) -> bool:
        if isinstance(self.content, list):
            return any(b.get("type") == "image_url" for b in self.content)
        return False

    def to_api_dict(self) -> dict[str, Any]:
        """Serialise for the OpenAI-compat API wire format."""
        return {"role": self.role, "content": self.content}

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def text(cls, role: Literal["system", "user", "assistant"], text: str) -> "Message":
        return cls(role=role, content=text)

    @classmethod
    def with_images(
        cls,
        role: Literal["system", "user", "assistant"],
        text: str,
        image_paths: list[str | Path] | None = None,
        image_urls: list[str] | None = None,
        image_b64s: list[tuple[str, str]] | None = None,  # [(mime, b64data), ...]
    ) -> "Message":
        """
        Build a multimodal message with text + images.

        Args:
            image_paths:  Local file paths — read and base64-encode automatically.
            image_urls:   Remote URLs — passed through directly.
            image_b64s:   Pre-encoded: [(mime_type, base64_data), ...]
        """
        blocks: list[dict] = [{"type": "text", "text": text}]

        for path in (image_paths or []):
            path = Path(path)
            mime = _mime_for_path(path)
            b64  = base64.b64encode(path.read_bytes()).decode()
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

        for url in (image_urls or []):
            blocks.append({"type": "image_url", "image_url": {"url": url}})

        for mime, b64data in (image_b64s or []):
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64data}"},
            })

        return cls(role=role, content=blocks)


def _mime_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
    }.get(ext, "image/png")


# ---------------------------------------------------------------------------
# Ollama-specific (kept for ollama provider compatibility)
# ---------------------------------------------------------------------------


class OllamaOptions(BaseModel):
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p:       float | None = Field(default=None, ge=0.0, le=1.0)
    top_k:       int   | None = Field(default=None, ge=1)
    num_predict: int   | None = None
    stop:        list[str] | None = None
    repeat_penalty: float | None = Field(default=None, ge=0.0)
    seed:        int   | None = None

    def to_ollama_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ChatRequest(BaseModel):
    model:    str
    messages: list[Message]
    stream:   bool = False
    options:  OllamaOptions = Field(default_factory=OllamaOptions)
    keep_alive: str = "5m"

    def to_ollama_dict(self) -> dict[str, Any]:
        return {
            "model":    self.model,
            "messages": [m.to_api_dict() for m in self.messages],
            "stream":   self.stream,
            "options":  self.options.to_ollama_dict(),
            "keep_alive": self.keep_alive,
        }


class GenerateRequest(BaseModel):
    model:  str
    prompt: str
    stream: bool = False
    options: OllamaOptions = Field(default_factory=OllamaOptions)

    def to_ollama_dict(self) -> dict[str, Any]:
        return {
            "model":   self.model,
            "prompt":  self.prompt,
            "stream":  self.stream,
            "options": self.options.to_ollama_dict(),
        }


# ---------------------------------------------------------------------------
# LLM response types
# ---------------------------------------------------------------------------


class UsageStats(BaseModel):
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0


class ChatResponse(BaseModel):
    model:      str
    message:    Message
    done:       bool = True
    done_reason: str = "stop"
    usage:      UsageStats = Field(default_factory=UsageStats)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def content(self) -> str:
        return self.message.text_content


class StreamChunk(BaseModel):
    model:   str
    message: Message
    done:    bool = False
    usage:   UsageStats | None = None

    @property
    def content(self) -> str:
        return self.message.text_content


class LLMResult(BaseModel):
    response:        ChatResponse
    attempt_number:  int = 1
    total_latency_ms: int = 0
    model_used:      str = ""
    was_retried:     bool = False

    @property
    def content(self) -> str:
        return self.response.content

    @property
    def usage(self) -> UsageStats:
        return self.response.usage


# ---------------------------------------------------------------------------
# Agent I/O contracts
# ---------------------------------------------------------------------------


class MemoryExcerpt(BaseModel):
    memory_id:  str
    category:   str
    title:      str
    content:    str
    confidence: float
    relevance:  float
    tags:       list[str] = Field(default_factory=list)


class AgentInputBundle(BaseModel):
    """Full context package delivered to an agent before it runs."""

    agent_role:  str
    task_id:     str
    project_id:  str

    task_objective:  str
    task_description: str = ""

    project_title:    str = ""
    primary_goal:     str = ""
    hard_constraints: list[str] = Field(default_factory=list)

    relevant_memories: list[MemoryExcerpt] = Field(default_factory=list)
    prior_failures:    list[MemoryExcerpt] = Field(default_factory=list)

    validation_criteria:    list[str] = Field(default_factory=list)
    expected_output_types:  list[str] = Field(default_factory=list)
    quality_threshold:      float = 0.75

    input_artifact_ids: list[str] = Field(default_factory=list)
    skill_contents:     list[str] = Field(default_factory=list)

    attempt_number:       int = 1
    corrective_feedback:  str = ""
    prior_attempt_summary: str = ""

    human_notes: list[str] = Field(default_factory=list)

    max_tokens:          int = 4096
    max_runtime_seconds: int = 300

    # Multimodal inputs — images/files the agent can "see"
    visual_inputs: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Pre-rendered visual representations of workspace files. "
            "Each item: {path, type, b64, mime, description}. "
            "Injected as image blocks in the user message for vision-capable models."
        ),
    )


class AgentOutput(BaseModel):
    """Structured output every agent must return."""

    task_id:    str
    agent_role: str
    status:     Literal["complete", "blocked", "needs_clarification"]

    summary:         str
    findings:        list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    risks:           list[str] = Field(default_factory=list)
    assumptions:     list[str] = Field(default_factory=list)
    confidence:      float = 0.5

    evidence_refs:   list[str] = Field(default_factory=list)
    open_questions:  list[str] = Field(default_factory=list)
    next_actions:    list[str] = Field(default_factory=list)

    memory_candidates: list[dict[str, Any]] = Field(default_factory=list)

    artifacts_created: list[str] = Field(default_factory=list)

    # Tool calls the agent wants to make
    tool_calls: list[dict] = Field(
        default_factory=list,
        description="Tool calls to execute before producing final output.",
    )

    blocked_reason:      str = ""
    blocked_waiting_for: str = ""
