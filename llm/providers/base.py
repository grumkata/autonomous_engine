"""
llm/providers/base.py — Abstract LLM provider contract.

All providers (Ollama, Groq, OpenRouter, …) implement this interface.
The LLMClient in llm/client.py wraps any provider and adds higher-level
helpers like chat_json() and complete().

Design rules:
  - Providers handle HTTP transport, auth, retry, and response normalisation.
  - Providers do NOT handle JSON extraction or prompt engineering — that lives
    in LLMClient.
  - Every provider returns the same LLMResult / ChatResponse types so the rest
    of the engine is provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from llm.schemas import LLMResult, Message, StreamChunk


class LLMProvider(ABC):
    """
    Abstract base for every LLM backend.

    Subclasses must implement:
        chat()          — non-streaming turn
        stream_chat()   — streaming turn (yields StreamChunk objects)
        health_check()  — quick liveness probe
        close()         — release HTTP connections
        default_model   — the model name used when no override is given
    """

    # ------------------------------------------------------------------
    # Core methods (must override)
    # ------------------------------------------------------------------

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retry_limit: int = 3,
    ) -> LLMResult:
        """
        Send a non-streaming chat request.

        Parameters
        ----------
        messages    Full conversation history (system / user / assistant).
        model       Override the provider default model for this call.
        temperature Sampling temperature. Provider default when None.
        max_tokens  Max completion tokens. Provider default when None.
        retry_limit Max retries on transient errors (429 / 5xx / network).

        Returns
        -------
        LLMResult with the parsed response and usage stats.
        """
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream a chat response token by token.

        Yields StreamChunk objects. The final chunk has done=True and
        carries usage stats.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider endpoint is reachable right now."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release all held resources (HTTP connections, etc.)."""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Model name used when the caller passes model=None."""
        ...

    # ------------------------------------------------------------------
    # Optional — subclasses may override for provider-specific info
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Human-readable provider identifier, e.g. 'ollama', 'groq'."""
        return self.__class__.__name__.lower().replace("provider", "")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.default_model!r})"
