"""
llm/client.py — LLM client entry point.

Public API (unchanged from original):
    get_client()              → LLMClient singleton
    init_llm_client(settings) → creates and stores the singleton
    close_client()            → graceful shutdown

The LLMClient wraps whichever LLMProvider is configured and adds:
    chat_json()   — JSON extraction wrapper (used by agent runner + planner)
    complete()    — single-turn convenience wrapper
    stream_chat() — streaming passthrough
    chat()        — plain non-streaming passthrough

Provider is selected via settings.llm_provider:
    "ollama"      — local Ollama instance (default, always free)
    "groq"        — Groq free tier (requires GROQ_API_KEY in .env)
    "openrouter"  — OpenRouter free models (requires OPENROUTER_API_KEY in .env)

All providers honour the same retry / backoff rules internally.  The
chat_json() error-recovery loop at this layer is provider-agnostic.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog

from config import Settings, get_settings
from llm.providers.base import LLMProvider
from llm.schemas import LLMResult, Message, StreamChunk

log = structlog.get_logger(__name__)

# Module-level singleton
_client_instance: "LLMClient | None" = None


# ---------------------------------------------------------------------------
# Lifecycle helpers (called from main.py lifespan)
# ---------------------------------------------------------------------------


def get_client() -> "LLMClient":
    """Return the module-level singleton. Raises if not yet initialised."""
    if _client_instance is None:
        raise RuntimeError(
            "LLM client not initialised. Call await init_llm_client(settings) during app startup."
        )
    return _client_instance


async def init_llm_client(settings: Settings) -> "LLMClient":
    """Create and store the module-level client. Called from main.py lifespan."""
    global _client_instance
    if _client_instance is not None:
        log.warning("init_llm_client called more than once — returning existing instance.")
        return _client_instance

    if getattr(settings, "llm_mode", "single") == "tiered":
        # In tiered mode we still need a fallback single client (e.g. for
        # anything that calls get_client() directly), but the main routing
        # goes through TieredRouter.  Build both.
        from llm.tier_router import init_tier_router
        await init_tier_router(settings)
        log.info(
            "llm.mode.tiered",
            tier1=getattr(settings, "tier1_providers", ""),
            tier2=getattr(settings, "tier2_providers", ""),
            tier3=getattr(settings, "tier3_providers", ""),
            tier3_enabled=getattr(settings, "tier3_enabled", False),
        )

    provider = _build_provider(settings)
    _client_instance = LLMClient(provider)
    await _client_instance.startup_check()
    return _client_instance


async def close_client() -> None:
    """Gracefully close the underlying provider and any tier router."""
    global _client_instance
    if _client_instance:
        await _client_instance.close()
        _client_instance = None
    # Also shut down the tier router if it was started
    try:
        from llm.tier_router import close_router
        await close_router()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def _build_provider(settings: Settings) -> LLMProvider:
    """
    Instantiate the correct provider from settings.llm_provider.

    Hierarchy:
        1. "groq"       — requires GROQ_API_KEY
        2. "openrouter" — requires OPENROUTER_API_KEY
        3. "ollama"     — local or proxied Ollama (default)
    """
    provider_name = (settings.llm_provider or "ollama").lower().strip()

    if provider_name == "groq":
        from llm.providers.openai_compat import make_groq_provider
        if not settings.groq_api_key:
            log.warning(
                "llm.provider.groq_key_missing",
                hint="Set GROQ_API_KEY in .env — falling back to llm7",
            )
            return _make_llm7_provider(settings)
        log.info("llm.provider.selected", provider="groq", model=settings.groq_default_model)
        return make_groq_provider(
            api_key=settings.groq_api_key,
            default_model=settings.groq_default_model,
        )

    if provider_name == "openrouter":
        from llm.providers.openai_compat import make_openrouter_provider
        if not settings.openrouter_api_key:
            log.warning(
                "llm.provider.openrouter_key_missing",
                hint="Set OPENROUTER_API_KEY in .env — falling back to llm7",
            )
            return _make_llm7_provider(settings)
        log.info(
            "llm.provider.selected",
            provider="openrouter",
            model=settings.openrouter_default_model,
        )
        return make_openrouter_provider(
            api_key=settings.openrouter_api_key,
            default_model=settings.openrouter_default_model,
        )

    if provider_name == "anthropic":
        from llm.providers.anthropic_provider import AnthropicProvider
        if not settings.anthropic_api_key:
            log.warning(
                "llm.provider.anthropic_key_missing",
                hint="Set ANTHROPIC_API_KEY in .env — falling back to llm7",
            )
            return _make_llm7_provider(settings)
        log.info(
            "llm.provider.selected",
            provider="anthropic",
            model=settings.anthropic_default_model,
        )
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            default_model=settings.anthropic_default_model,
            timeout_seconds=settings.anthropic_timeout_seconds,
        )

    if provider_name == "ollama":
        return _make_ollama_provider(settings)

    # Default — llm7 (keyless, always available)
    if provider_name != "llm7":
        log.warning(
            "llm.provider.unknown",
            requested=provider_name,
            fallback="llm7",
        )
    return _make_llm7_provider(settings)


def _make_llm7_provider(settings: Settings):
    from llm.providers.openai_compat import make_llm7_provider
    model = getattr(settings, "llm7_default_model", "deepseek-ai/DeepSeek-R1")
    log.info("llm.provider.selected", provider="llm7", model=model)
    return make_llm7_provider(default_model=model)


def _make_ollama_provider(settings: Settings):
    from llm.providers.ollama import OllamaProvider
    log.info(
        "llm.provider.selected",
        provider="ollama",
        model=settings.ollama_default_model,
        base_url=settings.ollama_base_url,
    )
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        default_model=settings.ollama_default_model,
        api_key=getattr(settings, "ollama_api_key", ""),
        timeout_seconds=settings.ollama_timeout_seconds,
    )


# ---------------------------------------------------------------------------
# LLMClient — wraps a provider, adds high-level helpers
# ---------------------------------------------------------------------------


class LLMClient:
    """
    Provider-agnostic LLM client used everywhere in the engine.

    Wraps an LLMProvider and adds:
      - chat_json()  : JSON extraction with parse-retry loop
      - complete()   : single-turn shorthand
      - stream_chat(): streaming passthrough
      - chat()       : plain non-streaming passthrough

    The agent runner and planner import get_client() and call chat_json()
    or chat() directly — they never interact with the provider layer.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        log.info(
            "llm.client.ready",
            provider=provider.provider_name,
            model=provider.default_model,
        )

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def startup_check(self) -> None:
        """
        Run provider-specific startup verification.
        For Ollama: checks reachability + model availability.
        For remote providers: lightweight health_check() ping.
        """
        from llm.providers.ollama import OllamaProvider
        if isinstance(self._provider, OllamaProvider):
            await self._provider.startup_check()
        else:
            ok = await self._provider.health_check()
            if ok:
                log.info(
                    "llm.startup.ok",
                    provider=self._provider.provider_name,
                    model=self._provider.default_model,
                )
            else:
                log.warning(
                    "llm.startup.unreachable",
                    provider=self._provider.provider_name,
                    hint="LLM calls will fail until the provider is reachable.",
                )

    # ------------------------------------------------------------------
    # Core passthrough methods
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retry_limit: int = 3,
    ) -> LLMResult:
        """Non-streaming chat. Delegates to the configured provider."""
        return await self._provider.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_limit=retry_limit,
        )

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming chat. Delegates to the configured provider."""
        return self._provider.stream_chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Single-turn convenience wrapper
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retry_limit: int = 3,
    ) -> LLMResult:
        """
        Single-turn text completion.
        Builds a message list and delegates to chat().
        """
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        return await self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_limit=retry_limit,
        )

    # ------------------------------------------------------------------
    # Structured JSON extraction
    # ------------------------------------------------------------------

    async def chat_json(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        retry_limit: int = 3,
    ) -> tuple[dict[str, Any], LLMResult]:
        """
        Request JSON output from the model and parse it.

        Appends a JSON instruction to the last user message, then parses
        the response.  On parse failure, injects a correction turn and
        retries.  Raises ValueError if all attempts fail.

        Returns (parsed_dict, LLMResult).

        Used by: agent_runner.py, planner.py, validator.py
        """
        json_messages = list(messages)

        # Inject JSON instruction into the last user turn
        if json_messages and json_messages[-1].role == "user":
            last = json_messages[-1]
            json_messages[-1] = Message(
                role="user",
                content=last.content
                + "\n\nRespond ONLY with valid JSON. No markdown fences, no preamble.",
            )

        last_content = ""
        for attempt_num in range(retry_limit + 1):
            result = await self.chat(
                json_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                retry_limit=0,  # per-attempt; outer loop handles retries
            )
            raw = result.content.strip()
            last_content = raw

            # Strip markdown code fences — some models add them despite instruction
            if raw.startswith("```"):
                lines = raw.split("\n")
                inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
                raw = "\n".join(inner)

            try:
                return json.loads(raw), result
            except json.JSONDecodeError:
                log.warning(
                    "llm.chat_json.parse_failed",
                    attempt=attempt_num + 1,
                    preview=raw[:200],
                )
                if attempt_num < retry_limit:
                    # Append correction turn and retry
                    json_messages = json_messages + [
                        Message(role="assistant", content=last_content),
                        Message(
                            role="user",
                            content=(
                                "That response was not valid JSON. "
                                "Output ONLY a raw JSON object, nothing else."
                            ),
                        ),
                    ]
                    await asyncio.sleep(0.5)

        raise ValueError(
            f"LLM failed to produce valid JSON after {retry_limit + 1} attempts. "
            f"Last output: {last_content[:500]}"
        )

    # ------------------------------------------------------------------
    # Health / introspection
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """Quick liveness check for the configured provider."""
        return await self._provider.health_check()

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    @property
    def default_model(self) -> str:
        return self._provider.default_model

    async def close(self) -> None:
        await self._provider.close()
        log.info("llm.client.closed", provider=self._provider.provider_name)

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"LLMClient(provider={self._provider.provider_name!r}, "
            f"model={self._provider.default_model!r})"
        )
