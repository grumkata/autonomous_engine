"""
llm/client.py — LLM client entry point.

Modes
-----
LLM_PROVIDER=ollama      Single local Ollama instance (default)
LLM_PROVIDER=groq        Single Groq provider
LLM_PROVIDER=openrouter  Single OpenRouter provider
LLM_PROVIDER=pool        Manual pool via LLM_POOL_CONFIG JSON
LLM_PROVIDER=auto        Auto-build pool from ALL configured API keys

"auto" mode inspects environment variables for every provider in the
catalog and adds a slot for each one that has a key set. This is the
recommended mode — just set the keys you have, and the engine uses all
of them automatically with MetaOrchestrator routing.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from typing import Any

import structlog

from config import Settings, get_settings
from llm.providers.base import LLMProvider
from llm.schemas import LLMResult, Message, StreamChunk

log = structlog.get_logger(__name__)

_client_instance: "LLMClient | None" = None
_router_instance: "ModelRouter | None" = None  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_client() -> "LLMClient":
    if _client_instance is None:
        raise RuntimeError("LLM client not initialised.")
    return _client_instance


def get_client_for(department: str, task_type: str) -> "BoundClient":
    """
    Return a BoundClient pre-filled with routing context.

    When pool/auto mode is active → routes via ModelRouter + MetaOrchestrator.
    Otherwise → wraps the single LLMClient singleton transparently.
    """
    from llm.router import BoundClient
    if _router_instance is not None:
        return BoundClient(_router_instance, department, task_type)
    return BoundClient(get_client(), department, task_type)


async def init_llm_client(settings: Settings) -> "LLMClient":
    global _client_instance, _router_instance

    if _client_instance is not None:
        log.warning("init_llm_client called more than once.")
        return _client_instance

    provider_name = (settings.llm_provider or "ollama").lower().strip()

    if provider_name in ("pool", "auto"):
        await _init_pool(settings, auto=(provider_name == "auto"))
        if _router_instance is None or not _router_instance._slots:
            raise RuntimeError("Pool mode: no slots were configured.")
        _, first_client = _router_instance._slots[0]
        _client_instance = first_client
        return _client_instance

    provider = _build_provider(settings)
    _client_instance = LLMClient(provider)
    await _client_instance.startup_check()
    return _client_instance


async def close_client() -> None:
    global _client_instance, _router_instance
    if _router_instance is not None:
        for _, client in _router_instance._slots:
            await client.close()
        _router_instance = None
    if _client_instance is not None:
        await _client_instance.close()
        _client_instance = None


# ---------------------------------------------------------------------------
# Pool initialisation
# ---------------------------------------------------------------------------


async def _init_pool(settings: Settings, auto: bool = False) -> None:
    global _router_instance

    from llm.router import MetaOrchestrator, ModelRouter, ProviderSlotConfig, TokenLedger

    if auto:
        slot_configs = _build_slots_from_catalog(settings)
        if not slot_configs:
            log.warning(
                "llm.pool.auto.no_keys",
                hint="Set at least one provider API key. Falling back to Ollama.",
            )
            slot_configs = [_ollama_slot_config(settings)]
    else:
        slot_configs = ProviderSlotConfig.parse_pool_config(settings.llm_pool_config)
        if not slot_configs:
            raise ValueError("LLM_PROVIDER=pool but LLM_POOL_CONFIG is empty.")

    # Build LLMClient per slot (run startup checks concurrently)
    async def _make_slot(cfg: ProviderSlotConfig):
        provider = _build_provider_from_slot(cfg, settings)
        client = LLMClient(provider)
        await client.startup_check()
        return cfg, client

    results = await asyncio.gather(
        *[_make_slot(cfg) for cfg in slot_configs],
        return_exceptions=True,
    )

    slots = []
    for res in results:
        if isinstance(res, Exception):
            log.warning("llm.pool.slot.startup_failed", error=str(res))
        else:
            slots.append(res)

    if not slots:
        raise RuntimeError("Pool: every slot failed startup check.")

    log.info("llm.pool.slots_ready", count=len(slots), names=[c.name for c, _ in slots])

    # Build MetaOrchestrator using the configured fast model
    meta = None
    if settings.meta_orchestrator_provider and len(slots) > 1:
        meta = await _build_meta_orchestrator(settings, slots)

    _router_instance = ModelRouter(slots, TokenLedger(), meta=meta)


def _build_slots_from_catalog(settings: Settings) -> list["ProviderSlotConfig"]:
    """
    Inspect all CATALOG providers and build a slot for each one
    whose env var is set (or that requires no key at all).
    """
    from llm.providers.catalog import CATALOG
    from llm.router import ProviderSlotConfig

    slots = []
    cf_account_id = getattr(settings, "cf_account_id", "") or os.environ.get("CF_ACCOUNT_ID", "")

    for name, entry in CATALOG.items():
        # Skip providers that need an account ID we don't have
        if entry.requires_account_id and not cf_account_id:
            continue

        # Get the API key (empty string = keyless provider)
        api_key = ""
        if entry.env_key:
            api_key = getattr(settings, entry.env_key.lower(), "") or os.environ.get(entry.env_key, "")
            if not api_key:
                continue  # Key not configured — skip

        # Build the base URL (may need account_id substitution)
        base_url = entry.base_url
        if entry.requires_account_id:
            base_url = base_url.replace("{account_id}", cf_account_id)

        slots.append(ProviderSlotConfig(
            name=name,
            provider=name,
            model=entry.default_model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=600 if name == "ollama" else 120,
            daily_token_limit=None,
            daily_request_limit=entry.rate_limits.daily_req or None,
            strengths=entry.strengths,
            speed_tier=entry.speed_tier,
            priority=entry.priority,
        ))
        log.info("llm.pool.auto.slot_added", name=name, model=entry.default_model)

    return slots


def _ollama_slot_config(settings: Settings) -> "ProviderSlotConfig":
    from llm.router import ProviderSlotConfig
    return ProviderSlotConfig(
        name="ollama",
        provider="ollama",
        model=settings.ollama_default_model,
        api_key="",
        base_url=settings.ollama_base_url,
        timeout_seconds=settings.ollama_timeout_seconds,
        priority=10,
        speed_tier="slow",
    )


async def _build_meta_orchestrator(
    settings: Settings,
    slots: list[tuple["ProviderSlotConfig", "LLMClient"]],
) -> "MetaOrchestrator | None":
    from llm.router import MetaOrchestrator

    provider_name = settings.meta_orchestrator_provider.lower()
    model = settings.meta_orchestrator_model
    api_key = settings.meta_orchestrator_api_key

    # Check if we already have a slot for this provider (reuse its client)
    for cfg, client in slots:
        if cfg.name == provider_name or cfg.provider == provider_name:
            log.info(
                "meta_orchestrator.reusing_slot",
                slot=cfg.name,
                model=model,
            )
            return MetaOrchestrator(client=client, model=model)

    # Build a dedicated lightweight client for the meta model
    from llm.router import ProviderSlotConfig
    meta_cfg = ProviderSlotConfig(
        name="_meta",
        provider=provider_name,
        model=model,
        api_key=api_key,
        timeout_seconds=5,
        priority=0,
    )
    try:
        meta_provider = _build_provider_from_slot(meta_cfg, settings)
        meta_client = LLMClient(meta_provider)
        ok = await meta_client.ping()
        if not ok:
            log.warning("meta_orchestrator.unreachable", provider=provider_name)
            return None
        log.info("meta_orchestrator.built", provider=provider_name, model=model)
        return MetaOrchestrator(client=meta_client, model=model)
    except Exception as exc:
        log.warning("meta_orchestrator.build_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Provider factories
# ---------------------------------------------------------------------------


def _build_provider(settings: Settings) -> LLMProvider:
    name = (settings.llm_provider or "ollama").lower().strip()

    if name == "groq":
        from llm.providers.openai_compat import make_groq_provider
        if not settings.groq_api_key:
            log.warning("groq key missing, falling back to ollama")
            return _make_ollama_provider(settings)
        return make_groq_provider(api_key=settings.groq_api_key, default_model=settings.groq_default_model)

    if name == "openrouter":
        from llm.providers.openai_compat import make_openrouter_provider
        if not settings.openrouter_api_key:
            log.warning("openrouter key missing, falling back to ollama")
            return _make_ollama_provider(settings)
        return make_openrouter_provider(api_key=settings.openrouter_api_key, default_model=settings.openrouter_default_model)

    return _make_ollama_provider(settings)


def _build_provider_from_slot(cfg: "ProviderSlotConfig", settings: Settings | None = None) -> LLMProvider:  # type: ignore[name-defined]
    """
    Build a provider from any slot config.
    All providers except Ollama use the openai_compat adapter.
    """
    if settings is None:
        settings = get_settings()

    if cfg.provider == "ollama":
        from llm.providers.ollama import OllamaProvider
        return OllamaProvider(
            base_url=cfg.base_url or settings.ollama_base_url,
            default_model=cfg.model,
            api_key=cfg.api_key,
            timeout_seconds=cfg.timeout_seconds,
        )

    # All other providers are OpenAI-compatible
    from llm.providers.openai_compat import OpenAICompatProvider
    return OpenAICompatProvider(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        default_model=cfg.model,
        timeout_seconds=cfg.timeout_seconds,
    )


def _make_ollama_provider(settings: Settings) -> LLMProvider:
    from llm.providers.ollama import OllamaProvider
    log.info("llm.provider.selected", provider="ollama", model=settings.ollama_default_model)
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        default_model=settings.ollama_default_model,
        api_key=getattr(settings, "ollama_api_key", ""),
        timeout_seconds=settings.ollama_timeout_seconds,
    )


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        log.info("llm.client.ready", provider=provider.provider_name, model=provider.default_model)

    async def startup_check(self) -> None:
        from llm.providers.ollama import OllamaProvider
        if isinstance(self._provider, OllamaProvider):
            await self._provider.startup_check()
        else:
            ok = await self._provider.health_check()
            level = "info" if ok else "warning"
            log.msg(
                "llm.startup.check",
                level=level,
                provider=self._provider.provider_name,
                reachable=ok,
            )

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retry_limit: int = 3,
    ) -> LLMResult:
        return await self._provider.chat(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, retry_limit=retry_limit,
        )

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        return self._provider.stream_chat(
            messages, model=model, temperature=temperature, max_tokens=max_tokens,
        )

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
        msgs: list[Message] = []
        if system:
            msgs.append(Message(role="system", content=system))
        msgs.append(Message(role="user", content=prompt))
        return await self.chat(msgs, model=model, temperature=temperature,
                               max_tokens=max_tokens, retry_limit=retry_limit)

    async def chat_json(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        retry_limit: int = 3,
    ) -> tuple[dict[str, Any], LLMResult]:
        import asyncio as _asyncio

        json_messages = list(messages)
        if json_messages and json_messages[-1].role == "user":
            last = json_messages[-1]
            json_messages[-1] = Message(
                role="user",
                content=last.content + "\n\nRespond ONLY with valid JSON. No markdown fences, no preamble.",
            )

        last_content = ""
        for attempt in range(retry_limit + 1):
            result = await self.chat(
                json_messages, model=model, temperature=temperature,
                max_tokens=max_tokens, retry_limit=0,
            )
            raw = result.content.strip()
            last_content = raw
            if raw.startswith("```"):
                lines = raw.split("\n")
                inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
                raw = "\n".join(inner)
            try:
                return json.loads(raw), result
            except json.JSONDecodeError:
                log.warning("llm.chat_json.parse_failed", attempt=attempt + 1, preview=raw[:200])
                if attempt < retry_limit:
                    json_messages = json_messages + [
                        Message(role="assistant", content=last_content),
                        Message(role="user", content="That was not valid JSON. Output ONLY a raw JSON object."),
                    ]
                    await _asyncio.sleep(0.5)

        raise ValueError(
            f"LLM failed to produce valid JSON after {retry_limit + 1} attempts. "
            f"Last: {last_content[:500]}"
        )

    async def ping(self) -> bool:
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


from llm.router import BoundClient  # noqa: E402
