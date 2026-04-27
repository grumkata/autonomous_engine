"""
llm/tier_router.py — Tiered provider routing for the autonomous engine.

Solves the "Claude is too expensive for everything" problem by routing
each LLM call to the cheapest provider that can handle the task quality
requirements, escalating automatically when needed.

Tier structure
--------------
TIER 1 — Free, no practical limits (general / planning / low-stakes tasks)
    Best for: planning, research, initial drafts, grunt work
    Examples: Ollama (local), Groq fast models, Cerebras, SambaNova
    Rule: quality_threshold < 0.73  AND  attempt == 1

TIER 2 — Free, rate-limited (important tasks that need more capability)
    Best for: implementation tasks, critiques, validations
    Examples: Gemini 1.5 Flash, Groq 70B, OpenRouter 70B :free models
    Rule: quality_threshold 0.73–0.84  OR  (tier1 task on attempt 2+)

TIER 3 — Paid, toggleable (heavy / final / high-stakes tasks)
    Best for: anything that keeps failing tiers 1-2, project finalization
    Examples: Anthropic Claude, OpenAI, DeepSeek (near-free)
    Rule: quality_threshold >= 0.85  OR  tier3_enabled AND attempt >= 3
    DISABLED by default — set TIER3_ENABLED=true to activate

Auto-escalation
---------------
When a task retries, it automatically escalates one tier:
    attempt 1 → assigned tier
    attempt 2 → assigned tier + 1  (or highest available)
    attempt 3 → assigned tier + 2  (goes to tier 3 if enabled)

This means a tier-1 task that fails twice will be retried by a tier-2
provider, and if that fails, by tier-3 (if enabled).  Expensive models
are only called as a last resort.

Round-robin within a tier
--------------------------
Multiple providers in the same tier are cycled round-robin so you spread
load across free limits rather than burning one provider's quota first.

Configuration (.env)
---------------------
    # List providers per tier in priority order (comma-separated)
    TIER1_PROVIDERS=groq,cerebras,ollama
    TIER2_PROVIDERS=gemini,sambanova,openrouter
    TIER3_PROVIDERS=anthropic,deepseek
    TIER3_ENABLED=false

    # Quality thresholds that define tier boundaries
    TIER1_MAX_THRESHOLD=0.72     # tasks below this → tier 1
    TIER2_MAX_THRESHOLD=0.84     # tasks between tier1 and this → tier 2
    # tasks above TIER2_MAX_THRESHOLD → tier 3

    # Tier 3 is opt-in per project via project settings (governance API)
    # or globally via TIER3_ENABLED=true

Usage
-----
    from llm.tier_router import get_tier_router, task_tier

    router = get_tier_router()
    tier   = task_tier(task)     # 1, 2, or 3

    raw_json, result = await router.chat_json(
        messages,
        tier=tier,
        attempt_number=task.attempt_count + 1,
        max_tokens=task.budget.max_tokens,
    )
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any

import structlog

from llm.providers.base import LLMProvider
from llm.schemas import LLMResult, Message, StreamChunk

log = structlog.get_logger(__name__)

_router_instance: "TieredRouter | None" = None


# ---------------------------------------------------------------------------
# Tier assignment — maps task properties → tier number
# ---------------------------------------------------------------------------


def task_tier(task: Any, settings: Any | None = None) -> int:
    """
    Return the appropriate tier (1, 2, or 3) for a task.

    Rules (in order):
      1. quality_threshold >= TIER2_MAX_THRESHOLD → tier 3
      2. quality_threshold >= TIER1_MAX_THRESHOLD → tier 2
      3. else → tier 1

    The caller can override with task.llm_tier if set.
    """
    if settings is None:
        from config import get_settings
        settings = get_settings()

    # Allow per-task override if the model ever sets it
    if hasattr(task, "llm_tier") and task.llm_tier in (1, 2, 3):
        return task.llm_tier

    t1_max = getattr(settings, "tier1_max_threshold", 0.72)
    t2_max = getattr(settings, "tier2_max_threshold", 0.84)

    qt = getattr(task, "quality_threshold", 0.75)

    if qt >= t2_max:
        return 3
    if qt >= t1_max:
        return 2
    return 1


# ---------------------------------------------------------------------------
# TieredRouter
# ---------------------------------------------------------------------------


class TieredRouter:
    """
    Manages a pool of providers per tier and routes chat_json calls.

    Implements the same interface as LLMClient so the rest of the engine
    can swap between a single provider and a tiered pool without changes.
    """

    def __init__(
        self,
        tiers: dict[int, list[LLMProvider]],
        tier3_enabled: bool = False,
    ) -> None:
        self._tiers: dict[int, list[LLMProvider]] = {
            k: v for k, v in tiers.items() if v
        }
        self._tier3_enabled = tier3_enabled
        self._rr: defaultdict[int, int] = defaultdict(int)  # round-robin counters

        # Log the loaded configuration
        for tier_num, providers in sorted(self._tiers.items()):
            enabled = tier_num < 3 or self._tier3_enabled
            log.info(
                "tier_router.tier_loaded",
                tier=tier_num,
                providers=[p.provider_name for p in providers],
                models=[p.default_model for p in providers],
                enabled=enabled,
            )

    # ------------------------------------------------------------------
    # Provider selection
    # ------------------------------------------------------------------

    def _pick(self, tier: int) -> LLMProvider | None:
        """Round-robin pick from providers in `tier`, fallback to adjacent."""
        # Don't use tier 3 unless enabled
        if tier == 3 and not self._tier3_enabled:
            tier = 2

        candidates = self._tiers.get(tier, [])
        if candidates:
            idx = self._rr[tier] % len(candidates)
            self._rr[tier] = idx + 1
            return candidates[idx]

        # Fallback: find nearest non-empty tier
        for fallback_tier in sorted(self._tiers.keys()):
            if fallback_tier == 3 and not self._tier3_enabled:
                continue
            if self._tiers[fallback_tier]:
                p = self._tiers[fallback_tier][0]
                log.warning(
                    "tier_router.fallback",
                    requested_tier=tier,
                    fallback_tier=fallback_tier,
                    provider=p.provider_name,
                )
                return p

        return None

    def provider_for(self, tier: int, attempt_number: int = 1) -> LLMProvider | None:
        """
        Return the provider to use, applying auto-escalation.
        attempt_number 1 → use assigned tier
        attempt_number 2 → escalate to tier+1
        attempt_number 3 → escalate to tier+2 (tier 3 if enabled)
        """
        escalated = tier + max(0, attempt_number - 1)
        escalated = min(escalated, 3)
        if escalated != tier:
            log.info(
                "tier_router.escalating",
                from_tier=tier,
                to_tier=escalated,
                attempt=attempt_number,
            )
        return self._pick(escalated)

    # ------------------------------------------------------------------
    # chat_json — drop-in replacement for LLMClient.chat_json
    # ------------------------------------------------------------------

    async def chat_json(
        self,
        messages: list[Message],
        *,
        tier: int = 1,
        attempt_number: int = 1,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        retry_limit: int = 3,
    ) -> tuple[dict[str, Any], LLMResult]:
        """
        Route a chat_json call to the appropriate provider.

        Mirrors LLMClient.chat_json() — callers can swap between them.
        """
        from llm.client import LLMClient

        provider = self.provider_for(tier, attempt_number)
        if provider is None:
            raise RuntimeError(
                f"TieredRouter: no provider available for tier {tier} "
                f"(attempt {attempt_number}). "
                "Check your TIER*_PROVIDERS config in .env."
            )

        client = LLMClient(provider)
        log.debug(
            "tier_router.routing",
            tier=tier,
            attempt=attempt_number,
            provider=provider.provider_name,
            model=model or provider.default_model,
        )
        return await client.chat_json(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_limit=retry_limit,
        )

    # ------------------------------------------------------------------
    # Passthrough for non-tiered callers (planner, health checks, etc.)
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        *,
        tier: int = 1,
        attempt_number: int = 1,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retry_limit: int = 3,
    ) -> LLMResult:
        """Route a plain chat call through the tier system."""
        from llm.client import LLMClient

        provider = self.provider_for(tier, attempt_number)
        if provider is None:
            raise RuntimeError("TieredRouter: no providers configured.")
        client = LLMClient(provider)
        return await client.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_limit=retry_limit,
        )

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        tier: int = 1,
        attempt_number: int = 1,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retry_limit: int = 3,
    ) -> LLMResult:
        """Route a single-turn completion through the tier system."""
        from llm.client import LLMClient

        provider = self.provider_for(tier, attempt_number)
        if provider is None:
            raise RuntimeError("TieredRouter: no providers configured.")
        client = LLMClient(provider)
        return await client.complete(
            prompt,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_limit=retry_limit,
        )

    async def ping(self) -> dict[int, dict[str, bool]]:
        """
        Liveness check all providers.
        Returns {tier: {provider_name: is_alive}}.
        """
        from llm.client import LLMClient

        results: dict[int, dict[str, bool]] = {}
        for tier_num, providers in self._tiers.items():
            results[tier_num] = {}
            for p in providers:
                try:
                    ok = await LLMClient(p).ping()
                except Exception:
                    ok = False
                results[tier_num][p.provider_name] = ok
        return results

    @property
    def provider_name(self) -> str:
        return "tiered_router"

    @property
    def default_model(self) -> str:
        p = self._pick(1)
        return p.default_model if p else "none"

    async def close(self) -> None:
        tasks = []
        for providers in self._tiers.values():
            for p in providers:
                tasks.append(p.close())
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("tier_router.closed")


# ---------------------------------------------------------------------------
# Singleton lifecycle
# ---------------------------------------------------------------------------


def get_router() -> TieredRouter:
    """Return the module-level router singleton. Raises if not initialised."""
    if _router_instance is None:
        raise RuntimeError(
            "TieredRouter not initialised. "
            "Call init_tier_router(settings) during app startup."
        )
    return _router_instance


async def init_tier_router(settings: Any) -> TieredRouter:
    """
    Build and store the router singleton from settings.
    Called from llm/client.py init_llm_client() when tiers are configured.
    """
    global _router_instance

    tiers = _build_tiers(settings)
    tier3_enabled = getattr(settings, "tier3_enabled", False)
    _router_instance = TieredRouter(tiers, tier3_enabled=tier3_enabled)
    return _router_instance


async def close_router() -> None:
    global _router_instance
    if _router_instance:
        await _router_instance.close()
        _router_instance = None


# ---------------------------------------------------------------------------
# Tier builder — constructs provider instances from settings
# ---------------------------------------------------------------------------

# Maps provider name → (factory_function, key_attr, model_attr)
_PROVIDER_REGISTRY: dict[str, tuple] = {
    "ollama":      ("llm.providers.ollama",        "OllamaProvider",             None,                    "ollama_default_model"),
    "groq":        ("llm.providers.openai_compat",  "make_groq_provider",         "groq_api_key",          "groq_default_model"),
    "openrouter":  ("llm.providers.openai_compat",  "make_openrouter_provider",   "openrouter_api_key",    "openrouter_default_model"),
    "cerebras":    ("llm.providers.openai_compat",  "make_cerebras_provider",     "cerebras_api_key",      "cerebras_default_model"),
    "sambanova":   ("llm.providers.openai_compat",  "make_sambanova_provider",    "sambanova_api_key",     "sambanova_default_model"),
    "gemini":      ("llm.providers.openai_compat",  "make_gemini_provider",       "gemini_api_key",        "gemini_default_model"),
    "deepseek":    ("llm.providers.openai_compat",  "make_deepseek_provider",     "deepseek_api_key",      "deepseek_default_model"),
    "mistral":     ("llm.providers.openai_compat",  "make_mistral_provider",      "mistral_api_key",       "mistral_default_model"),
    "anthropic":   ("llm.providers.anthropic_provider", "AnthropicProvider",      "anthropic_api_key",     "anthropic_default_model"),
}


def _build_provider(name: str, settings: Any) -> LLMProvider | None:
    """Instantiate a provider by name, returning None if key is missing."""
    entry = _PROVIDER_REGISTRY.get(name.lower().strip())
    if not entry:
        log.warning("tier_router.unknown_provider", name=name)
        return None

    module_path, callable_name, key_attr, model_attr = entry

    # Check API key (skip for Ollama which has no key requirement)
    if key_attr:
        api_key = getattr(settings, key_attr, "")
        if not api_key:
            log.warning(
                "tier_router.provider_skipped",
                provider=name,
                reason=f"{key_attr.upper()} not set in .env",
            )
            return None

    import importlib
    mod = importlib.import_module(module_path)
    factory = getattr(mod, callable_name)

    try:
        if name == "ollama":
            # OllamaProvider has a different constructor
            return factory(
                base_url=getattr(settings, "ollama_base_url", "http://localhost:11434"),
                default_model=getattr(settings, model_attr, "llama3.1:8b"),
                api_key=getattr(settings, "ollama_api_key", ""),
                timeout_seconds=getattr(settings, "ollama_timeout_seconds", 120),
            )
        elif name == "anthropic":
            return factory(
                api_key=getattr(settings, key_attr),
                default_model=getattr(settings, model_attr, "claude-haiku-4-5-20251001"),
                timeout_seconds=getattr(settings, "anthropic_timeout_seconds", 120),
            )
        else:
            return factory(
                api_key=getattr(settings, key_attr),
                default_model=getattr(settings, model_attr,
                    _default_model_for(name)),
            )
    except Exception as exc:
        log.warning(
            "tier_router.provider_init_failed",
            provider=name,
            error=str(exc),
        )
        return None


def _default_model_for(name: str) -> str:
    defaults = {
        "groq":       "llama-3.1-8b-instant",
        "openrouter": "meta-llama/llama-3.2-3b-instruct:free",
        "cerebras":   "llama-3.3-70b",
        "sambanova":  "Meta-Llama-3.3-70B-Instruct",
        "gemini":     "gemini-1.5-flash",
        "deepseek":   "deepseek-chat",
        "mistral":    "mistral-small-latest",
        "anthropic":  "claude-haiku-4-5-20251001",
    }
    return defaults.get(name, "default")


def _build_tiers(settings: Any) -> dict[int, list[LLMProvider]]:
    """Parse TIER1/2/3_PROVIDERS from settings and build provider instances."""
    tiers: dict[int, list[LLMProvider]] = {1: [], 2: [], 3: []}

    for tier_num in (1, 2, 3):
        attr = f"tier{tier_num}_providers"
        provider_list_str = getattr(settings, attr, "")
        if not provider_list_str:
            continue

        names = [n.strip() for n in provider_list_str.split(",") if n.strip()]
        for name in names:
            p = _build_provider(name, settings)
            if p is not None:
                tiers[tier_num].append(p)
                log.info(
                    "tier_router.provider_added",
                    tier=tier_num,
                    provider=name,
                    model=p.default_model,
                )

    return tiers
