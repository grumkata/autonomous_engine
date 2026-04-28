"""
llm/tier_router.py — Tiered provider routing for the autonomous engine.

Routes each LLM call to the cheapest provider capable of handling the task,
escalating automatically on retry.

Tier structure
--------------
TIER 1 — Free, no practical limits  (general work, grunt tasks)
TIER 2 — Free, rate-limited         (important tasks needing real capability)
TIER 3 — Paid / near-free           (heavy/final tasks, last resort)
LOCAL   — Self-hosted               (always free, hardware-limited)

Auto-escalation
---------------
  attempt 1 → assigned tier
  attempt 2 → tier + 1
  attempt 3 → tier + 2  (reaches tier 3 only if TIER3_ENABLED=true)

Round-robin within a tier spreads load across quotas.

Configuration (.env)
---------------------
  TIER1_PROVIDERS=groq,cerebras,siliconflow,llm7,pollinations,ollama
  TIER2_PROVIDERS=gemini,sambanova,openrouter,mistral,zhipu,huggingface
  TIER3_PROVIDERS=anthropic,deepseek,deepinfra
  TIER3_ENABLED=false

Provider catalogue (all supported names)
-----------------------------------------
  TIER 1 free / unlimited:
    groq, cerebras, siliconflow, llm7, kluster, bazaarlink,
    pollinations, ollama_cloud, featherless, github_models, huggingface

  TIER 2 free / rate-limited:
    gemini, sambanova, openrouter, mistral, cohere, nvidia_nim,
    zhipu, moonshot, together, ai21, ollama

  TIER 3 paid / near-free:
    anthropic, deepseek, openai, xai, fireworks, hyperbolic,
    deepinfra, perplexity, aimlapi, yi, qwen, novita, lambda

  LOCAL (always free, add to any tier):
    lmstudio, jan, localai, llamacpp, vllm, sglang
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

import structlog

from llm.providers.base import LLMProvider

log = structlog.get_logger(__name__)

_router_instance: "TieredRouter | None" = None


# ---------------------------------------------------------------------------
# Provider registry — maps name → (factory_module, factory_fn, key_attr, model_attr)
# key_attr=None means no API key required (local or keyless)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[str, str, str | None, str]] = {
    # ── TIER 1: free, unlimited ───────────────────────────────────────────
    "groq":          ("llm.providers.openai_compat", "make_groq_provider",          "groq_api_key",          "groq_default_model"),
    "cerebras":      ("llm.providers.openai_compat", "make_cerebras_provider",      "cerebras_api_key",      "cerebras_default_model"),
    "siliconflow":   ("llm.providers.openai_compat", "make_siliconflow_provider",   "siliconflow_api_key",   "siliconflow_default_model"),
    "llm7":          ("llm.providers.openai_compat", "make_llm7_provider",          None,                    "llm7_default_model"),
    "kluster":       ("llm.providers.openai_compat", "make_kluster_provider",       "kluster_api_key",       "kluster_default_model"),
    "bazaarlink":    ("llm.providers.openai_compat", "make_bazaarlink_provider",    "bazaarlink_api_key",    "bazaarlink_default_model"),
    "pollinations":  ("llm.providers.openai_compat", "make_pollinations_provider",  None,                    "pollinations_default_model"),
    "ollama_cloud":  ("llm.providers.openai_compat", "make_ollama_cloud_provider",  "ollama_cloud_api_key",  "ollama_cloud_default_model"),
    "featherless":   ("llm.providers.openai_compat", "make_featherless_provider",   "featherless_api_key",   "featherless_default_model"),
    "github_models": ("llm.providers.openai_compat", "make_github_models_provider", "github_models_api_key", "github_models_default_model"),
    "huggingface":   ("llm.providers.openai_compat", "make_huggingface_provider",   "huggingface_api_key",   "huggingface_default_model"),

    # ── TIER 2: free, rate-limited ────────────────────────────────────────
    "gemini":        ("llm.providers.openai_compat", "make_gemini_provider",        "gemini_api_key",        "gemini_default_model"),
    "sambanova":     ("llm.providers.openai_compat", "make_sambanova_provider",     "sambanova_api_key",     "sambanova_default_model"),
    "openrouter":    ("llm.providers.openai_compat", "make_openrouter_provider",    "openrouter_api_key",    "openrouter_default_model"),
    "mistral":       ("llm.providers.openai_compat", "make_mistral_provider",       "mistral_api_key",       "mistral_default_model"),
    "cohere":        ("llm.providers.openai_compat", "make_cohere_provider",        "cohere_api_key",        "cohere_default_model"),
    "nvidia_nim":    ("llm.providers.openai_compat", "make_nvidia_nim_provider",    "nvidia_nim_api_key",    "nvidia_nim_default_model"),
    "zhipu":         ("llm.providers.openai_compat", "make_zhipu_provider",         "zhipu_api_key",         "zhipu_default_model"),
    "moonshot":      ("llm.providers.openai_compat", "make_moonshot_provider",      "moonshot_api_key",      "moonshot_default_model"),
    "together":      ("llm.providers.openai_compat", "make_together_provider",      "together_api_key",      "together_default_model"),
    "ai21":          ("llm.providers.openai_compat", "make_ai21_provider",          "ai21_api_key",          "ai21_default_model"),
    "ollama":        ("llm.providers.ollama",         "OllamaProvider",             None,                    "ollama_default_model"),

    # ── TIER 3: paid / near-free ──────────────────────────────────────────
    "anthropic":     ("llm.providers.anthropic_provider", "AnthropicProvider",      "anthropic_api_key",     "anthropic_default_model"),
    "deepseek":      ("llm.providers.openai_compat", "make_deepseek_provider",      "deepseek_api_key",      "deepseek_default_model"),
    "openai":        ("llm.providers.openai_compat", "make_openai_provider",        "openai_api_key",        "openai_default_model"),
    "xai":           ("llm.providers.openai_compat", "make_xai_provider",           "xai_api_key",           "xai_default_model"),
    "fireworks":     ("llm.providers.openai_compat", "make_fireworks_provider",     "fireworks_api_key",     "fireworks_default_model"),
    "hyperbolic":    ("llm.providers.openai_compat", "make_hyperbolic_provider",    "hyperbolic_api_key",    "hyperbolic_default_model"),
    "deepinfra":     ("llm.providers.openai_compat", "make_deepinfra_provider",     "deepinfra_api_key",     "deepinfra_default_model"),
    "perplexity":    ("llm.providers.openai_compat", "make_perplexity_provider",    "perplexity_api_key",    "perplexity_default_model"),
    "aimlapi":       ("llm.providers.openai_compat", "make_aimlapi_provider",       "aimlapi_api_key",       "aimlapi_default_model"),
    "yi":            ("llm.providers.openai_compat", "make_yi_provider",            "yi_api_key",            "yi_default_model"),
    "qwen":          ("llm.providers.openai_compat", "make_alibaba_qwen_provider",  "qwen_api_key",          "qwen_default_model"),
    "novita":        ("llm.providers.openai_compat", "make_novita_provider",        "novita_api_key",        "novita_default_model"),
    "lambda":        ("llm.providers.openai_compat", "make_lambda_provider",        "lambda_api_key",        "lambda_default_model"),

    # ── LOCAL: self-hosted ────────────────────────────────────────────────
    "lmstudio":      ("llm.providers.openai_compat", "make_lmstudio_provider",      None,                    "lmstudio_base_url"),
    "jan":           ("llm.providers.openai_compat", "make_jan_provider",           None,                    "jan_base_url"),
    "localai":       ("llm.providers.openai_compat", "make_localai_provider",       None,                    "localai_base_url"),
    "llamacpp":      ("llm.providers.openai_compat", "make_llamacpp_provider",      None,                    "llamacpp_base_url"),
    "vllm":          ("llm.providers.openai_compat", "make_vllm_provider",          None,                    "vllm_base_url"),
    "sglang":        ("llm.providers.openai_compat", "make_sglang_provider",        None,                    "sglang_base_url"),
}


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------


def task_tier(task: Any, settings: Any | None = None) -> int:
    if settings is None:
        from config import get_settings
        settings = get_settings()

    if hasattr(task, "llm_tier") and isinstance(task.llm_tier, int) and task.llm_tier in (1, 2, 3):
        return task.llm_tier

    t1_max = getattr(settings, "tier1_max_threshold", 0.72)
    t2_max = getattr(settings, "tier2_max_threshold", 0.84)
    qt     = getattr(task, "quality_threshold", 0.75)

    if qt >= t2_max:
        return 3
    if qt >= t1_max:
        return 2
    return 1


# ---------------------------------------------------------------------------
# TieredRouter
# ---------------------------------------------------------------------------


class TieredRouter:
    def __init__(self, tiers: dict[int, list[LLMProvider]], tier3_enabled: bool = False) -> None:
        self._tiers: dict[int, list[LLMProvider]] = {k: v for k, v in tiers.items() if v}
        self._tier3_enabled = tier3_enabled
        self._rr: defaultdict[int, int] = defaultdict(int)

        for tier_num, providers in sorted(self._tiers.items()):
            enabled = tier_num < 3 or self._tier3_enabled
            log.info(
                "tier_router.tier_loaded",
                tier=tier_num,
                providers=[p.provider_name for p in providers],
                enabled=enabled,
            )

    # ── Provider selection ─────────────────────────────────────────────────

    def _pick(self, tier: int) -> LLMProvider | None:
        if tier == 3 and not self._tier3_enabled:
            tier = 2

        candidates = self._tiers.get(tier, [])
        if candidates:
            idx = self._rr[tier] % len(candidates)
            self._rr[tier] = idx + 1
            return candidates[idx]

        for fallback in sorted(self._tiers.keys()):
            if fallback == 3 and not self._tier3_enabled:
                continue
            if self._tiers[fallback]:
                p = self._tiers[fallback][0]
                log.warning("tier_router.fallback", requested=tier, fallback=fallback, provider=p.provider_name)
                return p
        return None

    def provider_for(self, tier: int, attempt_number: int = 1) -> LLMProvider | None:
        escalated = min(tier + max(0, attempt_number - 1), 3)
        if escalated != tier:
            log.info("tier_router.escalating", from_tier=tier, to_tier=escalated, attempt=attempt_number)
        return self._pick(escalated)

    # ── chat_json ──────────────────────────────────────────────────────────

    async def chat_json(
        self,
        messages: list,
        *,
        tier: int = 1,
        attempt_number: int = 1,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        retry_limit: int = 3,
    ) -> tuple[dict, Any]:
        from llm.client import LLMClient
        provider = self.provider_for(tier, attempt_number)
        if provider is None:
            raise RuntimeError(f"TieredRouter: no provider available for tier {tier}. Check TIER*_PROVIDERS in .env")
        log.debug("tier_router.routing", tier=tier, attempt=attempt_number, provider=provider.provider_name)
        return await LLMClient(provider).chat_json(
            messages, model=model, temperature=temperature, max_tokens=max_tokens, retry_limit=retry_limit,
        )

    async def chat(self, messages: list, *, tier: int = 1, attempt_number: int = 1, **kwargs) -> Any:
        from llm.client import LLMClient
        provider = self.provider_for(tier, attempt_number)
        if provider is None:
            raise RuntimeError("TieredRouter: no providers configured.")
        return await LLMClient(provider).chat(messages, **kwargs)

    async def complete(self, prompt: str, *, tier: int = 1, attempt_number: int = 1, **kwargs) -> Any:
        from llm.client import LLMClient
        provider = self.provider_for(tier, attempt_number)
        if provider is None:
            raise RuntimeError("TieredRouter: no providers configured.")
        return await LLMClient(provider).complete(prompt, **kwargs)

    async def ping(self) -> dict[int, dict[str, bool]]:
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

    async def close(self) -> None:
        tasks = [p.close() for providers in self._tiers.values() for p in providers]
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("tier_router.closed")

    @property
    def provider_name(self) -> str:
        return "tiered_router"

    @property
    def default_model(self) -> str:
        p = self._pick(1)
        return p.default_model if p else "none"


# ---------------------------------------------------------------------------
# Singleton lifecycle
# ---------------------------------------------------------------------------


def get_router() -> TieredRouter:
    if _router_instance is None:
        raise RuntimeError("TieredRouter not initialised. Call init_tier_router(settings) during app startup.")
    return _router_instance


async def init_tier_router(settings: Any) -> TieredRouter:
    global _router_instance
    tiers = _build_tiers(settings)
    _router_instance = TieredRouter(tiers, tier3_enabled=getattr(settings, "tier3_enabled", False))
    return _router_instance


async def close_router() -> None:
    global _router_instance
    if _router_instance:
        await _router_instance.close()
        _router_instance = None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_provider(name: str, settings: Any) -> LLMProvider | None:
    name = name.lower().strip()
    entry = _REGISTRY.get(name)
    if not entry:
        log.warning("tier_router.unknown_provider", name=name,
                    hint=f"Valid providers: {', '.join(sorted(_REGISTRY.keys()))}")
        return None

    module_path, callable_name, key_attr, model_attr = entry

    # Check API key (skip for keyless providers)
    api_key = ""
    if key_attr:
        api_key = getattr(settings, key_attr, "")
        if not api_key:
            log.warning("tier_router.provider_skipped", provider=name,
                        reason=f"{key_attr.upper()} not set in .env")
            return None

    import importlib
    mod = importlib.import_module(module_path)
    factory = getattr(mod, callable_name)

    try:
        model = getattr(settings, model_attr, None) or _default_model(name)

        # Special cases with different constructors
        if name == "ollama":
            return factory(
                base_url=getattr(settings, "ollama_base_url", "http://localhost:11434"),
                default_model=model,
                api_key=getattr(settings, "ollama_api_key", ""),
                timeout_seconds=getattr(settings, "ollama_timeout_seconds", 120),
            )
        elif name == "anthropic":
            return factory(
                api_key=api_key,
                default_model=model,
                timeout_seconds=getattr(settings, "anthropic_timeout_seconds", 120),
            )
        # Local providers without API keys use base_url as model_attr
        elif name in ("lmstudio", "jan", "localai", "llamacpp", "vllm", "sglang"):
            base_url = model  # model_attr stores the base_url for local providers
            return factory(base_url=base_url)
        # Keyless cloud providers
        elif name in ("llm7", "pollinations"):
            return factory(default_model=model)
        else:
            return factory(api_key=api_key, default_model=model)

    except Exception as exc:
        log.warning("tier_router.provider_init_failed", provider=name, error=str(exc))
        return None


def _default_model(name: str) -> str:
    defaults = {
        "groq":          "llama-3.1-8b-instant",
        "cerebras":      "llama-3.3-70b",
        "siliconflow":   "Qwen/Qwen3-8B",
        "llm7":          "deepseek-ai/DeepSeek-R1",
        "kluster":       "klusterai/Meta-Llama-3.3-70B-Instruct-Turbo",
        "bazaarlink":    "auto:free",
        "pollinations":  "openai-large",
        "ollama_cloud":  "deepseek-v3.2",
        "featherless":   "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "github_models": "meta-llama-3.3-70b-instruct",
        "huggingface":   "meta-llama/Llama-3.3-70B-Instruct",
        "gemini":        "gemini-2.5-flash",
        "sambanova":     "Meta-Llama-3.3-70B-Instruct",
        "openrouter":    "meta-llama/llama-3.3-70b-instruct:free",
        "mistral":       "mistral-small-latest",
        "cohere":        "command-r-plus",
        "nvidia_nim":    "meta/llama-3.3-70b-instruct",
        "zhipu":         "glm-4-flash",
        "moonshot":      "moonshot-v1-128k",
        "together":      "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "ai21":          "jamba-1.6-mini",
        "ollama":        "llama3.1:8b",
        "anthropic":     "claude-haiku-4-5-20251001",
        "deepseek":      "deepseek-chat",
        "openai":        "gpt-4.1-mini",
        "xai":           "grok-3-fast",
        "fireworks":     "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "hyperbolic":    "meta-llama/Llama-3.3-70B-Instruct",
        "deepinfra":     "meta-llama/Llama-3.3-70B-Instruct",
        "perplexity":    "sonar",
        "aimlapi":       "gpt-4o",
        "yi":            "yi-large",
        "qwen":          "qwen-plus",
        "novita":        "meta-llama/llama-3.3-70b-instruct",
        "lambda":        "llama3.3-70b-instruct-fp8",
        "lmstudio":      "http://localhost:1234",
        "jan":           "http://localhost:1337",
        "localai":       "http://localhost:8080",
        "llamacpp":      "http://localhost:8080",
        "vllm":          "http://localhost:8000",
        "sglang":        "http://localhost:30000",
    }
    return defaults.get(name, "default")


def _build_tiers(settings: Any) -> dict[int, list[LLMProvider]]:
    tiers: dict[int, list[LLMProvider]] = {1: [], 2: [], 3: []}
    for tier_num in (1, 2, 3):
        provider_str = getattr(settings, f"tier{tier_num}_providers", "")
        for name in [n.strip() for n in provider_str.split(",") if n.strip()]:
            p = _build_provider(name, settings)
            if p is not None:
                tiers[tier_num].append(p)
                log.info("tier_router.provider_added", tier=tier_num, provider=name, model=p.default_model)
    return tiers
