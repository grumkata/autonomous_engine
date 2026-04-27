"""
llm/providers/catalog.py — Master registry of every known free/freemium provider.

Each entry documents:
  - base_url          OpenAI-compatible chat/completions endpoint
  - env_key           Environment variable name for the API key
  - default_model     Best free model to use when none is specified
  - free_models       Known free models on this provider
  - rate_limits       Approximate free-tier limits (rpm, tpm, daily_requests)
  - strengths         Task types / departments this provider excels at
  - speed_tier        "instant" | "fast" | "moderate" | "slow"
  - notes             Signup URL + any quirks

ADDING A PROVIDER
-----------------
1. Add a ProviderEntry below.
2. Add the matching PROVIDER_KEY = "" line to config.py.
3. The pool builder will auto-include it if the env var is set.

All providers here use the standard OpenAI /v1/chat/completions schema
unless `openai_compat=False` is noted (those require a custom provider file).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RateLimits:
    rpm: int = 0          # requests per minute  (0 = unknown)
    tpm: int = 0          # tokens per minute    (0 = unknown)
    daily_req: int = 0    # requests per day     (0 = unknown)
    daily_tok: int = 0    # tokens per day       (0 = unknown)
    monthly_tok: int = 0  # tokens per month     (0 = unknown)


@dataclass
class ProviderEntry:
    name: str                            # unique slug, matches config key prefix
    display_name: str                    # human label
    base_url: str                        # OpenAI-compat endpoint
    env_key: str                         # env var for api key (blank = keyless)
    default_model: str                   # recommended free model
    free_models: list[str]              # known no-cost models
    rate_limits: RateLimits
    strengths: list[str]                # task types / departments
    speed_tier: str                     # "instant" | "fast" | "moderate" | "slow"
    signup_url: str = ""
    notes: str = ""
    openai_compat: bool = True          # False = needs custom adapter
    requires_account_id: bool = False   # True = base_url needs {account_id}
    priority: int = 5                   # default routing priority (lower = better)


# ---------------------------------------------------------------------------
# The catalog — 20+ providers, all free or free-tier
# ---------------------------------------------------------------------------

CATALOG: dict[str, ProviderEntry] = {

    # ── Speed kings (instant LPU/WSE hardware) ──────────────────────────────

    "groq": ProviderEntry(
        name="groq",
        display_name="Groq (LPU)",
        base_url="https://api.groq.com/openai/v1",
        env_key="GROQ_API_KEY",
        default_model="llama-3.1-8b-instant",
        free_models=[
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "llama-3.3-70b-specdec",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "deepseek-r1-distill-llama-70b",
            "qwen-qwq-32b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ],
        rate_limits=RateLimits(rpm=30, tpm=6000, daily_req=14400),
        strengths=["red_team", "qa", "validation", "routing"],
        speed_tier="instant",
        signup_url="https://console.groq.com",
        notes="Custom LPU hardware. 300+ tok/s. Best for latency-sensitive tasks.",
        priority=1,
    ),

    "cerebras": ProviderEntry(
        name="cerebras",
        display_name="Cerebras (WSE)",
        base_url="https://api.cerebras.ai/v1",
        env_key="CEREBRAS_API_KEY",
        default_model="llama3.1-8b",
        free_models=[
            "llama3.1-8b",
            "llama-3.3-70b",
            "qwen-3-32b",
            "qwen-3-235b-a22b",
            "deepseek-r1-distill-llama-70b",
        ],
        rate_limits=RateLimits(rpm=30, tpm=60000, daily_tok=1_000_000),
        strengths=["research", "synthesis", "drafting"],
        speed_tier="instant",
        signup_url="https://cloud.cerebras.ai",
        notes="Wafer-scale engine. ~1000 tok/s on small models. 1M tok/day free.",
        priority=2,
    ),

    "sambanova": ProviderEntry(
        name="sambanova",
        display_name="SambaNova (RDU)",
        base_url="https://api.sambanova.ai/v1",
        env_key="SAMBANOVA_API_KEY",
        default_model="Meta-Llama-3.1-8B-Instruct",
        free_models=[
            "Meta-Llama-3.1-8B-Instruct",
            "Meta-Llama-3.1-70B-Instruct",
            "Meta-Llama-3.1-405B-Instruct",
            "Meta-Llama-3.3-70B-Instruct",
            "Qwen2.5-72B-Instruct",
            "Qwen2.5-Coder-32B-Instruct",
            "QwQ-32B",
            "DeepSeek-R1-Distill-Llama-70B",
            "DeepSeek-V3-0324",
        ],
        rate_limits=RateLimits(rpm=10, tpm=0, daily_req=0),
        strengths=["writing", "analysis", "research", "coding"],
        speed_tier="fast",
        signup_url="https://cloud.sambanova.ai",
        notes="Persistent free tier (no expiry). 405B model available free. 294 tok/s.",
        priority=3,
    ),

    # ── Free API key providers (OpenAI-compat) ──────────────────────────────

    "google": ProviderEntry(
        name="google",
        display_name="Google AI Studio (Gemini)",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        env_key="GOOGLE_API_KEY",
        default_model="gemini-2.0-flash",
        free_models=[
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        ],
        rate_limits=RateLimits(rpm=15, daily_req=1500, tpm=250000),
        strengths=["research", "analysis", "multimodal", "long_context"],
        speed_tier="fast",
        signup_url="https://ai.google.dev",
        notes="Most generous free tier. 1M token context. Multimodal. No credit card.",
        priority=2,
    ),

    "openrouter": ProviderEntry(
        name="openrouter",
        display_name="OpenRouter (multi-model)",
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        default_model="meta-llama/llama-3.2-3b-instruct:free",
        free_models=[
            "meta-llama/llama-3.2-3b-instruct:free",
            "meta-llama/llama-3.2-1b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "google/gemma-3-1b-it:free",
            "google/gemma-3-4b-it:free",
            "google/gemma-3-12b-it:free",
            "google/gemma-3-27b-it:free",
            "microsoft/phi-4-reasoning:free",
            "microsoft/phi-4-reasoning-plus:free",
            "deepseek/deepseek-r1:free",
            "deepseek/deepseek-v3-base:free",
            "qwen/qwen3-14b:free",
            "qwen/qwen3-30b-a3b:free",
            "qwen/qwen3-235b-a22b:free",
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "tngtech/deepseek-r1t-chimera:free",
        ],
        rate_limits=RateLimits(rpm=20, daily_req=50),
        strengths=["research", "writing", "red_team", "variety"],
        speed_tier="moderate",
        signup_url="https://openrouter.ai",
        notes="Gateway to 200+ models. Append :free to model name for zero-cost access.",
        priority=4,
    ),

    "nvidia": ProviderEntry(
        name="nvidia",
        display_name="NVIDIA NIM (100+ models)",
        base_url="https://integrate.api.nvidia.com/v1",
        env_key="NVIDIA_API_KEY",
        default_model="meta/llama-3.1-8b-instruct",
        free_models=[
            "meta/llama-3.1-8b-instruct",
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-405b-instruct",
            "meta/llama-3.3-70b-instruct",
            "meta/llama-4-scout-17b-16e-instruct",
            "deepseek-ai/deepseek-r1",
            "deepseek-ai/deepseek-v3",
            "qwen/qwen3-235b-a22b",
            "qwen/qwen3-32b",
            "moonshotai/kimi-k2-5",
            "zhipuai/glm-4.7",
            "openai/gpt-oss-120b",
            "google/gemma-3-27b-it",
            "microsoft/phi-4",
            "nvidia/nemotron-nano-8b-instruct",
            "nvidia/llama-3.1-nemotron-70b-instruct",
        ],
        rate_limits=RateLimits(rpm=40),
        strengths=["research", "coding", "analysis", "reasoning"],
        speed_tier="moderate",
        signup_url="https://build.nvidia.com",
        notes="1000 free credits on signup. 100+ models. API key starts with nvapi-.",
        priority=4,
    ),

    "together": ProviderEntry(
        name="together",
        display_name="Together AI",
        base_url="https://api.together.xyz/v1",
        env_key="TOGETHER_API_KEY",
        default_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        free_models=[
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
            "google/gemma-2-27b-it",
            "databricks/dbrx-instruct",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        ],
        rate_limits=RateLimits(rpm=60),
        strengths=["research", "writing", "coding", "analysis"],
        speed_tier="fast",
        signup_url="https://www.together.ai",
        notes="$25 free credits on signup. Very broad model selection.",
        priority=4,
    ),

    "fireworks": ProviderEntry(
        name="fireworks",
        display_name="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        env_key="FIREWORKS_API_KEY",
        default_model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        free_models=[
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "accounts/fireworks/models/llama-v3p1-405b-instruct",
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "accounts/fireworks/models/mixtral-8x7b-instruct",
            "accounts/fireworks/models/deepseek-v3",
            "accounts/fireworks/models/deepseek-r1",
            "accounts/fireworks/models/qwen2p5-72b-instruct",
            "accounts/fireworks/models/phi-3-5-vision-instruct",
            "accounts/fireworks/models/gemma2-9b-it",
        ],
        rate_limits=RateLimits(rpm=600, tpm=0),
        strengths=["coding", "analysis", "structured_output"],
        speed_tier="fast",
        signup_url="https://fireworks.ai",
        notes="Free credits on signup. High RPM. Good for parallel workloads.",
        priority=4,
    ),

    "mistral": ProviderEntry(
        name="mistral",
        display_name="Mistral AI",
        base_url="https://api.mistral.ai/v1",
        env_key="MISTRAL_API_KEY",
        default_model="mistral-small-latest",
        free_models=[
            "mistral-small-latest",
            "mistral-small-3.1-24b",
            "codestral-latest",
            "mistral-nemo",
            "open-mistral-7b",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b",
            "open-codestral-mamba",
        ],
        rate_limits=RateLimits(rpm=5),
        strengths=["coding", "reasoning", "european_languages"],
        speed_tier="moderate",
        signup_url="https://console.mistral.ai",
        notes="Free trial tier. Codestral is excellent for code tasks.",
        priority=5,
    ),

    "huggingface": ProviderEntry(
        name="huggingface",
        display_name="HuggingFace Inference API",
        base_url="https://api-inference.huggingface.co/v1",
        env_key="HF_API_KEY",
        default_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        free_models=[
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/Phi-3.5-mini-instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "google/gemma-2-27b-it",
            "HuggingFaceH4/zephyr-7b-beta",
            "NousResearch/Hermes-3-Llama-3.1-8B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        ],
        rate_limits=RateLimits(rpm=30),
        strengths=["research", "specialized_models", "variety"],
        speed_tier="slow",
        signup_url="https://huggingface.co",
        notes="Access to thousands of models. Cold starts on free tier. HF_TOKEN as env var.",
        priority=7,
    ),

    "github_models": ProviderEntry(
        name="github_models",
        display_name="GitHub Models",
        base_url="https://models.inference.ai.azure.com",
        env_key="GITHUB_TOKEN",
        default_model="gpt-4o-mini",
        free_models=[
            "gpt-4o-mini",
            "gpt-4o",
            "Llama-3.3-70B-Instruct",
            "Llama-3.2-90B-Vision-Instruct",
            "Phi-4",
            "Phi-3.5-MoE-instruct",
            "Mistral-Nemo",
            "Mistral-Large-2411",
            "Mistral-small",
            "Cohere-command-r-plus-08-2024",
            "AI21-Jamba-1.5-Large",
            "DeepSeek-R1",
            "DeepSeek-V3-0324",
        ],
        rate_limits=RateLimits(rpm=15, daily_req=150),
        strengths=["coding", "analysis", "qa"],
        speed_tier="moderate",
        signup_url="https://github.com/marketplace/models",
        notes="GitHub personal access token. GPT-4o-mini free. Good model variety.",
        priority=5,
    ),

    "deepseek": ProviderEntry(
        name="deepseek",
        display_name="DeepSeek Platform",
        base_url="https://api.deepseek.com",
        env_key="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        free_models=[
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-coder",
        ],
        rate_limits=RateLimits(rpm=60),
        strengths=["coding", "reasoning", "research"],
        speed_tier="moderate",
        signup_url="https://platform.deepseek.com",
        notes="5M free tokens on signup. Very cheap beyond that. Best for coding/reasoning.",
        priority=4,
    ),

    "cloudflare": ProviderEntry(
        name="cloudflare",
        display_name="Cloudflare Workers AI",
        base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        env_key="CF_API_KEY",
        default_model="@cf/meta/llama-3.1-8b-instruct",
        free_models=[
            "@cf/meta/llama-3.1-8b-instruct",
            "@cf/meta/llama-3.1-70b-instruct-fp8-fast",
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "@cf/mistral/mistral-7b-instruct-v0.2-lora",
            "@cf/google/gemma-3-12b-it",
            "@cf/qwen/qwq-32b",
            "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
            "@cf/microsoft/phi-4-multimodal-instruct",
        ],
        rate_limits=RateLimits(daily_req=10000),
        strengths=["edge_inference", "low_latency", "variety"],
        speed_tier="fast",
        signup_url="https://dash.cloudflare.com",
        notes="10k free req/day. Needs CF_ACCOUNT_ID env var too. Edge-distributed.",
        requires_account_id=True,
        priority=5,
    ),

    "moonshot": ProviderEntry(
        name="moonshot",
        display_name="Moonshot AI (Kimi)",
        base_url="https://api.moonshot.cn/v1",
        env_key="MOONSHOT_API_KEY",
        default_model="moonshot-v1-8k",
        free_models=[
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
            "kimi-k2-0711-preview",
        ],
        rate_limits=RateLimits(rpm=3),
        strengths=["long_context", "chinese", "research"],
        speed_tier="moderate",
        signup_url="https://platform.moonshot.cn",
        notes="Free credits on signup. 128K context. Kimi K2 available.",
        priority=6,
    ),

    "zhipu": ProviderEntry(
        name="zhipu",
        display_name="Zhipu AI (GLM)",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        env_key="ZHIPU_API_KEY",
        default_model="glm-4-flash",
        free_models=[
            "glm-4-flash",
            "glm-4-flashx",
            "glm-4-air",
            "glm-4-plus",
            "glm-z1-flash",
        ],
        rate_limits=RateLimits(rpm=5),
        strengths=["chinese", "analysis", "coding"],
        speed_tier="fast",
        signup_url="https://open.bigmodel.cn",
        notes="GLM-4-Flash is completely free. Excellent for Chinese language tasks.",
        priority=6,
    ),

    "hyperbolic": ProviderEntry(
        name="hyperbolic",
        display_name="Hyperbolic AI",
        base_url="https://api.hyperbolic.xyz/v1",
        env_key="HYPERBOLIC_API_KEY",
        default_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        free_models=[
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-V3",
        ],
        rate_limits=RateLimits(rpm=60),
        strengths=["research", "reasoning", "large_models"],
        speed_tier="moderate",
        signup_url="https://app.hyperbolic.xyz",
        notes="$10 free credits on signup. Access to 405B and R1.",
        priority=5,
    ),

    "novita": ProviderEntry(
        name="novita",
        display_name="Novita AI",
        base_url="https://api.novita.ai/v3/openai",
        env_key="NOVITA_API_KEY",
        default_model="meta-llama/llama-3.1-8b-instruct",
        free_models=[
            "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-r1",
            "deepseek/deepseek-v3-0324",
            "qwen/qwen2.5-72b-instruct",
            "mistralai/mistral-7b-instruct-v0.3",
        ],
        rate_limits=RateLimits(rpm=60),
        strengths=["writing", "coding", "research"],
        speed_tier="moderate",
        signup_url="https://novita.ai",
        notes="Free credits on signup. Large model catalog. Cost-effective.",
        priority=6,
    ),

    "nebius": ProviderEntry(
        name="nebius",
        display_name="Nebius AI Studio",
        base_url="https://api.studio.nebius.ai/v1/",
        env_key="NEBIUS_API_KEY",
        default_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        free_models=[
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "deepseek-ai/DeepSeek-R1",
        ],
        rate_limits=RateLimits(rpm=30),
        strengths=["research", "writing", "analysis"],
        speed_tier="moderate",
        signup_url="https://studio.nebius.ai",
        notes="Free trial credits. European data residency available.",
        priority=6,
    ),

    "featherless": ProviderEntry(
        name="featherless",
        display_name="Featherless AI",
        base_url="https://api.featherless.ai/v1",
        env_key="FEATHERLESS_API_KEY",
        default_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        free_models=[
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Qwen/Qwen2.5-7B-Instruct",
            "NousResearch/Hermes-3-Llama-3.1-8B",
        ],
        rate_limits=RateLimits(rpm=20),
        strengths=["writing", "roleplay", "creative"],
        speed_tier="moderate",
        signup_url="https://featherless.ai",
        notes="Focused on HuggingFace model hosting. Good for niche fine-tunes.",
        priority=7,
    ),

    "kluster": ProviderEntry(
        name="kluster",
        display_name="Kluster AI",
        base_url="https://api.kluster.ai/v1",
        env_key="KLUSTER_API_KEY",
        default_model="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
        free_models=[
            "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
            "klusterai/Meta-Llama-3.3-70B-Instruct-Turbo",
            "klusterai/Meta-Llama-3.1-405B-Instruct-Turbo",
            "klusterai/DeepSeek-R1-Distill-Llama-70B",
        ],
        rate_limits=RateLimits(rpm=30),
        strengths=["coding", "analysis"],
        speed_tier="fast",
        signup_url="https://kluster.ai",
        notes="Free credits on signup. Optimised for batch inference.",
        priority=6,
    ),

    "siliconflow": ProviderEntry(
        name="siliconflow",
        display_name="SiliconFlow",
        base_url="https://api.siliconflow.cn/v1",
        env_key="SILICONFLOW_API_KEY",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        free_models=[
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/QwQ-32B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-V3",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "internlm/internlm2_5-7b-chat",
            "THUDM/glm-4-9b-chat",
        ],
        rate_limits=RateLimits(rpm=20),
        strengths=["chinese", "coding", "research"],
        speed_tier="moderate",
        signup_url="https://cloud.siliconflow.cn",
        notes="Chinese provider. Strong Qwen and DeepSeek support. Free tier generous.",
        priority=6,
    ),

    # ── Keyless / anonymous providers ───────────────────────────────────────

    "pollinations": ProviderEntry(
        name="pollinations",
        display_name="Pollinations.ai (keyless)",
        base_url="https://text.pollinations.ai/openai",
        env_key="",    # no key required
        default_model="openai",
        free_models=[
            "openai",
            "openai-fast",
            "openai-large",
            "mistral",
            "mistral-large",
            "deepseek-v3",
            "qwen-coder",
            "gemini-2.5-flash-lite",
            "claude-haiku-4.5",
            "amazon-nova-micro",
            "kimi-k2-thinking",
        ],
        rate_limits=RateLimits(rpm=20),
        strengths=["quick_tasks", "fallback", "variety"],
        speed_tier="moderate",
        signup_url="https://pollinations.ai",
        notes="No signup, no key. Completely anonymous. Good as last-resort fallback.",
        priority=9,
    ),

    "llm7": ProviderEntry(
        name="llm7",
        display_name="LLM7.io (keyless)",
        base_url="https://llm7.io/v1",
        env_key="",    # no key required
        default_model="mistral-small-2503",
        free_models=[
            "mistral-small-2503",
            "mistral-small-3.1-24b",
            "open-mixtral-8x7b",
            "llama-3.1-8b-instruct-fp8",
            "llama-4-scout-17b-16e-instruct",
            "deepseek-r1-0528",
            "gpt-4.1-nano-2025-04-14",
            "qwen2.5-coder-32b-instruct",
            "grok-3-mini-high",
            "codestral-2501",
        ],
        rate_limits=RateLimits(rpm=150),
        strengths=["quick_tasks", "coding", "fallback"],
        speed_tier="moderate",
        signup_url="https://llm7.io",
        notes="No signup needed at all. 150 rpm free. Great anonymous fallback.",
        priority=8,
    ),

    # ── Local ───────────────────────────────────────────────────────────────

    "ollama": ProviderEntry(
        name="ollama",
        display_name="Ollama (local)",
        base_url="http://localhost:11434",
        env_key="",    # no remote key; uses local binary
        default_model="llama3.1:8b",
        free_models=[
            "llama3.1:8b",
            "llama3.1:70b",
            "llama3.3:70b",
            "deepseek-r1:7b",
            "deepseek-r1:32b",
            "qwen2.5:7b",
            "qwen2.5:32b",
            "mistral:7b",
            "codellama:7b",
            "phi4:14b",
        ],
        rate_limits=RateLimits(),   # hardware-limited, not rate-limited
        strengths=["privacy", "offline", "unrestricted"],
        speed_tier="slow",          # depends on hardware
        signup_url="https://ollama.com",
        notes="Fully local. No rate limits. Speed depends on your GPU/CPU.",
        priority=10,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_provider(name: str) -> Optional[ProviderEntry]:
    return CATALOG.get(name.lower())


def list_providers() -> list[str]:
    return list(CATALOG.keys())


def providers_for_strength(strength: str) -> list[ProviderEntry]:
    return [p for p in CATALOG.values() if strength in p.strengths]
