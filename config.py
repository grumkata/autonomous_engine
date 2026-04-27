from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "Autonomous AI Engine"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Database
    database_url: str = "sqlite+aiosqlite:///./engine.db"

    # Vector store
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_persist_dir: str = "./chroma_data"

    # ── LLM provider selection ────────────────────────────────────────────────
    # Options:
    #   "ollama"      — single local Ollama instance (default, zero config)
    #   "groq"        — single Groq provider
    #   "openrouter"  — single OpenRouter provider
    #   "pool"        — manual pool via LLM_POOL_CONFIG JSON array
    #   "auto"        — AUTO-BUILD pool from ALL env vars below (recommended)
    #                   Just set the keys you have — engine uses them all.
    llm_provider: str = "ollama"

    # ── Manual pool config (LLM_PROVIDER=pool only) ───────────────────────────
    # JSON array of slot objects. See llm/router.py ProviderSlotConfig for fields.
    llm_pool_config: str = "[]"

    # ── Meta-orchestrator (the routing brain) ─────────────────────────────────
    # A small, fast, free model that decides WHICH provider slot handles each task.
    # Runs in ~100ms. Falls back to score-based routing on any failure.
    # Recommended: groq + llama-3.1-8b-instant (fastest free model available).
    # Set META_ORCHESTRATOR_PROVIDER="" to disable and use score-only routing.
    meta_orchestrator_provider: str = "groq"
    meta_orchestrator_model: str = "llama-3.1-8b-instant"
    meta_orchestrator_api_key: str = ""   # leave blank to reuse GROQ_API_KEY slot

    # ── Speed-tier routing hints ──────────────────────────────────────────────
    # Departments/task-types that should always use the fastest available slot.
    fast_departments: str = "red_team,qa,validation"    # comma-separated
    smart_departments: str = "research,analysis,writing" # prefer capable models

    # ── Ollama (local) ────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.1:8b"
    ollama_timeout_seconds: int = 600
    ollama_api_key: str = ""

    # ── Groq — fast LPU inference, free tier ─────────────────────────────────
    # Sign up: https://console.groq.com  (no credit card)
    # Free: 30 RPM, 6K TPM, 14400 req/day
    groq_api_key: str = ""
    groq_default_model: str = "llama-3.1-8b-instant"

    # ── Cerebras — wafer-scale engine, ~1000 tok/s ────────────────────────────
    # Sign up: https://cloud.cerebras.ai  (no credit card)
    # Free: 30 RPM, 60K TPM, 1M tok/day
    cerebras_api_key: str = ""
    cerebras_default_model: str = "llama3.1-8b"

    # ── SambaNova — RDU hardware, 405B free ───────────────────────────────────
    # Sign up: https://cloud.sambanova.ai  (persistent free tier, no expiry)
    # Free: 10-30 RPM depending on model size
    sambanova_api_key: str = ""
    sambanova_default_model: str = "Meta-Llama-3.1-8B-Instruct"

    # ── Google AI Studio — Gemini, most generous free tier ────────────────────
    # Sign up: https://ai.google.dev  (no credit card)
    # Free: Gemini Flash 1500 req/day, 250K TPM
    google_api_key: str = ""
    google_default_model: str = "gemini-2.0-flash"

    # ── OpenRouter — gateway to 200+ models ───────────────────────────────────
    # Sign up: https://openrouter.ai  (no credit card)
    # Append :free to model name for zero-cost access
    openrouter_api_key: str = ""
    openrouter_default_model: str = "meta-llama/llama-3.2-3b-instruct:free"

    # ── NVIDIA NIM — 100+ models, 1000 free credits ───────────────────────────
    # Sign up: https://build.nvidia.com  (join NVIDIA Developer Program)
    # Free: 1000 credits on signup, 40 RPM
    nvidia_api_key: str = ""
    nvidia_default_model: str = "meta/llama-3.1-8b-instruct"

    # ── Together AI — broad model selection, $25 free credits ─────────────────
    # Sign up: https://www.together.ai
    together_api_key: str = ""
    together_default_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    # ── Fireworks AI — high RPM, free credits ─────────────────────────────────
    # Sign up: https://fireworks.ai
    fireworks_api_key: str = ""
    fireworks_default_model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct"

    # ── Mistral AI — free trial, strong coding ────────────────────────────────
    # Sign up: https://console.mistral.ai
    mistral_api_key: str = ""
    mistral_default_model: str = "mistral-small-latest"

    # ── HuggingFace Inference API — thousands of models ───────────────────────
    # Sign up: https://huggingface.co  (no credit card, create access token)
    # Free: rate-limited, cold starts on popular models
    hf_api_key: str = ""
    hf_default_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # ── GitHub Models — GPT-4o-mini free ─────────────────────────────────────
    # Sign up: https://github.com/marketplace/models  (needs GitHub account)
    # Free: 15 RPM, 150 req/day
    github_token: str = ""
    github_models_default_model: str = "gpt-4o-mini"

    # ── DeepSeek — best coding model, 5M free tokens ─────────────────────────
    # Sign up: https://platform.deepseek.com  (no credit card required initially)
    deepseek_api_key: str = ""
    deepseek_default_model: str = "deepseek-chat"

    # ── Cloudflare Workers AI — edge inference, 10K req/day ───────────────────
    # Sign up: https://dash.cloudflare.com  (needs Workers plan)
    # Requires BOTH CF_API_KEY and CF_ACCOUNT_ID
    cf_api_key: str = ""
    cf_account_id: str = ""
    cf_default_model: str = "@cf/meta/llama-3.1-8b-instruct"

    # ── Moonshot AI (Kimi) — 128K context, free credits ───────────────────────
    # Sign up: https://platform.moonshot.cn
    moonshot_api_key: str = ""
    moonshot_default_model: str = "moonshot-v1-8k"

    # ── Zhipu AI (GLM) — GLM-4-Flash completely free ──────────────────────────
    # Sign up: https://open.bigmodel.cn
    zhipu_api_key: str = ""
    zhipu_default_model: str = "glm-4-flash"

    # ── Hyperbolic AI — $10 free credits, 405B access ─────────────────────────
    # Sign up: https://app.hyperbolic.xyz
    hyperbolic_api_key: str = ""
    hyperbolic_default_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # ── Novita AI — free credits, large catalog ───────────────────────────────
    # Sign up: https://novita.ai
    novita_api_key: str = ""
    novita_default_model: str = "meta-llama/llama-3.1-8b-instruct"

    # ── Nebius AI Studio — European provider, free trial ─────────────────────
    # Sign up: https://studio.nebius.ai
    nebius_api_key: str = ""
    nebius_default_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # ── Featherless AI — HuggingFace model hosting ────────────────────────────
    # Sign up: https://featherless.ai
    featherless_api_key: str = ""
    featherless_default_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # ── Kluster AI — batch-optimised inference ────────────────────────────────
    # Sign up: https://kluster.ai
    kluster_api_key: str = ""
    kluster_default_model: str = "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo"

    # ── SiliconFlow — Chinese provider, strong Qwen/DeepSeek ─────────────────
    # Sign up: https://cloud.siliconflow.cn
    siliconflow_api_key: str = ""
    siliconflow_default_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # ── Keyless providers (always available, no signup needed) ────────────────
    # These are included automatically in auto mode.
    use_pollinations: bool = True   # https://pollinations.ai  — anonymous, unlimited
    use_llm7: bool = True           # https://llm7.io          — anonymous, 150 rpm

    # ── Orchestration limits ──────────────────────────────────────────────────
    max_concurrent_tasks: int = 8
    task_default_retry_limit: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
