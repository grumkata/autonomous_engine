from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

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

    # ── LLM Mode ──────────────────────────────────────────────────────────────
    # "single"  — use one provider for everything (original behaviour)
    # "tiered"  — use the tier routing system (recommended)
    llm_mode: str = "single"

    # ── Single-provider mode ──────────────────────────────────────────────────
    llm_provider: str = "ollama"

    # ── Tiered provider pools ────────────────────────────────────────────────
    #
    # TIER 1 — Free, no practical limits. Handles majority of work.
    #   Recommended: groq,cerebras,sambanova,ollama
    #   Tasks: quality_threshold < tier1_max_threshold AND attempt 1
    #
    # TIER 2 — Free, rate-limited. For more important tasks.
    #   Recommended: gemini,openrouter (70B :free models)
    #   gemini-1.5-flash is 15 RPM / 1M tokens/day free — extremely generous
    #
    # TIER 3 — Paid, opt-in. TIER3_ENABLED=false by default.
    #   Recommended: anthropic,deepseek (deepseek is near-free at $0.07/1M)
    #   Only activates when TIER3_ENABLED=true or toggled per-project
    #
    tier1_providers: str = "groq,ollama"
    tier2_providers: str = "gemini,openrouter"
    tier3_providers: str = "anthropic,deepseek"
    tier3_enabled: bool = False

    # Threshold boundaries for tier assignment
    tier1_max_threshold: float = 0.72   # below this → tier 1
    tier2_max_threshold: float = 0.84   # below this → tier 2, above → tier 3

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.1:8b"
    ollama_timeout_seconds: int = 120
    ollama_api_key: str = ""

    # ── Groq (free tier) ──────────────────────────────────────────────────────
    # console.groq.com — no credit card. 14,400 req/day on 8B, ~500/day on 70B+
    # Tier 1 model: llama-3.1-8b-instant (very fast)
    # Tier 2 model: llama-3.3-70b-versatile (change groq_default_model)
    groq_api_key: str = ""
    groq_default_model: str = "llama-3.1-8b-instant"

    # ── OpenRouter (free :free models) ───────────────────────────────────────
    # openrouter.ai — no credit card for free models
    # Good free options: meta-llama/llama-3.3-70b-instruct:free
    openrouter_api_key: str = ""
    openrouter_default_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    # ── Cerebras Cloud (free, 1000+ tokens/sec) ──────────────────────────────
    # cloud.cerebras.ai — free, no credit card
    # Free models: llama-3.3-70b, llama-3.1-8b
    cerebras_api_key: str = ""
    cerebras_default_model: str = "llama-3.3-70b"

    # ── SambaNova Cloud (free, large models) ─────────────────────────────────
    # cloud.sambanova.ai — free, no credit card
    # Has Llama 3.1 405B free — largest freely available model anywhere
    # Free models: Meta-Llama-3.3-70B-Instruct, Meta-Llama-3.1-405B-Instruct
    sambanova_api_key: str = ""
    sambanova_default_model: str = "Meta-Llama-3.3-70B-Instruct"

    # ── Google Gemini (best free tier for Tier 2) ─────────────────────────────
    # aistudio.google.com — no billing required
    # gemini-1.5-flash: 15 RPM, 1,000,000 tokens/day FREE
    # gemini-2.0-flash-exp: experimental, very capable
    gemini_api_key: str = ""
    gemini_default_model: str = "gemini-1.5-flash"

    # ── Mistral (free experimental tier) ─────────────────────────────────────
    # console.mistral.ai — limited free tier
    mistral_api_key: str = ""
    mistral_default_model: str = "mistral-small-latest"

    # ── DeepSeek (near-free paid — great tier 3 option) ──────────────────────
    # platform.deepseek.com — V3: ~$0.07/1M input, $0.28/1M output tokens
    # Practically free for development. Strong coder + planner.
    deepseek_api_key: str = ""
    deepseek_default_model: str = "deepseek-chat"

    # ── Anthropic Claude (paid, tier 3) ──────────────────────────────────────
    # Only called when tier3_enabled=true or toggled per-project
    # claude-haiku-4-5-20251001: $0.25/$1.25 per 1M tokens (fast, cheap)
    anthropic_api_key: str = ""
    anthropic_default_model: str = "claude-haiku-4-5-20251001"
    anthropic_timeout_seconds: int = 120

    # ── Validation tuning ────────────────────────────────────────────────────
    # 1.0 = full thresholds (for 70B+ or Claude-class models)
    # 0.75 = for 13B-30B models
    # 0.60 = minimum viable for 7B local models
    validation_score_scale: float = 1.0

    # ── Orchestration limits ──────────────────────────────────────────────────
    # For local Ollama: 1-2. For fast cloud providers: 4-8.
    max_concurrent_tasks: int = 4
    task_default_retry_limit: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
