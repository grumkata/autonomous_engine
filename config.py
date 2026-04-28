from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "Autonomous AI Engine"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── Database / Vector store ───────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./engine.db"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_persist_dir: str = "./chroma_data"

    # ── LLM mode ─────────────────────────────────────────────────────────────
    # "single"  — one provider for everything (LLM_PROVIDER)
    # "tiered"  — route tasks to cheapest capable provider (recommended)
    llm_mode: str = "single"
    llm_provider: str = "ollama"  # used in single mode

    # ── Tier pools ────────────────────────────────────────────────────────────
    # Comma-separated. Providers with missing API keys are silently skipped.
    tier1_providers: str = "groq,cerebras,ollama"
    tier2_providers: str = "gemini,openrouter,sambanova"
    tier3_providers: str = "anthropic,deepseek"
    tier3_enabled: bool = False
    tier1_max_threshold: float = 0.72
    tier2_max_threshold: float = 0.84

    # ────────────────────────────────────────────────────────────────────────
    # PROVIDER KEYS + MODELS
    # Only add keys for providers you list in TIER*_PROVIDERS above.
    # ────────────────────────────────────────────────────────────────────────

    # ── Ollama (local, always free) ───────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.1:8b"
    ollama_timeout_seconds: int = 120
    ollama_api_key: str = ""

    # ── Groq — free, 14.4K req/day — console.groq.com ────────────────────────
    groq_api_key: str = ""
    groq_default_model: str = "llama-3.1-8b-instant"
    # Tier 2 upgrade: llama-3.3-70b-versatile

    # ── Cerebras — free, 1M tok/day, 2600 tok/s — cloud.cerebras.ai ──────────
    cerebras_api_key: str = ""
    cerebras_default_model: str = "llama-3.3-70b"

    # ── SiliconFlow — free, 1K RPM / 50K TPM — siliconflow.cn ────────────────
    siliconflow_api_key: str = ""
    siliconflow_default_model: str = "Qwen/Qwen3-8B"

    # ── LLM7.io — free, no key needed, 30-120 RPM — llm7.io ─────────────────
    llm7_default_model: str = "deepseek-ai/DeepSeek-R1"

    # ── Kluster AI — free, Qwen3-235B, Llama 4 — kluster.ai ─────────────────
    kluster_api_key: str = ""
    kluster_default_model: str = "klusterai/Meta-Llama-3.3-70B-Instruct-Turbo"

    # ── BazaarLink — free, multi-model router — bazaarlink.ai ────────────────
    bazaarlink_api_key: str = ""
    bazaarlink_default_model: str = "auto:free"

    # ── Pollinations.ai — free, no key needed — pollinations.ai ──────────────
    pollinations_default_model: str = "openai-large"

    # ── Ollama Cloud — free session limits — ollama.com ──────────────────────
    ollama_cloud_api_key: str = ""
    ollama_cloud_default_model: str = "deepseek-v3.2"

    # ── Featherless.ai — free, 4000+ models — featherless.ai ─────────────────
    featherless_api_key: str = ""
    featherless_default_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    # ── GitHub Models — free for GitHub users — github.com/marketplace/models ─
    github_models_api_key: str = ""
    github_models_default_model: str = "meta-llama-3.3-70b-instruct"

    # ── Hugging Face — free, 2K req/day — huggingface.co ─────────────────────
    huggingface_api_key: str = ""
    huggingface_default_model: str = "meta-llama/Llama-3.3-70B-Instruct"

    # ── Google Gemini — free, 10 RPM / 1M context — aistudio.google.com ──────
    gemini_api_key: str = ""
    gemini_default_model: str = "gemini-2.5-flash"

    # ── SambaNova — free, 10-30 RPM — cloud.sambanova.ai ─────────────────────
    sambanova_api_key: str = ""
    sambanova_default_model: str = "Meta-Llama-3.3-70B-Instruct"

    # ── OpenRouter — free :free models, 20 RPM — openrouter.ai ──────────────
    openrouter_api_key: str = ""
    openrouter_default_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    # ── Mistral — free Experiment plan, 1B tok/month — console.mistral.ai ────
    mistral_api_key: str = ""
    mistral_default_model: str = "mistral-small-latest"

    # ── Cohere — free Trial, 1K req/month, best RAG — cohere.com ─────────────
    cohere_api_key: str = ""
    cohere_default_model: str = "command-r-plus"

    # ── NVIDIA NIM — free 1K credits, 40 RPM — build.nvidia.com ─────────────
    nvidia_nim_api_key: str = ""
    nvidia_nim_default_model: str = "meta/llama-3.3-70b-instruct"

    # ── Zhipu AI — GLM-4-Flash free, NO rate cap — open.bigmodel.cn ──────────
    zhipu_api_key: str = ""
    zhipu_default_model: str = "glm-4-flash"

    # ── Moonshot (Kimi) — 1M context, long docs — platform.moonshot.cn ───────
    moonshot_api_key: str = ""
    moonshot_default_model: str = "moonshot-v1-128k"

    # ── Together AI — 200+ models, free :Free models — together.ai ───────────
    together_api_key: str = ""
    together_default_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

    # ── AI21 Labs — Jamba 256K context, 1B tok/month free — ai21.com ─────────
    ai21_api_key: str = ""
    ai21_default_model: str = "jamba-1.6-mini"

    # ── Anthropic Claude — paid, tier 3 — anthropic.com ──────────────────────
    anthropic_api_key: str = ""
    anthropic_default_model: str = "claude-haiku-4-5-20251001"
    anthropic_timeout_seconds: int = 120

    # ── DeepSeek — ~$0.07/1M tokens — platform.deepseek.com ─────────────────
    deepseek_api_key: str = ""
    deepseek_default_model: str = "deepseek-chat"

    # ── OpenAI — GPT-4.1, o3 — platform.openai.com ───────────────────────────
    openai_api_key: str = ""
    openai_default_model: str = "gpt-4.1-mini"

    # ── xAI Grok — 2M context, real-time web — console.x.ai ─────────────────
    xai_api_key: str = ""
    xai_default_model: str = "grok-3-fast"

    # ── Fireworks AI — FireAttention 4x faster — fireworks.ai ────────────────
    fireworks_api_key: str = ""
    fireworks_default_model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct"

    # ── Hyperbolic — 80% cheaper than AWS — hyperbolic.xyz ───────────────────
    hyperbolic_api_key: str = ""
    hyperbolic_default_model: str = "meta-llama/Llama-3.3-70B-Instruct"

    # ── DeepInfra — cheapest hosted open-source — deepinfra.com ──────────────
    deepinfra_api_key: str = ""
    deepinfra_default_model: str = "meta-llama/Llama-3.3-70B-Instruct"

    # ── Perplexity Sonar — LLM + web search — pplx.ai ────────────────────────
    perplexity_api_key: str = ""
    perplexity_default_model: str = "sonar"

    # ── AI/ML API — 300+ models unified — aimlapi.com ────────────────────────
    aimlapi_api_key: str = ""
    aimlapi_default_model: str = "gpt-4o"

    # ── 01.AI (Yi) — 200K context, bilingual — platform.lingyiwanwu.com ──────
    yi_api_key: str = ""
    yi_default_model: str = "yi-large"

    # ── Alibaba Qwen — Qwen3-235B, multilingual — dashscope.aliyuncs.com ─────
    qwen_api_key: str = ""
    qwen_default_model: str = "qwen-plus"

    # ── Novita AI — multi-modal budget — novita.ai ────────────────────────────
    novita_api_key: str = ""
    novita_default_model: str = "meta-llama/llama-3.3-70b-instruct"

    # ── Lambda Labs — H100 clusters — lambda.chat ────────────────────────────
    lambda_api_key: str = ""
    lambda_default_model: str = "llama3.3-70b-instruct-fp8"

    # ── LOCAL: self-hosted OpenAI-compat servers ──────────────────────────────
    # Add any of these to any tier (they are always free)
    lmstudio_base_url: str = "http://localhost:1234"
    jan_base_url: str = "http://localhost:1337"
    localai_base_url: str = "http://localhost:8080"
    llamacpp_base_url: str = "http://localhost:8080"
    vllm_base_url: str = "http://localhost:8000"
    sglang_base_url: str = "http://localhost:30000"

    # ── Tool provider keys ────────────────────────────────────────────────────
    # Web search
    tavily_api_key: str = ""      # tavily.com — $0.01/search, best structured results
    serper_api_key: str = ""      # serper.dev — $0.001/search, Google results
    # (DuckDuckGo is always available as a free fallback — no key needed)

    # Image generation
    stability_api_key: str = ""   # stability.ai — paid, high quality
    # (Pollinations.ai is always available free — no key needed)

    # Audio / TTS
    elevenlabs_api_key: str = ""  # elevenlabs.io — 10K chars/month free

    # ── Validation ────────────────────────────────────────────────────────────
    # 1.0 = full (for 70B+ or Claude). 0.75 = 13B-30B. 0.60 = 7B local.
    validation_score_scale: float = 1.0

    # ── Orchestration ─────────────────────────────────────────────────────────
    # Local Ollama: 1-2. Fast cloud APIs: 4-8.
    max_concurrent_tasks: int = 4
    task_default_retry_limit: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
