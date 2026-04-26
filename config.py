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
    # Dev: "sqlite+aiosqlite:///./engine.db"
    # Prod: "postgresql+asyncpg://user:pass@host/db"
    database_url: str = "sqlite+aiosqlite:///./engine.db"

    # Vector store
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_persist_dir: str = "./chroma_data"

    # ── LLM provider selection ────────────────────────────────────────────────
    # Which backend to use. Options: "ollama" | "groq" | "openrouter"
    # Falls back to "ollama" if the chosen provider's API key is missing.
    llm_provider: str = "ollama"

    # ── Ollama ────────────────────────────────────────────────────────────────
    # Local:  http://localhost:11434  (default — completely free)
    # Hosted: https://api.ollamafreeapi.com (or any compatible proxy)
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.1:8b"
    ollama_timeout_seconds: int = 120
    # API key for hosted Ollama proxies — leave blank for local installs
    ollama_api_key: str = ""

    # ── Groq (free tier) ──────────────────────────────────────────────────────
    # Sign up at https://console.groq.com — no credit card required.
    # Free limits: ~14,400 requests/day on fast models.
    # Activate: set LLM_PROVIDER=groq and GROQ_API_KEY in .env
    groq_api_key: str = ""
    groq_default_model: str = "llama-3.1-8b-instant"

    # ── OpenRouter (free models) ──────────────────────────────────────────────
    # Sign up at https://openrouter.ai — free models available (append :free).
    # Free models: meta-llama/llama-3.2-3b-instruct:free, mistral-7b:free, etc.
    # Activate: set LLM_PROVIDER=openrouter and OPENROUTER_API_KEY in .env
    openrouter_api_key: str = ""
    openrouter_default_model: str = "meta-llama/llama-3.2-3b-instruct:free"

    # ── Orchestration limits ──────────────────────────────────────────────────
    max_concurrent_tasks: int = 8
    task_default_retry_limit: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
