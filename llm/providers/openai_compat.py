"""
llm/providers/openai_compat.py — OpenAI-compatible provider.

Used for any backend that speaks the OpenAI /v1/chat/completions format.
Currently targets two FREE providers:

  Groq (https://console.groq.com)
    base_url : https://api.groq.com/openai/v1
    free tier: ~14,400 req/day on llama-3.1-8b-instant, mixtral-8x7b, etc.
    set GROQ_API_KEY in .env

  OpenRouter (https://openrouter.ai)
    base_url : https://openrouter.ai/api/v1
    free models: meta-llama/llama-3.2-3b-instruct:free, mistral 7b :free, etc.
    set OPENROUTER_API_KEY in .env

Both are instantiated via factory helpers at the bottom of this module.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from llm.providers.base import LLMProvider
from llm.schemas import (
    ChatResponse,
    LLMResult,
    Message,
    StreamChunk,
    UsageStats,
)

log = structlog.get_logger(__name__)

_RETRYABLE = {429, 500, 502, 503, 504}
_BASE_DELAY = 1.0
_MAX_DELAY = 30.0
_BACKOFF = 2.0


class OpenAICompatProvider(LLMProvider):
    """
    Provider for any backend that implements the OpenAI chat completions API.

    Converts between the engine's internal Message / LLMResult types and the
    OpenAI wire format.  Streaming yields StreamChunk objects identical to the
    Ollama provider so the rest of the engine is unaffected.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str,
        provider_label: str = "openai_compat",
        timeout_seconds: int = 120,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model_name = default_model
        self._provider_label = provider_label
        self._timeout = httpx.Timeout(
            connect=10.0,
            read=float(timeout_seconds),
            write=30.0,
            pool=5.0,
        )

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if extra_headers:
            headers.update(extra_headers)

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=self._timeout,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

        log.info(
            f"{provider_label}.provider.created",
            base_url=self._base_url,
            model=self._default_model_name,
        )

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    @property
    def default_model(self) -> str:
        return self._default_model_name

    @property
    def provider_name(self) -> str:
        return self._provider_label

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retry_limit: int = 3,
    ) -> LLMResult:
        t_start = time.monotonic()
        model = model or self._default_model_name

        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        data = await self._request("POST", "/chat/completions", body, retry_limit=retry_limit)
        latency_ms = int((time.monotonic() - t_start) * 1000)

        response = self._parse_response(data, model)
        log.debug(
            f"{self._provider_label}.chat.complete",
            model=model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
        )
        return LLMResult(
            response=response,
            total_latency_ms=latency_ms,
            model_used=model,
        )

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        model = model or self._default_model_name

        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        accumulated_tokens = 0
        async with self._http.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]  # strip "data: "
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content") or ""
                    finish = data["choices"][0].get("finish_reason")
                    done = finish is not None

                    usage_data = data.get("usage")
                    usage = None
                    if done and usage_data:
                        usage = UsageStats(
                            prompt_tokens=usage_data.get("prompt_tokens", 0),
                            completion_tokens=usage_data.get("completion_tokens", 0),
                            total_tokens=usage_data.get("total_tokens", 0),
                        )

                    yield StreamChunk(
                        model=data.get("model", model),
                        message=Message(role="assistant", content=content),
                        done=done,
                        usage=usage,
                    )
                except (json.JSONDecodeError, KeyError):
                    log.warning(f"{self._provider_label}.stream.unparseable", line=line[:100])

    async def health_check(self) -> bool:
        """
        Lightweight check — list available models (works for Groq/OpenRouter).
        Falls back to a HEAD request if models endpoint fails.
        """
        try:
            resp = await self._http.get("/models", timeout=5.0)
            return resp.status_code < 500
        except Exception:
            return False

    async def close(self) -> None:
        await self._http.aclose()
        log.info(f"{self._provider_label}.provider.closed")

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, data: dict[str, Any], model: str) -> ChatResponse:
        """Convert an OpenAI /chat/completions response to ChatResponse."""
        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        usage_raw = data.get("usage", {})

        return ChatResponse(
            model=data.get("model", model),
            message=Message(role="assistant", content=content),
            done=True,
            done_reason=choice.get("finish_reason", "stop") or "stop",
            usage=UsageStats(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            ),
            created_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Internal HTTP with retry
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        retry_limit: int = 3,
    ) -> dict[str, Any]:
        url = path if path.startswith("/") else f"/{path}"
        attempt = 0
        last_exc: Exception | None = None

        while attempt <= retry_limit:
            try:
                resp = await self._http.request(method, url, json=body)

                if resp.status_code in _RETRYABLE:
                    delay = min(_BASE_DELAY * (_BACKOFF ** attempt), _MAX_DELAY)
                    # Respect Retry-After header from rate-limit responses
                    retry_after = resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = min(float(retry_after), _MAX_DELAY)
                        except ValueError:
                            pass
                    log.warning(
                        f"{self._provider_label}.retryable_error",
                        status=resp.status_code,
                        attempt=attempt + 1,
                        retry_in_s=delay,
                    )
                    attempt += 1
                    if attempt > retry_limit:
                        resp.raise_for_status()
                    await asyncio.sleep(delay)
                    continue

                resp.raise_for_status()
                return resp.json()

            except httpx.TransportError as exc:
                last_exc = exc
                delay = min(_BASE_DELAY * (_BACKOFF ** attempt), _MAX_DELAY)
                log.warning(
                    f"{self._provider_label}.transport_error",
                    error=str(exc),
                    attempt=attempt + 1,
                    retry_in_s=delay,
                )
                attempt += 1
                if attempt > retry_limit:
                    raise
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"{self._provider_label} request failed after {retry_limit + 1} attempts: {last_exc}"
        )


# ---------------------------------------------------------------------------
# Convenience factories — used by llm/client.py

# =============================================================================
# Provider catalogue
# =============================================================================
# All entries use the same OpenAICompatProvider class.
# Format: make_<name>_provider(api_key, default_model) → OpenAICompatProvider
#
# Providers are grouped by tier:
#   TIER 1 — Free, no practical limits (general grunt work)
#   TIER 2 — Free, rate-limited (important tasks needing capability)
#   TIER 3 — Paid / near-free (heavy tasks, last resort)
#   LOCAL   — Self-hosted (OpenAI-compat servers on localhost)
# =============================================================================

# ---------------------------------------------------------------------------
# Generic factory — used internally by the catalogue
# ---------------------------------------------------------------------------

def _make(
    base_url: str,
    api_key: str,
    default_model: str,
    provider_label: str,
    timeout_seconds: int = 120,
    extra_headers: dict | None = None,
) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        base_url=base_url,
        api_key=api_key,
        default_model=default_model,
        provider_label=provider_label,
        timeout_seconds=timeout_seconds,
        extra_headers=extra_headers or {},
    )


# ── TIER 1: Free, no practical limits ─────────────────────────────────────


def make_groq_provider(api_key: str, default_model: str = "llama-3.1-8b-instant") -> OpenAICompatProvider:
    """Groq LPU — 300-800 tok/s, 14.4K req/day free. console.groq.com"""
    return _make("https://api.groq.com/openai/v1", api_key, default_model, "groq")


def make_cerebras_provider(api_key: str, default_model: str = "llama-3.3-70b") -> OpenAICompatProvider:
    """Cerebras wafer-scale — 1000-2600 tok/s, 1M tokens/day free. cloud.cerebras.ai"""
    return _make("https://api.cerebras.ai/v1", api_key, default_model, "cerebras", timeout_seconds=60)


def make_siliconflow_provider(api_key: str, default_model: str = "Qwen/Qwen3-8B") -> OpenAICompatProvider:
    """SiliconFlow — 1K RPM / 50K TPM free, highest free RPM. siliconflow.cn"""
    return _make("https://api.siliconflow.cn/v1", api_key, default_model, "siliconflow")


def make_llm7_provider(api_key: str = "", default_model: str = "deepseek-ai/DeepSeek-R1") -> OpenAICompatProvider:
    """LLM7.io — 30-120 RPM free, 27+ models, no credit card. llm7.io"""
    return _make("https://api.llm7.io/v1", api_key or "llm7-free", default_model, "llm7")


def make_kluster_provider(api_key: str, default_model: str = "klusterai/Meta-Llama-3.3-70B-Instruct-Turbo") -> OpenAICompatProvider:
    """Kluster AI — free Qwen3-235B, Llama 4, DeepSeek R1. kluster.ai"""
    return _make("https://api.kluster.ai/v1", api_key, default_model, "kluster")


def make_bazaarlink_provider(api_key: str, default_model: str = "auto:free") -> OpenAICompatProvider:
    """BazaarLink — free multi-model router (GPT-4o, Claude, Gemini). bazaarlink.ai"""
    return _make("https://api.bazaarlink.ai/v1", api_key, default_model, "bazaarlink")


def make_pollinations_provider(api_key: str = "", default_model: str = "openai-large") -> OpenAICompatProvider:
    """Pollinations.ai — no API key, no limits (fair use). gen.pollinations.ai"""
    # Uses POST /openai (not /v1/chat/completions) but compatible wrapper exists
    return _make("https://api.pollinations.ai", api_key or "noop", default_model, "pollinations")


def make_ollama_cloud_provider(api_key: str, default_model: str = "deepseek-v3.2") -> OpenAICompatProvider:
    """Ollama Cloud — same API as local Ollama, session-based free limits. ollama.com"""
    return _make("https://ollama.com/api", api_key, default_model, "ollama_cloud")


def make_featherless_provider(api_key: str, default_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct") -> OpenAICompatProvider:
    """Featherless.ai — 4000+ HuggingFace models serverless free. featherless.ai"""
    return _make("https://api.featherless.ai/v1", api_key, default_model, "featherless")


def make_github_models_provider(api_key: str, default_model: str = "meta-llama-3.3-70b-instruct") -> OpenAICompatProvider:
    """GitHub Models — free for all GitHub users, many top models. github.com/marketplace/models"""
    return _make("https://models.inference.ai.azure.com", api_key, default_model, "github_models")


def make_huggingface_provider(api_key: str, default_model: str = "meta-llama/Llama-3.3-70B-Instruct") -> OpenAICompatProvider:
    """Hugging Face Inference API — 150K+ models, 2K req/day free. huggingface.co"""
    return _make("https://api-inference.huggingface.co/v1", api_key, default_model, "huggingface")


# ── TIER 2: Free, rate-limited ─────────────────────────────────────────────


def make_gemini_provider(api_key: str, default_model: str = "gemini-2.5-flash") -> OpenAICompatProvider:
    """Google Gemini — 10 RPM / 250 RPD / 1M context free. aistudio.google.com"""
    return _make(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key, default_model, "gemini", timeout_seconds=90,
    )


def make_sambanova_provider(api_key: str, default_model: str = "Meta-Llama-3.3-70B-Instruct") -> OpenAICompatProvider:
    """SambaNova RDU — third fastest free inference, 10-30 RPM. cloud.sambanova.ai"""
    return _make("https://api.sambanova.ai/v1", api_key, default_model, "sambanova", timeout_seconds=90)


def make_openrouter_provider(api_key: str, default_model: str = "meta-llama/llama-3.3-70b-instruct:free") -> OpenAICompatProvider:
    """OpenRouter — 300+ models, 30+ free :free models, auto-fallback. openrouter.ai"""
    return _make(
        "https://openrouter.ai/api/v1", api_key, default_model, "openrouter",
        extra_headers={"HTTP-Referer": "https://github.com/autonomous_engine"},
    )


def make_mistral_provider(api_key: str, default_model: str = "mistral-small-latest") -> OpenAICompatProvider:
    """Mistral — Experiment plan: 1 req/s, 1B tokens/month free. console.mistral.ai"""
    return _make("https://api.mistral.ai/v1", api_key, default_model, "mistral")


def make_cohere_provider(api_key: str, default_model: str = "command-r-plus") -> OpenAICompatProvider:
    """Cohere — 1K req/month free Trial key, best RAG embeddings. cohere.com"""
    return _make("https://api.cohere.ai/compatibility/v1", api_key, default_model, "cohere")


def make_nvidia_nim_provider(api_key: str, default_model: str = "meta/llama-3.3-70b-instruct") -> OpenAICompatProvider:
    """NVIDIA NIM — 1K free credits, 40 RPM, H100-grade inference. build.nvidia.com"""
    return _make("https://integrate.api.nvidia.com/v1", api_key, default_model, "nvidia_nim")


def make_zhipu_provider(api_key: str, default_model: str = "glm-4-flash") -> OpenAICompatProvider:
    """Zhipu AI — GLM-4-Flash free with NO published rate cap. open.bigmodel.cn"""
    return _make("https://open.bigmodel.cn/api/paas/v4", api_key, default_model, "zhipu")


def make_moonshot_provider(api_key: str, default_model: str = "moonshot-v1-128k") -> OpenAICompatProvider:
    """Moonshot (Kimi) — up to 1M context, best long-doc processing. platform.moonshot.cn"""
    return _make("https://api.moonshot.cn/v1", api_key, default_model, "moonshot")


def make_together_provider(api_key: str, default_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free") -> OpenAICompatProvider:
    """Together AI — 200+ models; free :free models in catalogue. together.ai"""
    return _make("https://api.together.xyz/v1", api_key, default_model, "together")


def make_ai21_provider(api_key: str, default_model: str = "jamba-1.6-mini") -> OpenAICompatProvider:
    """AI21 Labs — Jamba hybrid SSM-Transformer, 256K context, 1B tok/month free. ai21.com"""
    return _make("https://api.ai21.com/studio/v1", api_key, default_model, "ai21")


# ── TIER 3: Paid / near-free ───────────────────────────────────────────────


def make_anthropic_openai_compat_provider(api_key: str, default_model: str = "claude-haiku-4-5-20251001") -> OpenAICompatProvider:
    """Anthropic via OpenAI-compat shim (use AnthropicProvider for full features)."""
    return _make("https://api.anthropic.com/v1", api_key, default_model, "anthropic_compat",
                 extra_headers={"anthropic-version": "2023-06-01"})


def make_deepseek_provider(api_key: str, default_model: str = "deepseek-chat") -> OpenAICompatProvider:
    """DeepSeek — ~$0.07/1M input tokens, near-free for dev. platform.deepseek.com"""
    return _make("https://api.deepseek.com/v1", api_key, default_model, "deepseek")


def make_openai_provider(api_key: str, default_model: str = "gpt-4.1-mini") -> OpenAICompatProvider:
    """OpenAI — GPT-4.1, o3, o4-mini. Most expensive but gold standard. platform.openai.com"""
    return _make("https://api.openai.com/v1", api_key, default_model, "openai")


def make_xai_provider(api_key: str, default_model: str = "grok-3-fast") -> OpenAICompatProvider:
    """xAI Grok — 2M context, real-time X/web search, 60 RPM. console.x.ai"""
    return _make("https://api.x.ai/v1", api_key, default_model, "xai")


def make_fireworks_provider(api_key: str, default_model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct") -> OpenAICompatProvider:
    """Fireworks AI — FireAttention 4x faster than vLLM, HIPAA/SOC2. fireworks.ai"""
    return _make("https://api.fireworks.ai/inference/v1", api_key, default_model, "fireworks")


def make_hyperbolic_provider(api_key: str, default_model: str = "meta-llama/Llama-3.3-70B-Instruct") -> OpenAICompatProvider:
    """Hyperbolic — up to 80% cheaper than AWS, Llama 4, DeepSeek. hyperbolic.xyz"""
    return _make("https://api.hyperbolic.xyz/v1", api_key, default_model, "hyperbolic")


def make_deepinfra_provider(api_key: str, default_model: str = "meta-llama/Llama-3.3-70B-Instruct") -> OpenAICompatProvider:
    """DeepInfra — cheapest hosted open-source ($0.07/M for 8B). deepinfra.com"""
    return _make("https://api.deepinfra.com/v1/openai", api_key, default_model, "deepinfra")


def make_perplexity_provider(api_key: str, default_model: str = "sonar") -> OpenAICompatProvider:
    """Perplexity Sonar — LLM + real-time web search + citations. pplx.ai"""
    return _make("https://api.perplexity.ai", api_key, default_model, "perplexity")


def make_aimlapi_provider(api_key: str, default_model: str = "gpt-4o") -> OpenAICompatProvider:
    """AI/ML API — 300+ models unified (text+image+audio+video). aimlapi.com"""
    return _make("https://api.aimlapi.com/v1", api_key, default_model, "aimlapi")


def make_yi_provider(api_key: str, default_model: str = "yi-large") -> OpenAICompatProvider:
    """01.AI (Yi) — 200K context, strong bilingual Chinese-English. platform.lingyiwanwu.com"""
    return _make("https://api.lingyiwanwu.com/v1", api_key, default_model, "yi")


def make_alibaba_qwen_provider(api_key: str, default_model: str = "qwen-plus") -> OpenAICompatProvider:
    """Alibaba Qwen — Qwen3-235B, multilingual, DashScope API. dashscope.aliyuncs.com"""
    return _make("https://dashscope.aliyuncs.com/compatible-mode/v1", api_key, default_model, "qwen")


def make_novita_provider(api_key: str, default_model: str = "meta-llama/llama-3.3-70b-instruct") -> OpenAICompatProvider:
    """Novita AI — 200+ models (LLM+image+video+speech), 50% cheaper. novita.ai"""
    return _make("https://api.novita.ai/v3/openai", api_key, default_model, "novita")


def make_lambda_provider(api_key: str, default_model: str = "llama3.3-70b-instruct-fp8") -> OpenAICompatProvider:
    """Lambda Labs — H100 clusters, transparent per-token pricing. lambda.chat"""
    return _make("https://api.lambdalabs.com/v1", api_key, default_model, "lambda")


# ── LOCAL: Self-hosted OpenAI-compat servers ───────────────────────────────


def make_lmstudio_provider(base_url: str = "http://localhost:1234", default_model: str = "local-model") -> OpenAICompatProvider:
    """LM Studio server — best desktop GUI for local LLMs. localhost:1234"""
    return _make(f"{base_url}/v1", "lm-studio", default_model, "lmstudio")


def make_jan_provider(base_url: str = "http://localhost:1337", default_model: str = "local-model") -> OpenAICompatProvider:
    """Jan AI — open-source LM Studio alternative. localhost:1337"""
    return _make(f"{base_url}/v1", "jan", default_model, "jan")


def make_localai_provider(base_url: str = "http://localhost:8080", default_model: str = "gpt-3.5-turbo") -> OpenAICompatProvider:
    """LocalAI — full OpenAI replacement (LLM+STT+TTS+images). localhost:8080"""
    return _make(f"{base_url}/v1", "localai", default_model, "localai")


def make_llamacpp_provider(base_url: str = "http://localhost:8080", default_model: str = "local-model") -> OpenAICompatProvider:
    """llama.cpp server — raw fastest inference, basis for Ollama/LM Studio. localhost:8080"""
    return _make(f"{base_url}/v1", "llama.cpp", default_model, "llamacpp")


def make_vllm_provider(base_url: str = "http://localhost:8000", default_model: str = "local-model") -> OpenAICompatProvider:
    """vLLM — production serving, best throughput for concurrent load. localhost:8000"""
    return _make(f"{base_url}/v1", "vllm", default_model, "vllm")


def make_sglang_provider(base_url: str = "http://localhost:30000", default_model: str = "local-model") -> OpenAICompatProvider:
    """SGLang — fastest structured output / agentic workflows. localhost:30000"""
    return _make(f"{base_url}/v1", "sglang", default_model, "sglang")
