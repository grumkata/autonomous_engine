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
# ---------------------------------------------------------------------------


def make_groq_provider(
    api_key: str,
    default_model: str = "llama-3.1-8b-instant",
) -> OpenAICompatProvider:
    """
    Factory for the Groq free-tier provider.

    Free models include:
        llama-3.1-8b-instant   (fast, generous limits)
        llama-3.3-70b-versatile (larger, slower)
        mixtral-8x7b-32768
        gemma2-9b-it

    Sign up at https://console.groq.com — no credit card required.
    """
    return OpenAICompatProvider(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        default_model=default_model,
        provider_label="groq",
    )


def make_openrouter_provider(
    api_key: str,
    default_model: str = "meta-llama/llama-3.2-3b-instruct:free",
    app_name: str = "Autonomous AI Engine",
) -> OpenAICompatProvider:
    """
    Factory for the OpenRouter provider.

    Free models (append :free) include:
        meta-llama/llama-3.2-3b-instruct:free
        meta-llama/llama-3.2-1b-instruct:free
        mistralai/mistral-7b-instruct:free
        google/gemma-2-9b-it:free

    Sign up at https://openrouter.ai — free tier, no credit card required.
    The HTTP-Referer and X-Title headers are recommended by OpenRouter for
    attribution but are not required.
    """
    return OpenAICompatProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_model=default_model,
        provider_label="openrouter",
        extra_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": app_name,
        },
    )
