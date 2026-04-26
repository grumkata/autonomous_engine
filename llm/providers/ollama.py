"""
llm/providers/ollama.py — Ollama LLM provider.

Wraps the Ollama REST API (/api/chat).  Supports local installs and
hosted Ollama proxies (ollamafreeapi.com etc.) via Bearer token auth.

Free to run: requires a local `ollama serve` or a compatible hosted proxy.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import structlog

from llm.providers.base import LLMProvider
from llm.schemas import (
    AvailableModels,
    ChatRequest,
    ChatResponse,
    LLMResult,
    Message,
    ModelInfo,
    OllamaOptions,
    StreamChunk,
    UsageStats,
)

log = structlog.get_logger(__name__)

_RETRYABLE = {429, 500, 502, 503, 504}
_BASE_DELAY = 1.0
_MAX_DELAY = 30.0
_BACKOFF = 2.0


class OllamaProvider(LLMProvider):
    """
    Async provider for a local (or proxied) Ollama instance.

    The underlying httpx.AsyncClient manages a connection pool — do not
    create multiple instances of this class.
    """

    def __init__(
        self,
        base_url: str,
        default_model: str,
        api_key: str = "",
        timeout_seconds: int = 120,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model_name = default_model
        self._timeout = httpx.Timeout(
            connect=10.0,
            read=float(timeout_seconds),
            write=30.0,
            pool=5.0,
        )

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=self._timeout,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

        log.info(
            "ollama.provider.created",
            base_url=self._base_url,
            model=self._default_model_name,
            auth="bearer" if api_key else "none",
        )

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    @property
    def default_model(self) -> str:
        return self._default_model_name

    @property
    def provider_name(self) -> str:
        return "ollama"

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
        options = OllamaOptions(temperature=temperature, num_predict=max_tokens)
        request = ChatRequest(model=model, messages=messages, options=options)

        data = await self._request(
            "POST", "/api/chat", request.to_ollama_dict(), retry_limit=retry_limit
        )
        latency_ms = int((time.monotonic() - t_start) * 1000)
        response = ChatResponse.from_ollama_dict(data)

        log.debug(
            "ollama.chat.complete",
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
        options = OllamaOptions(temperature=temperature, num_predict=max_tokens)
        request = ChatRequest(model=model, messages=messages, stream=True, options=options)

        async with self._http.stream("POST", "/api/chat", json=request.to_ollama_dict()) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield StreamChunk.from_ollama_dict(data)
                except json.JSONDecodeError:
                    log.warning("ollama.stream.unparseable", line=line[:100])

    async def health_check(self) -> bool:
        try:
            resp = await self._http.get("/", timeout=5.0)
            return resp.status_code < 500
        except Exception:
            return False

    async def close(self) -> None:
        await self._http.aclose()
        log.info("ollama.provider.closed")

    # ------------------------------------------------------------------
    # Ollama-specific helpers (used by startup check + model management)
    # ------------------------------------------------------------------

    async def startup_check(self) -> None:
        """
        Verify Ollama is reachable and the default model is available.
        Logs warnings but never raises — the engine starts even if Ollama
        is temporarily offline (tasks will fail gracefully).
        """
        try:
            models = await self.list_models()
            if models.has_model(self._default_model_name):
                log.info(
                    "ollama.startup.ok",
                    model=self._default_model_name,
                    available=[m.name for m in models.models],
                )
            else:
                log.warning(
                    "ollama.startup.model_missing",
                    default_model=self._default_model_name,
                    available=[m.name for m in models.models],
                    hint=f"ollama pull {self._default_model_name}",
                )
        except Exception as exc:
            log.warning(
                "ollama.startup.unreachable",
                error=str(exc),
                base_url=self._base_url,
            )

    async def list_models(self) -> AvailableModels:
        try:
            data = await self._request("GET", "/api/tags", retry_limit=2)
            return AvailableModels(models=[ModelInfo(**m) for m in data.get("models", [])])
        except Exception as exc:
            log.warning("ollama.list_models.failed", error=str(exc))
            return AvailableModels(models=[])

    async def pull_model(self, model_name: str) -> None:
        log.info("ollama.pull_model.start", model=model_name)
        await self._request(
            "POST", "/api/pull", {"name": model_name, "stream": False}, retry_limit=0
        )
        log.info("ollama.pull_model.complete", model=model_name)

    # ------------------------------------------------------------------
    # Internal HTTP request with retry
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
                    log.warning(
                        "ollama.retryable_error",
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
                    "ollama.transport_error",
                    error=str(exc),
                    attempt=attempt + 1,
                    retry_in_s=delay,
                )
                attempt += 1
                if attempt > retry_limit:
                    raise
                await asyncio.sleep(delay)

        raise RuntimeError(f"Ollama request failed after {retry_limit + 1} attempts: {last_exc}")
