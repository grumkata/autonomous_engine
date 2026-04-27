"""
llm/providers/anthropic_provider.py — Anthropic Claude provider.

Uses the Anthropic Messages API directly via httpx (no SDK dependency).
Supports claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001, and any
other model in the claude-* family.

Activate via .env:
    LLM_PROVIDER=anthropic
    ANTHROPIC_API_KEY=sk-ant-...
    ANTHROPIC_DEFAULT_MODEL=claude-haiku-4-5-20251001   # fast + cheap
    # or for best quality:
    ANTHROPIC_DEFAULT_MODEL=claude-sonnet-4-6

Why this matters for autonomous_engine
---------------------------------------
The 8-dimension validator (validator.py) requires responses with:
  - Rich findings (4+ items for full completeness score)
  - Substantive summaries (>200 chars of non-vague content)
  - Specific, mechanistic risks
  - High coherence and usefulness scores

llama3.1:8b routinely scores 0.45–0.62 on these dimensions, well below
the 0.75–0.82 thresholds.  Claude Haiku typically scores 0.78–0.90 on
the same tasks, and Sonnet/Opus score 0.85–0.95.

The planner (planner.py) asks for a full DAG in one JSON shot.  Claude
reliably produces valid task graphs on the first attempt; local 8B models
require 2–3 retries and often still fail the schema check.
"""

from __future__ import annotations

import asyncio
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

_API_BASE = "https://api.anthropic.com"
_API_VERSION = "2023-06-01"
_RETRYABLE = {429, 500, 502, 503, 504}
_BASE_DELAY = 1.0
_MAX_DELAY = 30.0
_BACKOFF = 2.0


class AnthropicProvider(LLMProvider):
    """
    Provider for Anthropic Claude models.

    Key difference from OpenAI-compat: the system prompt is a top-level
    field in the request body, not a message with role="system".
    This class extracts it automatically from the message list so the
    rest of the engine doesn't need to change.
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-haiku-4-5-20251001",
        timeout_seconds: int = 120,
    ) -> None:
        self._default_model_name = default_model
        self._timeout = httpx.Timeout(
            connect=10.0,
            read=float(timeout_seconds),
            write=30.0,
            pool=5.0,
        )
        self._http = httpx.AsyncClient(
            base_url=_API_BASE,
            headers={
                "x-api-key": api_key,
                "anthropic-version": _API_VERSION,
                "content-type": "application/json",
            },
            timeout=self._timeout,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        log.info(
            "anthropic.provider.created",
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
        return "anthropic"

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

        system_prompt, user_messages = self._split_system(messages)

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or 4096,
            "messages": [{"role": m.role, "content": m.content} for m in user_messages],
        }
        if system_prompt:
            body["system"] = system_prompt
        if temperature is not None:
            body["temperature"] = temperature

        data = await self._request("POST", "/v1/messages", body, retry_limit=retry_limit)
        latency_ms = int((time.monotonic() - t_start) * 1000)

        response = self._parse_response(data, model)
        log.debug(
            "anthropic.chat.complete",
            model=model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
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
        """
        Streaming via Anthropic's SSE format.
        Yields StreamChunk objects compatible with the rest of the engine.
        """
        model = model or self._default_model_name
        system_prompt, user_messages = self._split_system(messages)

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or 4096,
            "messages": [{"role": m.role, "content": m.content} for m in user_messages],
            "stream": True,
        }
        if system_prompt:
            body["system"] = system_prompt
        if temperature is not None:
            body["temperature"] = temperature

        accumulated_content = ""
        input_tokens = 0
        output_tokens = 0

        async with self._http.stream("POST", "/v1/messages", json=body) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    import json as _json
                    data = _json.loads(payload)
                    event_type = data.get("type", "")

                    if event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        chunk_text = delta.get("text", "")
                        accumulated_content += chunk_text
                        yield StreamChunk(
                            model=model,
                            message=Message(role="assistant", content=chunk_text),
                            done=False,
                        )
                    elif event_type == "message_delta":
                        usage = data.get("usage", {})
                        output_tokens = usage.get("output_tokens", output_tokens)
                    elif event_type == "message_start":
                        msg = data.get("message", {})
                        usage = msg.get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)
                    elif event_type == "message_stop":
                        yield StreamChunk(
                            model=model,
                            message=Message(role="assistant", content=""),
                            done=True,
                            usage=UsageStats(
                                prompt_tokens=input_tokens,
                                completion_tokens=output_tokens,
                                total_tokens=input_tokens + output_tokens,
                            ),
                        )
                except Exception:
                    log.warning("anthropic.stream.unparseable", line=line[:100])

    async def health_check(self) -> bool:
        """Quick ping — list models endpoint."""
        try:
            resp = await self._http.get("/v1/models", timeout=5.0)
            return resp.status_code < 500
        except Exception:
            return False

    async def close(self) -> None:
        await self._http.aclose()
        log.info("anthropic.provider.closed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
        """
        Anthropic requires system prompt as a top-level field, not a message.
        Extract leading system messages and return (system_str, remaining_messages).
        """
        system_parts: list[str] = []
        rest: list[Message] = []
        for m in messages:
            if m.role == "system" and not rest:
                system_parts.append(m.content)
            else:
                rest.append(m)
        return "\n\n".join(system_parts), rest

    def _parse_response(self, data: dict[str, Any], model: str) -> ChatResponse:
        """Convert Anthropic Messages API response to ChatResponse."""
        # content is a list of blocks; concatenate all text blocks
        content_blocks = data.get("content", [])
        text = "".join(
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        )

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        stop_reason = data.get("stop_reason", "end_turn") or "end_turn"
        # Normalise stop_reason to OpenAI convention for compatibility
        if stop_reason == "end_turn":
            stop_reason = "stop"
        elif stop_reason == "max_tokens":
            stop_reason = "length"

        return ChatResponse(
            model=data.get("model", model),
            message=Message(role="assistant", content=text),
            done=True,
            done_reason=stop_reason,
            usage=UsageStats(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            created_at=datetime.now(timezone.utc),
        )

    async def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        retry_limit: int = 3,
    ) -> dict[str, Any]:
        attempt = 0
        last_exc: Exception | None = None

        while attempt <= retry_limit:
            try:
                resp = await self._http.request(method, path, json=body)

                if resp.status_code in _RETRYABLE:
                    delay = min(_BASE_DELAY * (_BACKOFF ** attempt), _MAX_DELAY)
                    retry_after = resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = min(float(retry_after), _MAX_DELAY)
                        except ValueError:
                            pass
                    log.warning(
                        "anthropic.retryable_error",
                        status=resp.status_code,
                        attempt=attempt + 1,
                        retry_in_s=delay,
                    )
                    attempt += 1
                    if attempt > retry_limit:
                        resp.raise_for_status()
                    await asyncio.sleep(delay)
                    continue

                if not resp.is_success:
                    # Surface the Anthropic error message clearly
                    try:
                        err = resp.json()
                        msg = err.get("error", {}).get("message", resp.text)
                    except Exception:
                        msg = resp.text
                    log.error(
                        "anthropic.api_error",
                        status=resp.status_code,
                        message=msg[:300],
                    )
                    resp.raise_for_status()

                return resp.json()

            except httpx.TransportError as exc:
                last_exc = exc
                delay = min(_BASE_DELAY * (_BACKOFF ** attempt), _MAX_DELAY)
                log.warning(
                    "anthropic.transport_error",
                    error=str(exc),
                    attempt=attempt + 1,
                    retry_in_s=delay,
                )
                attempt += 1
                if attempt > retry_limit:
                    raise
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Anthropic request failed after {retry_limit + 1} attempts: {last_exc}"
        )
