"""
llm/router.py — Multi-model routing layer with MetaOrchestrator.

Architecture
------------

  ┌─────────────────────────────────────────────────────────────────┐
  │  agent_runner / planner                                         │
  │  get_client_for("red_team", "critique")                         │
  └────────────────────────┬────────────────────────────────────────┘
                           │  BoundClient.chat_json()
                           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  MetaOrchestrator  (Groq llama-3.1-8b-instant by default)       │
  │  • Sees: task context + all available/healthy slots              │
  │  • Returns: slot name in ~50 tokens (~100ms on Groq LPU)        │
  │  • Falls back to ScoreRouter on error or timeout                │
  └────────────────────────┬────────────────────────────────────────┘
                           │  chosen slot name
                           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  ModelRouter  (score-based fallback)                             │
  │  • Maintains TokenLedger (per-slot daily budget tracking)        │
  │  • Falls back to next candidate on any error                     │
  └────────────────────────┬────────────────────────────────────────┘
                           │  LLMClient (one per slot)
                           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Provider  (Groq | Cerebras | Google | OpenRouter | ...)        │
  └─────────────────────────────────────────────────────────────────┘

Configuration (in .env)
-----------------------
  # Choose routing strategy
  LLM_PROVIDER=pool
  LLM_POOL_CONFIG=[...JSON array of slot configs...]

  # Meta-orchestrator (the routing brain)
  META_ORCHESTRATOR_PROVIDER=groq
  META_ORCHESTRATOR_MODEL=llama-3.1-8b-instant
  META_ORCHESTRATOR_API_KEY=gsk_...   # can share with a groq slot
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from llm.schemas import LLMResult, Message

if TYPE_CHECKING:
    from llm.client import LLMClient

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Slot configuration
# ---------------------------------------------------------------------------


@dataclass
class ProviderSlotConfig:
    """
    One provider slot in the pool.

    Can be built manually via LLM_POOL_CONFIG JSON,
    or auto-generated from the catalog (see client.py build_slots_from_catalog).
    """
    name: str
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    timeout_seconds: int = 120
    daily_token_limit: int | None = None
    daily_request_limit: int | None = None
    departments: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    priority: int = 5
    speed_tier: str = "moderate"       # "instant" | "fast" | "moderate" | "slow"
    strengths: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProviderSlotConfig":
        return cls(
            name=d["name"],
            provider=d["provider"],
            model=d["model"],
            api_key=d.get("api_key", ""),
            base_url=d.get("base_url", ""),
            timeout_seconds=d.get("timeout_seconds", 120),
            daily_token_limit=d.get("daily_token_limit"),
            daily_request_limit=d.get("daily_request_limit"),
            departments=d.get("departments", []),
            task_types=d.get("task_types", []),
            priority=d.get("priority", 5),
            speed_tier=d.get("speed_tier", "moderate"),
            strengths=d.get("strengths", []),
        )

    @classmethod
    def parse_pool_config(cls, raw: str) -> list["ProviderSlotConfig"]:
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM_POOL_CONFIG is not valid JSON: {exc}") from exc
        if not isinstance(items, list):
            raise ValueError("LLM_POOL_CONFIG must be a JSON array.")
        return [cls.from_dict(item) for item in items]

    def to_summary(self) -> dict[str, Any]:
        """Compact dict for MetaOrchestrator prompts — keep it short."""
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "speed": self.speed_tier,
            "strengths": self.strengths,
            "departments": self.departments,
        }


# ---------------------------------------------------------------------------
# Token / request ledger
# ---------------------------------------------------------------------------


class TokenLedger:
    """
    In-memory per-slot daily counter for tokens and requests.
    Resets automatically at UTC midnight.
    """

    def __init__(self) -> None:
        self._day: dict[str, str] = {}
        self._tokens: dict[str, int] = {}
        self._requests: dict[str, int] = {}

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _ensure_today(self, slot: str) -> None:
        today = self._today()
        if self._day.get(slot) != today:
            self._day[slot] = today
            self._tokens[slot] = 0
            self._requests[slot] = 0

    def record(self, slot_name: str, tokens: int) -> None:
        self._ensure_today(slot_name)
        self._tokens[slot_name] += tokens
        self._requests[slot_name] += 1
        log.debug(
            "router.ledger.record",
            slot=slot_name,
            tokens=tokens,
            daily_tokens=self._tokens[slot_name],
            daily_requests=self._requests[slot_name],
        )

    def today_tokens(self, slot_name: str) -> int:
        self._ensure_today(slot_name)
        return self._tokens.get(slot_name, 0)

    def today_requests(self, slot_name: str) -> int:
        self._ensure_today(slot_name)
        return self._requests.get(slot_name, 0)

    def is_over_limit(self, slot_name: str, cfg: ProviderSlotConfig) -> bool:
        if cfg.daily_token_limit and self.today_tokens(slot_name) >= cfg.daily_token_limit:
            return True
        if cfg.daily_request_limit and self.today_requests(slot_name) >= cfg.daily_request_limit:
            return True
        return False

    def pressure(self, slot_name: str, cfg: ProviderSlotConfig) -> float:
        """
        Utilisation ratio 0.0–1.0. Used by MetaOrchestrator to prefer
        less-loaded slots. Returns 0.0 if no limits are set.
        """
        scores = []
        if cfg.daily_token_limit:
            scores.append(self.today_tokens(slot_name) / cfg.daily_token_limit)
        if cfg.daily_request_limit:
            scores.append(self.today_requests(slot_name) / cfg.daily_request_limit)
        return max(scores) if scores else 0.0

    def summary(self) -> list[dict[str, Any]]:
        today = self._today()
        return [
            {
                "slot": slot,
                "tokens_today": self._tokens.get(slot, 0),
                "requests_today": self._requests.get(slot, 0),
            }
            for slot, day in self._day.items()
            if day == today
        ]


# ---------------------------------------------------------------------------
# MetaOrchestrator — the routing brain
# ---------------------------------------------------------------------------


class MetaOrchestrator:
    """
    Uses a fast, free LLM (default: Groq llama-3.1-8b-instant) to pick the
    best provider slot for each task.

    Decision is ~50 tokens and completes in ~100ms on Groq LPU.
    Falls back to score-based routing on any error.

    The meta model receives:
      - department + task_type (the routing context)
      - available slots with their speed, strengths, and today's load pressure
      - instruction to return {"slot": "<name>"} and nothing else
    """

    _SYSTEM_PROMPT = (
        "You are a routing agent for an autonomous AI engine. "
        "Given a task context and a list of available provider slots, "
        "return ONLY a JSON object: {\"slot\": \"<slot_name>\"}. "
        "Pick the slot whose speed and strengths best match the task. "
        "Prefer low-pressure slots when capabilities are equal. "
        "Never explain. Never add text outside the JSON object."
    )

    def __init__(
        self,
        client: "LLMClient",
        model: str,
        timeout_ms: int = 3000,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout_ms = timeout_ms
        log.info(
            "meta_orchestrator.ready",
            model=model,
            timeout_ms=timeout_ms,
        )

    def _build_prompt(
        self,
        department: str,
        task_type: str,
        candidates: list[tuple[ProviderSlotConfig, float]],
    ) -> str:
        slots_desc = [
            {
                **cfg.to_summary(),
                "pressure": round(pressure, 2),
            }
            for cfg, pressure in candidates
        ]
        return (
            f"Task: department={department!r}, type={task_type!r}\n"
            f"Available slots:\n{json.dumps(slots_desc, indent=2)}\n"
            "Which slot name is best? Return {\"slot\": \"<name>\"}."
        )

    async def pick(
        self,
        department: str,
        task_type: str,
        candidates: list[tuple[ProviderSlotConfig, "LLMClient", float]],
    ) -> str | None:
        """
        Return the chosen slot name, or None if the decision fails
        (caller falls back to score-based routing).
        """
        if not candidates:
            return None

        # Build compact input — only name, speed, strengths, pressure
        candidate_info = [(cfg, pressure) for cfg, _, pressure in candidates]
        prompt = self._build_prompt(department, task_type, candidate_info)

        messages = [
            Message(role="system", content=self._SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        ]

        try:
            result = await asyncio.wait_for(
                self._client.chat(
                    messages,
                    model=self._model,
                    temperature=0.0,
                    max_tokens=30,
                    retry_limit=0,
                ),
                timeout=self._timeout_ms / 1000,
            )
            raw = result.content.strip()
            # Strip markdown fences if model ignores instruction
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            data = json.loads(raw)
            chosen = data.get("slot", "").strip()
            if chosen and any(cfg.name == chosen for cfg, _, _ in candidates):
                log.info(
                    "meta_orchestrator.decision",
                    slot=chosen,
                    department=department,
                    task_type=task_type,
                )
                return chosen
            log.warning("meta_orchestrator.invalid_slot", returned=chosen)
            return None
        except asyncio.TimeoutError:
            log.warning("meta_orchestrator.timeout", timeout_ms=self._timeout_ms)
            return None
        except Exception as exc:
            log.warning("meta_orchestrator.error", error=str(exc))
            return None


# ---------------------------------------------------------------------------
# Score-based fallback router
# ---------------------------------------------------------------------------

def _score_slot(
    cfg: ProviderSlotConfig,
    department: str | None,
    task_type: str | None,
) -> int:
    score = 0
    if department and cfg.departments and department in cfg.departments:
        score += 10
    if task_type and cfg.task_types and task_type in cfg.task_types:
        score += 5
    if department and cfg.strengths and department in cfg.strengths:
        score += 3
    if task_type and cfg.strengths and task_type in cfg.strengths:
        score += 3
    return score


# ---------------------------------------------------------------------------
# ModelRouter — dispatches calls with MetaOrchestrator + score fallback
# ---------------------------------------------------------------------------


class ModelRouter:
    """
    Routes LLM calls to the best available provider slot.

    Routing order:
      1. MetaOrchestrator picks a slot (fast LLM decision, ~100ms)
      2. If meta fails, fall back to score-based ranking
      3. If chosen slot errors, fall through to next candidate
      4. If all slots are exhausted, raise RuntimeError
    """

    def __init__(
        self,
        slots: list[tuple[ProviderSlotConfig, "LLMClient"]],
        ledger: TokenLedger,
        meta: MetaOrchestrator | None = None,
    ) -> None:
        self._slots = sorted(slots, key=lambda s: s[0].priority)
        self._ledger = ledger
        self._meta = meta

        log.info(
            "router.ready",
            slot_count=len(self._slots),
            meta_enabled=meta is not None,
            slots=[
                {
                    "name": cfg.name,
                    "provider": cfg.provider,
                    "model": cfg.model,
                    "speed": cfg.speed_tier,
                    "priority": cfg.priority,
                }
                for cfg, _ in self._slots
            ],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _available_candidates(
        self,
        department: str | None,
        task_type: str | None,
    ) -> list[tuple[ProviderSlotConfig, "LLMClient", float]]:
        """
        Return (config, client, pressure) for slots that are not over budget,
        ordered by routing score then priority.
        """
        result: list[tuple[int, int, ProviderSlotConfig, LLMClient, float]] = []

        for cfg, client in self._slots:
            if self._ledger.is_over_limit(cfg.name, cfg):
                log.warning(
                    "router.slot.over_budget",
                    slot=cfg.name,
                    tokens=self._ledger.today_tokens(cfg.name),
                    requests=self._ledger.today_requests(cfg.name),
                )
                continue
            score = _score_slot(cfg, department, task_type)
            pressure = self._ledger.pressure(cfg.name, cfg)
            result.append((score, cfg.priority, cfg, client, pressure))

        result.sort(key=lambda x: (-x[0], x[1], x[4]))  # score desc, priority asc, pressure asc
        return [(cfg, client, pressure) for _, _, cfg, client, pressure in result]

    def _find_client(self, slot_name: str) -> "LLMClient | None":
        for cfg, client in self._slots:
            if cfg.name == slot_name:
                return client
        return None

    def _find_cfg(self, slot_name: str) -> "ProviderSlotConfig | None":
        for cfg, _ in self._slots:
            if cfg.name == slot_name:
                return cfg
        return None

    # ------------------------------------------------------------------
    # Public chat_json (main entry point)
    # ------------------------------------------------------------------

    async def chat_json(
        self,
        messages: list[Message],
        *,
        department: str | None = None,
        task_type: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        retry_limit: int = 3,
    ) -> tuple[dict[str, Any], LLMResult]:
        candidates = self._available_candidates(department, task_type)
        if not candidates:
            raise RuntimeError("ModelRouter: all slots are over budget or unavailable.")

        # Ask MetaOrchestrator for a recommendation
        ordered = list(candidates)
        if self._meta is not None and len(candidates) > 1:
            chosen_name = await self._meta.pick(
                department or "", task_type or "", candidates
            )
            if chosen_name:
                # Move chosen slot to front
                ordered = (
                    [(cfg, cl, pr) for cfg, cl, pr in candidates if cfg.name == chosen_name]
                    + [(cfg, cl, pr) for cfg, cl, pr in candidates if cfg.name != chosen_name]
                )

        last_exc: Exception | None = None
        for cfg, client, _ in ordered:
            try:
                log.info(
                    "router.dispatch",
                    slot=cfg.name,
                    provider=cfg.provider,
                    model=cfg.model,
                    department=department,
                    task_type=task_type,
                )
                parsed, llm_result = await client.chat_json(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retry_limit=retry_limit,
                )
                self._ledger.record(cfg.name, llm_result.usage.total_tokens)
                return parsed, llm_result

            except Exception as exc:
                last_exc = exc
                remaining = len(ordered) - ordered.index((cfg, client, _)) - 1
                log.warning(
                    "router.slot.failed",
                    slot=cfg.name,
                    error=str(exc),
                    remaining=remaining,
                )

        raise RuntimeError(
            f"ModelRouter: all {len(ordered)} slot(s) failed. Last: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Public chat (plain, non-JSON)
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        *,
        department: str | None = None,
        task_type: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retry_limit: int = 3,
    ) -> LLMResult:
        candidates = self._available_candidates(department, task_type)
        if not candidates:
            raise RuntimeError("ModelRouter: all slots are over budget or unavailable.")

        ordered = list(candidates)
        if self._meta is not None and len(candidates) > 1:
            chosen_name = await self._meta.pick(
                department or "", task_type or "", candidates
            )
            if chosen_name:
                ordered = (
                    [(cfg, cl, pr) for cfg, cl, pr in candidates if cfg.name == chosen_name]
                    + [(cfg, cl, pr) for cfg, cl, pr in candidates if cfg.name != chosen_name]
                )

        last_exc: Exception | None = None
        for cfg, client, _ in ordered:
            try:
                result = await client.chat(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retry_limit=retry_limit,
                )
                self._ledger.record(cfg.name, result.usage.total_tokens)
                return result
            except Exception as exc:
                last_exc = exc
                log.warning("router.slot.failed", slot=cfg.name, error=str(exc))

        raise RuntimeError(
            f"ModelRouter: all {len(ordered)} slot(s) failed. Last: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def ledger(self) -> TokenLedger:
        return self._ledger

    def token_summary(self) -> list[dict[str, Any]]:
        return self._ledger.summary()

    async def health_check_all(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for cfg, client in self._slots:
            try:
                results[cfg.name] = await client.ping()
            except Exception:
                results[cfg.name] = False
        return results


# ---------------------------------------------------------------------------
# BoundClient — pre-filled routing context
# ---------------------------------------------------------------------------


class BoundClient:
    """
    A thin wrapper returned by get_client_for() that carries department +
    task_type so every chat_json() call is automatically routed correctly.

    Works transparently with both ModelRouter and a plain LLMClient.
    """

    def __init__(
        self,
        backend: "ModelRouter | LLMClient",
        department: str,
        task_type: str,
    ) -> None:
        self._backend = backend
        self._department = department
        self._task_type = task_type

    async def chat_json(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], LLMResult]:
        if isinstance(self._backend, ModelRouter):
            return await self._backend.chat_json(
                messages,
                department=self._department,
                task_type=self._task_type,
                **kwargs,
            )
        return await self._backend.chat_json(messages, **kwargs)

    async def chat(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> LLMResult:
        if isinstance(self._backend, ModelRouter):
            return await self._backend.chat(
                messages,
                department=self._department,
                task_type=self._task_type,
                **kwargs,
            )
        return await self._backend.chat(messages, **kwargs)
