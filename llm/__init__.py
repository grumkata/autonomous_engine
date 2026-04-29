"""
LLM package — Ollama client, structured schemas, and prompt building.

Public surface:
    from llm import AsyncOllamaClient, get_client, PromptBuilder
    from llm.schemas import Message, ChatResponse, AgentInputBundle, AgentOutput
"""

from llm.client import LLMClient, get_client, close_client
from llm.prompts import PromptBuilder
from llm.schemas import (
    Message,
    ChatRequest,
    ChatResponse,
    StreamChunk,
    UsageStats,
    AgentInputBundle,
    AgentOutput,
)

__all__ = [
    "get_client",
    "close_client",
    "PromptBuilder",
    "Message",
    "ChatRequest",
    "ChatResponse",
    "StreamChunk",
    "UsageStats",
    "AgentInputBundle",
    "AgentOutput",
]
