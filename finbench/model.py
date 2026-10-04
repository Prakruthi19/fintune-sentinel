"""Model port and the one adapter we need.

Every serving option in this project (Ollama, llama.cpp server, vLLM,
Groq, OpenRouter, Gemini's OpenAI endpoint) speaks the OpenAI chat
completions API, so one adapter covers local, GPU and API baselines. The
eval code only depends on the ChatModel protocol, which keeps it testable
with a scripted fake and no network.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model


@dataclass
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn: ...


class OpenAICompatibleModel:
    def __init__(self, model: str, base_url: str, api_key: str = "not-needed",
                 temperature: float = 0.0, max_tokens: int = 1024, timeout_s: float = 300):
        from openai import OpenAI  # imported lazily so tests don't need the package configured

        self.name = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=messages,
            tools=tools,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        latency = time.perf_counter() - start
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=c.id or f"call_{i}", name=c.function.name, arguments=c.function.arguments or "")
            for i, c in enumerate(msg.tool_calls or [])
        ]
        usage = resp.usage
        return AssistantTurn(
            content=msg.content,
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
        )
