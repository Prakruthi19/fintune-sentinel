"""The tool-use loop: ask, execute tool calls, feed results back, repeat
until final_answer or the step budget runs out."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from finbench.data import Example, render_context
from finbench.model import AssistantTurn, ChatModel
from finbench.tools import TOOLS, ToolCallRecord, ToolRuntime

SYSTEM_PROMPT = (
    "You answer numerical questions about company financial filings. "
    "Find the numbers you need in the filing text or table, then use the "
    "calculate or table_aggregate tools for ALL arithmetic, one operation per call. "
    "When done, call final_answer with the raw numeric result. "
    "Do not do arithmetic yourself."
)


@dataclass
class Trajectory:
    example_id: str
    final_answer: object = None
    stop_reason: str = ""  # final_answer | text_only | max_turns | model_error
    turns: list[AssistantTurn] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def latency_s(self) -> float:
        return sum(t.latency_s for t in self.turns)

    @property
    def prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.turns)

    @property
    def completion_tokens(self) -> int:
        return sum(t.completion_tokens for t in self.turns)

    @property
    def invalid_calls(self) -> int:
        return sum(1 for c in self.tool_calls if c.error)


def base_messages(example: Example, fewshot: list[dict] | None = None) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, *(fewshot or []),
            {"role": "user", "content": render_context(example)}]


def solve(model: ChatModel, example: Example, fewshot: list[dict] | None = None,
          max_turns: int = 8) -> Trajectory:
    runtime = ToolRuntime(example)
    messages = base_messages(example, fewshot)
    traj = Trajectory(example_id=example.id, messages=messages)
    for _ in range(max_turns):
        try:
            turn = model.complete(messages, TOOLS)
        except Exception as e:  # network, server, malformed response
            traj.stop_reason, traj.error = "model_error", f"{type(e).__name__}: {e}"
            break
        traj.turns.append(turn)
        if not turn.tool_calls:
            messages.append({"role": "assistant", "content": turn.content or ""})
            traj.stop_reason = "text_only"
            break
        messages.append({
            "role": "assistant",
            "content": turn.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments}}
                           for c in turn.tool_calls],
        })
        for call in turn.tool_calls:
            record = runtime.execute(call.name, call.arguments)
            payload = {"error": record.error} if record.error else {"result": record.result}
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(payload)})
            if call.name == "final_answer" and not record.error:
                traj.final_answer = record.result
                traj.stop_reason = "final_answer"
        if traj.stop_reason == "final_answer":
            break
    else:
        traj.stop_reason = "max_turns"
    traj.tool_calls = runtime.calls
    return traj
