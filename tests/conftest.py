import json
from pathlib import Path

import pytest

from finbench.data import load_examples
from finbench.model import AssistantTurn, ToolCall

FIXTURE = Path(__file__).parent / "fixtures" / "finqa_sample.json"


@pytest.fixture
def examples():
    """Six real FinQA test items (MIT licensed, github.com/czyssrs/FinQA):
    1-step, 2-step, 5-step with constants, table_average, greater, percent literal."""
    return {e.id: e for e in load_examples(FIXTURE)}


class ScriptedModel:
    """Fake ChatModel: returns pre-written turns in order, records what it saw."""

    name = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def complete(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        if not self.turns:
            raise RuntimeError("script exhausted")
        return self.turns.pop(0)


def call(name, args, i=0):
    raw = args if isinstance(args, str) else json.dumps(args)
    return AssistantTurn(content="", tool_calls=[ToolCall(id=f"c{i}", name=name, arguments=raw)],
                         prompt_tokens=100, completion_tokens=10, latency_s=0.5)


def text(content):
    return AssistantTurn(content=content, prompt_tokens=100, completion_tokens=20, latency_s=0.4)


class GoldModel:
    """Fake ChatModel that replays the gold tool conversation for whichever
    example it is asked about. A perfect model; any harness bug shows up as
    accuracy below 100%."""

    name = "gold-oracle"

    def __init__(self, examples):
        from finbench.fewshot import gold_conversation
        self.scripts = {e.id: [m for m in gold_conversation(e) if m["role"] == "assistant"]
                        for e in examples}
        self.by_question = {}
        from finbench.data import render_context
        for e in examples:
            self.by_question[render_context(e)] = e.id

    def complete(self, messages, tools):
        user = [m for m in messages if m["role"] == "user"][-1]["content"]
        ex_id = self.by_question[user]
        done = sum(1 for m in messages[messages.index([m for m in messages if m["role"] == "user"][-1]):]
                   if m["role"] == "assistant")
        m = self.scripts[ex_id][done]
        c = m["tool_calls"][0]
        return AssistantTurn(content="", tool_calls=[ToolCall(c["id"], c["function"]["name"],
                                                              c["function"]["arguments"])],
                             prompt_tokens=50, completion_tokens=5, latency_s=0.1)
