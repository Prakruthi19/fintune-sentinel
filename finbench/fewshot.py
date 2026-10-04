"""Turn gold FinQA programs into worked tool-call conversations.

Used two ways: as few-shot examples in stage 1, and later as the format
for SFT training data (stage 2), so the baseline and the fine-tuned model
see identical conversation structure.
"""
from __future__ import annotations

import json
import random

from finbench.data import Example, render_context
from finbench.program import TABLE_OPS, parse_number, parse_program
from finbench.tools import aggregate, arithmetic, row_values


def gold_conversation(example: Example) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": render_context(example)}]
    results: list = []
    for i, step in enumerate(parse_program(example.program)):
        if step.op in TABLE_OPS:
            name = "table_aggregate"
            args = {"row_label": step.args[0], "operation": step.op.removeprefix("table_")}
            result = aggregate(row_values(example.table, step.args[0]), args["operation"])
        else:
            a, b = (results[int(x[1:])] if x.startswith("#") else parse_number(x) for x in step.args)
            name, args = "calculate", {"operation": step.op, "a": a, "b": b}
            result = arithmetic(step.op, float(a), float(b))
        results.append(result)
        call_id = f"call_{i}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps({"result": result})})
    final_id = f"call_{len(results)}"
    answer = results[-1] if isinstance(results[-1], str) else round(results[-1], 5)
    messages.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": final_id, "type": "function",
         "function": {"name": "final_answer", "arguments": json.dumps({"answer": answer})}}]})
    messages.append({"role": "tool", "tool_call_id": final_id, "content": json.dumps({"result": answer})})
    return messages


def pick_fewshot(train: list[Example], k: int, seed: int) -> tuple[list[dict], list[str]]:
    """k examples from the TRAIN split covering 1-, 2- and 3+-step programs.
    Returns the flattened messages and the example ids (saved with the run)."""
    if k == 0:
        return [], []
    rng = random.Random(seed)
    by_steps = {1: [], 2: [], 3: []}
    for ex in sorted(train, key=lambda e: e.id):
        by_steps[min(ex.num_steps, 3)].append(ex)
    chosen: list[Example] = []
    order = [1, 2, 3]
    while len(chosen) < k:
        bucket = by_steps[order[len(chosen) % 3]]
        cand = rng.choice(bucket)
        if cand not in chosen:
            chosen.append(cand)
    msgs = [m for ex in chosen for m in gold_conversation(ex)]
    return msgs, [ex.id for ex in chosen]
