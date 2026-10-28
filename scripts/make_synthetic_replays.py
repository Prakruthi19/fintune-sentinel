"""Write tests/fixtures/replays/synthetic.jsonl: hand-written model replies,
one or more per error category, on the six FinQA items in
tests/fixtures/finqa_sample.json.

SYNTHETIC: no model produced these replies. They imitate failure patterns
seen in the first llama3.2:3b smoke run (invented operations, ignoring
final_answer, tool calls written as text) so the replay gate covers every
category before real recordings exist. Replace or extend with recorded runs
via `python -m finbench.replay make-fixture`.

    python scripts/make_synthetic_replays.py
"""
from __future__ import annotations

import json
from pathlib import Path

from finbench.replay import replay_item

ROOT = Path(__file__).resolve().parent.parent
RAW = {r["id"]: r for r in json.loads((ROOT / "tests/fixtures/finqa_sample.json").read_text())}
OUT = ROOT / "tests/fixtures/replays/synthetic.jsonl"

ETR = "ETR/2016/page_23.pdf-2"    # subtract(5829, 5735) = 94
ADI = "ADI/2011/page_61.pdf-2"    # subtract(153.7, 139.9), divide(#0, 139.9) = 0.09864
MRO = "MRO/2011/page_108.pdf-1"   # table_average(net change for the year) = 3297.66667
MO = "MO/2016/page_19.pdf-1"      # greater(286.61, 198.09) = yes
AES = "AES/2010/page_227.pdf-4"   # multiply(690, 1e6), multiply(#0, 8.75%) = 60375000


def tool(name, i=0, **args):
    return {"content": "", "tool_calls": [{"id": f"c{i}", "name": name, "arguments": json.dumps(args)}]}


def calc(op, a, b, i=0):
    return tool("calculate", i, operation=op, a=a, b=b)


def final(answer, i=9):
    return tool("final_answer", i, answer=answer)


def say(text):
    return {"content": text, "tool_calls": []}


ADI_STEPS = [calc("subtract", 153.7, 139.9), calc("divide", 13.8, 139.9, 1)]

# (description, item id, recorded replies, recorded server error, intended category)
CASES = [
    ("correct two-step", ADI, [*ADI_STEPS, final(0.0986)], None, "correct"),
    ("correct table_average", MRO,
     [tool("table_aggregate", row_label="net change for the year", operation="average"), final(3297.67)],
     None, "correct"),
    ("correct yes/no via greater", MO, [calc("greater", 286.61, 198.09), final("yes")], None, "correct"),
    ("plain-text answer, right number (lenient only)", ETR, [say("The change is 94.")], None, "no_tool_use"),
    ("tool call written as text instead of a call", ETR,
     [say('{"name": "calculate", "parameters": {"operation": "subtract", "a": 5829, "b": 5735}}')],
     None, "no_tool_use"),
    ("invented operation, then text answer", ADI,
     [calc("percentage_change", 139.9, 153.7), say("The growth is about 9.86%.")], None, "no_final_answer"),
    ("right calculation, ignored final_answer", ETR,
     [calc("subtract", 5829, 5735), say("The answer is 94.")], None, "no_final_answer"),
    ("loops until the step budget runs out", ETR,
     [calc("add", 5829, 5735, i) for i in range(8)], None, "no_final_answer"),
    ("percent instead of decimal", ADI, [*ADI_STEPS, final(9.86)], None, "scale_error"),
    ("percent literal used as 8.75 not 0.0875", AES,
     [calc("multiply", 690, 1_000_000), calc("multiply", 690_000_000, 8.75, 1), final(6_037_500_000)],
     None, "scale_error"),
    ("copied the wrong number", ADI,
     [calc("subtract", 153.7, 100.0), calc("divide", 53.7, 100.0, 1), final(0.537)], None, "wrong_numbers"),
    ("right numbers, wrong operation", ADI,
     [calc("subtract", 153.7, 139.9), calc("multiply", 13.8, 139.9, 1), final(1930.62)],
     None, "wrong_operations"),
    ("divided by the wrong year", ADI,
     [calc("subtract", 153.7, 139.9), calc("divide", 13.8, 153.7, 1), final(0.0898)],
     None, "wrong_arrangement"),
    ("server failed (model tag not pulled)", ETR, [], "NotFoundError: model 'qwen3.5:4b' not found",
     "model_error"),
]


def build() -> list[dict]:
    items = []
    for desc, ex_id, turns, error, intended in CASES:
        item = {"source": "synthetic, hand-written", "description": desc, "example": RAW[ex_id],
                "model_turns": turns, "error": error}
        item["expected"] = replay_item(item)
        if item["expected"]["category"] != intended:
            raise SystemExit(f"{desc}: scored {item['expected']['category']}, intended {intended}")
        items.append(item)
    return items


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    items = build()
    OUT.write_text("".join(json.dumps(i) + "\n" for i in items))
    print(f"wrote {len(items)} items to {OUT.relative_to(ROOT)}")
