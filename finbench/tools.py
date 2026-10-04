"""The tools the model can call, and the runtime that executes them.

Design choice (see docs/adr/001): the model never does arithmetic itself.
It picks numbers and operations; code computes. A lookup tool for single
cells is deliberately NOT included yet. Stage 1 measures how often models
copy the wrong number; a lookup tool is added only if that error dominates.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from finbench.data import Example
from finbench.program import ARITH_OPS, TABLE_OPS, parse_number, parse_program

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Apply one arithmetic operation to two numbers and return the result. "
                "Use it for every calculation; do not compute in your head. "
                "'greater' returns 'yes' if a > b, else 'no'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": sorted(ARITH_OPS)},
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["operation", "a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "table_aggregate",
            "description": (
                "Sum, average, max or min of all numeric cells in one table row, "
                "identified by its row label (the first column). Percent cells are "
                "returned as decimals (8% -> 0.08)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "row_label": {"type": "string"},
                    "operation": {"type": "string", "enum": ["sum", "average", "max", "min"]},
                },
                "required": ["row_label", "operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Submit the final answer. Give the raw result of the calculation: "
                "a ratio or percentage change as a decimal (0.145, not 14.5%), "
                "or 'yes'/'no' for comparison questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": ["number", "string"]}},
                "required": ["answer"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


class ToolError(Exception):
    """A call the runtime refused: unknown tool, bad JSON, bad arguments."""


def cell_to_number(cell: str) -> float | None:
    """Parse a FinQA table cell. Observed formats:
    '$ 5735', '-603 ( 603 )', '$ -61.1 ( 61.1 )', '4.70% ( 4.70 % )', '( 123 )', '-'.
    A trailing '( ... )' repeats the value from the original PDF and is
    dropped when a value precedes it; a bare '( x )' is accounting notation
    for a negative. Percent cells are returned as decimals (8% -> 0.08),
    matching how FinQA's gold programs treat percentages."""
    s = cell.strip()
    m = re.match(r"^(.*?\S.*?)\s*\(([^()]*)\)\s*$", s)
    if m and m.group(1).strip("$ "):
        s = m.group(1)
    s = s.replace("$", "").replace(",", "").strip()
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("() ").strip()
    pct = s.endswith("%")
    s = s.rstrip("%").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if negative:
        v = -v
    return v / 100 if pct else v


def row_values(table: list[list[str]], row_label: str) -> list[float]:
    want = _norm(row_label)
    for row in table:
        if row and _norm(row[0]) == want:
            return [v for v in (cell_to_number(c) for c in row[1:]) if v is not None]
    raise ToolError(f"no table row labelled {row_label!r}")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def arithmetic(op: str, a: float, b: float) -> float | str:
    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    if op == "multiply":
        return a * b
    if op == "divide":
        if b == 0:
            raise ToolError("division by zero")
        return a / b
    if op == "exp":
        return math.pow(a, b)
    if op == "greater":
        return "yes" if a > b else "no"
    raise ToolError(f"unknown operation {op!r}")


def aggregate(values: list[float], op: str) -> float:
    if not values:
        raise ToolError("row has no numeric cells")
    return {"sum": sum, "average": lambda v: sum(v) / len(v), "max": max, "min": min}[op](values)


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict | None
    result: object = None
    error: str | None = None


@dataclass
class ToolRuntime:
    example: Example
    calls: list[ToolCallRecord] = field(default_factory=list)

    def execute(self, name: str, raw_arguments: str | dict) -> ToolCallRecord:
        record = ToolCallRecord(name=name, arguments=None)
        self.calls.append(record)
        try:
            args = raw_arguments if isinstance(raw_arguments, dict) else json.loads(raw_arguments or "{}")
            if not isinstance(args, dict):
                raise ToolError("arguments must be a JSON object")
            record.arguments = args
            record.result = self._dispatch(name, args)
        except (ToolError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            record.error = f"{type(e).__name__}: {e}"
        return record

    def _dispatch(self, name: str, args: dict) -> object:
        if name not in TOOL_NAMES:
            raise ToolError(f"unknown tool {name!r}")
        if name == "calculate":
            return arithmetic(args["operation"], float(args["a"]), float(args["b"]))
        if name == "table_aggregate":
            op = args["operation"]
            if op not in {"sum", "average", "max", "min"}:
                raise ToolError(f"unknown aggregate {op!r}")
            return aggregate(row_values(self.example.table, args["row_label"]), op)
        return args["answer"]  # final_answer: echoed back, scored later


def execute_gold_program(example: Example) -> float | str:
    """Run FinQA's gold program through the SAME tool implementations the
    model uses. If this disagrees with FinQA's own answer, the example is
    excluded from evaluation (the tools cannot express it), and the count is
    reported rather than hidden."""
    results: list[float | str] = []
    for step in parse_program(example.program):
        if step.op in TABLE_OPS:
            values = row_values(example.table, step.args[0])
            results.append(aggregate(values, step.op.removeprefix("table_")))
            continue
        a, b = (results[int(x[1:])] if x.startswith("#") else parse_number(x) for x in step.args)
        results.append(arithmetic(step.op, float(a), float(b)))
    return results[-1]
