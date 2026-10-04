"""FinQA's gold programs, e.g. "subtract(5829, 5735), divide(#0, 5735)".

Each step is op(arg1, arg2). Arguments are literal numbers, "#i" (the result
of step i), "const_X" (the constant X; const_m1 is -1), or for table ops a
row label plus "none".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ARITH_OPS = {"add", "subtract", "multiply", "divide", "exp", "greater"}
TABLE_OPS = {"table_sum", "table_average", "table_max", "table_min"}

_STEP = re.compile(r"(\w+)\(([^()]*)\)")


@dataclass(frozen=True)
class Step:
    op: str
    args: tuple[str, str]


def parse_program(program: str) -> list[Step]:
    steps = []
    for op, args in _STEP.findall(program):
        a, b = (x.strip() for x in args.split(",", 1))
        steps.append(Step(op, (a, b)))
    if not steps:
        raise ValueError(f"unparseable program: {program!r}")
    return steps


def parse_number(text: str) -> float:
    if text.startswith("const_"):
        v = text[len("const_"):]
        return -1.0 if v == "m1" else float(v)
    text = text.replace(",", "").strip()
    if text.endswith("%"):  # FinQA writes percentages in programs as "8.75%" = 0.0875
        return float(text[:-1]) / 100
    return float(text)


def literal_operands(program: str) -> list[float]:
    """Numbers the solver must pick out of the filing (not #refs, not
    constants, not table row labels). Used to tell 'picked the wrong
    numbers' apart from 'did the wrong operation'."""
    out = []
    for step in parse_program(program):
        if step.op in TABLE_OPS:
            continue
        for arg in step.args:
            if arg.startswith(("#", "const_")):
                continue
            out.append(parse_number(arg))
    return out


def operations(program: str) -> list[str]:
    return [s.op for s in parse_program(program)]


def constants(program: str) -> list[str]:
    return [a for step in parse_program(program) for a in step.args if a.startswith("const_")]
