"""Answer matching and the error taxonomy.

The taxonomy is heuristic: it compares the model's tool calls with FinQA's
gold program. It is meant to show *where* a model fails (format, numbers,
operations, scale), not to be a perfect judge of each item.
"""
from __future__ import annotations

from dataclasses import dataclass

from finbench.program import constants, literal_operands, operations, parse_number

REL_TOL = 0.005  # 0.5% relative; FinQA answers are rounded to 5 decimals
ABS_TOL = 1e-4


def to_float(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "").replace("$", "")
        pct = s.endswith("%")
        try:
            v = float(s.rstrip("%"))
        except ValueError:
            return None
        return v / 100 if pct else v
    return None


def answers_match(pred, gold) -> bool:
    if isinstance(gold, str):  # yes/no
        return isinstance(pred, str) and pred.strip().lower() == gold.lower()
    p = to_float(pred)
    if p is None:
        return False
    return abs(p - gold) <= max(REL_TOL * abs(gold), ABS_TOL)


def is_scale_error(pred, gold) -> bool:
    """Right calculation, wrong unit: off by exactly 100x or 1000x."""
    if isinstance(gold, str):
        return False
    p = to_float(pred)
    if p is None or gold == 0:
        return False
    return any(answers_match(p, gold * f) for f in (100, 0.01, 1000, 0.001))


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-6, 1e-4 * abs(b))


# Numbers a solver may legitimately introduce without finding them in the
# filing (unit conversions, "per year" splits). Not counted as wrong numbers.
FREE_CONSTANTS = (1.0, 2.0, 100.0, 1000.0, 1_000_000.0)


@dataclass(frozen=True)
class Verdict:
    correct: bool
    category: str


CATEGORIES = [
    "correct",
    "no_tool_use",       # answered in plain text, never called a tool
    "no_final_answer",   # used tools but never submitted (or ran out of steps)
    "scale_error",       # 100x / 1000x off: percent vs decimal, thousands vs millions
    "wrong_numbers",     # missed a number the gold program needs, or used one it doesn't
    "wrong_operations",  # right numbers, different operation sequence
    "wrong_arrangement", # right numbers and operations, combined differently
                         # (e.g. divided by the wrong year), or a rounding slip
]


def classify(program: str, gold, final, calculate_calls: list[dict], any_tool_called: bool,
             prior_results: list) -> Verdict:
    """calculate_calls: successful calculate() argument dicts in order.
    prior_results: results of earlier tool calls (so reused intermediate
    results are not mistaken for numbers copied from the filing)."""
    if final is not None and answers_match(final, gold):
        return Verdict(True, "correct")
    if not any_tool_called:
        return Verdict(False, "no_tool_use")
    if final is None:
        return Verdict(False, "no_final_answer")
    if is_scale_error(final, gold):
        return Verdict(False, "scale_error")
    used = []
    for args in calculate_calls:
        for key in ("a", "b"):
            v = to_float(args.get(key))
            if v is None:
                continue
            reused = any(isinstance(r, (int, float)) and not isinstance(r, bool) and _close(v, r)
                         for r in prior_results)
            if not reused:
                used.append(v)
    needed = literal_operands(program)
    allowed = needed + [parse_number(c) for c in constants(program)] + list(FREE_CONSTANTS)
    missing = any(not any(_close(u, n) for u in used) for n in needed)
    extraneous = any(not any(_close(u, a) for a in allowed) for u in used)
    if missing or extraneous:
        return Verdict(False, "wrong_numbers")
    if operations(program) != [a.get("operation") for a in calculate_calls]:
        return Verdict(False, "wrong_operations")
    return Verdict(False, "wrong_arrangement")
