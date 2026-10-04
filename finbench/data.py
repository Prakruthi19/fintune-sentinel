"""Loading, rendering and sampling FinQA examples."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from finbench.program import parse_program


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    pre_text: list[str]
    post_text: list[str]
    table: list[list[str]]
    program: str
    gold: float | str  # FinQA exe_ans: a number, or "yes"/"no" for greater()

    @property
    def num_steps(self) -> int:
        return len(parse_program(self.program))


def load_examples(path: str | Path) -> list[Example]:
    raw = json.loads(Path(path).read_text())
    return [
        Example(
            id=r["id"],
            question=r["qa"]["question"],
            pre_text=r["pre_text"],
            post_text=r["post_text"],
            table=r["table"],
            program=r["qa"]["program"],
            gold=r["qa"]["exe_ans"],
        )
        for r in raw
    ]


def render_table(table: list[list[str]]) -> str:
    rows = [" | ".join(cell.strip() for cell in row) for row in table]
    return "\n".join(rows)


def render_context(ex: Example) -> str:
    """The user message: filing text, table, then the question."""
    return (
        "Filing excerpt:\n"
        + " ".join(ex.pre_text)
        + "\n\nTable (first column is the row label):\n"
        + render_table(ex.table)
        + "\n\n"
        + " ".join(ex.post_text)
        + f"\n\nQuestion: {ex.question}"
    )


def stratified_sample(examples: list[Example], n: int, seed: int) -> list[Example]:
    """Sample n examples keeping the 1-step / 2-step / 3+-step mix of the
    full split, so a small run is not accidentally all easy questions."""
    if n >= len(examples):
        return list(examples)
    buckets: dict[int, list[Example]] = defaultdict(list)
    for ex in examples:
        buckets[min(ex.num_steps, 3)].append(ex)
    rng = random.Random(seed)
    picked: list[Example] = []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda e: e.id)
        share = round(n * len(group) / len(examples))
        picked.extend(rng.sample(group, min(share, len(group))))
    # rounding can leave us a few short or over; fix deterministically
    leftover = sorted({e.id for e in examples} - {e.id for e in picked})
    by_id = {e.id: e for e in examples}
    rng.shuffle(leftover)
    while len(picked) < n:
        picked.append(by_id[leftover.pop()])
    return sorted(picked[:n], key=lambda e: e.id)
