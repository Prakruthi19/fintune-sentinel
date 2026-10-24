"""Replay recorded model replies through the harness, with no model.

A real run saves every model reply in predictions.jsonl ("model_turns").
Replaying those replies through agent + tools + scoring must give the same
verdict as the original run. If it does not, the harness or the scoring
changed, and CI says so before any new number is published.

    # turn a real run into a CI fixture (embeds the FinQA items it used)
    python -m finbench.replay make-fixture results/qwen35-4b-0shot \
        --data-dir data/finqa --out tests/fixtures/replays/qwen35-4b-0shot.jsonl
    # check a fixture (what tests/test_replay_gate.py does in CI)
    python -m finbench.replay check tests/fixtures/replays/qwen35-4b-0shot.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from finbench.data import example_from_record
from finbench.model import AssistantTurn, ToolCall
from finbench.run import evaluate_one, load_rows

# The fields that define a verdict. Latency and tokens are not compared:
# they belong to the original run, not to the scoring code.
VERDICT_FIELDS = ("prediction", "correct", "lenient_correct", "category", "stop_reason", "invalid_calls")


class ReplayModel:
    """Fake ChatModel that returns one item's recorded replies in order. When
    they run out, it raises the recorded server error (or a generic one), the
    same way the original call failed."""

    def __init__(self, name: str, turns: list[dict], error: str | None = None):
        self.name = name
        self._turns = [AssistantTurn(content=t.get("content"),
                                     tool_calls=[ToolCall(c["id"], c["name"], c["arguments"])
                                                 for c in t.get("tool_calls", [])])
                       for t in turns]
        self._error = error

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        if not self._turns:
            raise RuntimeError(self._error or "recording exhausted")
        return self._turns.pop(0)


def verdict(row: dict) -> dict:
    return {k: row.get(k) for k in VERDICT_FIELDS}


def replay_item(item: dict, max_turns: int = 8) -> dict:
    """item: {"example": raw FinQA record, "model_turns": [...], "error": ...}.
    Returns the verdict the CURRENT code gives the recorded replies."""
    ex = example_from_record(item["example"])
    model = ReplayModel("replay", item["model_turns"], item.get("error"))
    return verdict(evaluate_one(model, ex, [], max_turns))


def check(items: list[dict], max_turns: int = 8) -> list[dict]:
    """Every item whose replayed verdict differs from the recorded one."""
    diffs = []
    for item in items:
        got = replay_item(item, max_turns)
        if got != item["expected"]:
            diffs.append({"id": item["example"]["id"], "expected": item["expected"], "got": got})
    return diffs


def load_fixture(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def make_fixture(run_dir: Path, raw_records: list[dict], source: str) -> list[dict]:
    """Fixture items from a finished run. Rows recorded before model_turns
    existed cannot be replayed and are refused rather than skipped silently."""
    by_id = {r["id"]: r for r in raw_records}
    config = json.loads((run_dir / "config.json").read_text())
    items = []
    for row in load_rows(run_dir / "predictions.jsonl"):
        if "model_turns" not in row:
            raise SystemExit(f"{row['id']}: no model_turns recorded; rerun with the current finbench")
        items.append({"source": source, "model": config["model"], "example": by_id[row["id"]],
                      "model_turns": row["model_turns"], "error": row.get("error"),
                      "expected": verdict(row)})
    return items


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make-fixture")
    mk.add_argument("run_dir")
    mk.add_argument("--data-dir", default="data/finqa")
    mk.add_argument("--out", required=True)
    ck = sub.add_parser("check")
    ck.add_argument("fixtures", nargs="+")
    a = p.parse_args(argv)

    if a.cmd == "make-fixture":
        run_dir = Path(a.run_dir)
        raw = json.loads((Path(a.data_dir) / "test.json").read_text())
        items = make_fixture(run_dir, raw, source=f"recorded run {run_dir.name}")
        Path(a.out).write_text("".join(json.dumps(i, default=str) + "\n" for i in items))
        print(f"wrote {len(items)} items to {a.out}")
        return
    failed = 0
    for path in a.fixtures:
        diffs = check(load_fixture(path))
        failed += len(diffs)
        for d in diffs:
            print(f"{path}: {d['id']}\n  expected {d['expected']}\n  got      {d['got']}")
    print("replay gate: " + ("FAILED" if failed else "ok"), f"({failed} changed verdicts)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
