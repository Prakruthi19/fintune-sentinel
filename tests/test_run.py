import json

import pytest
from conftest import GoldModel, ScriptedModel

from finbench.report import compare_markdown, summarize, wilson_interval
from finbench.run import run


def test_gold_oracle_scores_100_percent_end_to_end(examples, tmp_path):
    """If a perfect model doesn't score 100%, the harness is broken."""
    exs = list(examples.values())
    summary = run(GoldModel(exs), exs, exs, tmp_path / "oracle", n=len(exs), fewshot_k=0,
                  seed=0, max_turns=8)
    assert summary["accuracy"] == 1.0, summary
    assert summary["error_categories"]["correct"] == len(exs)
    assert summary["invalid_tool_call_rate"] == 0.0


def test_run_is_resumable_and_writes_artifacts(examples, tmp_path):
    exs = list(examples.values())
    out = tmp_path / "r"
    run(GoldModel(exs), exs, exs, out, n=3, fewshot_k=0, seed=0, max_turns=8)
    first = (out / "predictions.jsonl").read_text()
    resumed = GoldModel(exs)
    resumed.complete = lambda *a: (_ for _ in ()).throw(AssertionError("should not be called"))
    run(resumed, exs, exs, out, n=3, fewshot_k=0, seed=0, max_turns=8)  # all done: no model calls
    assert (out / "predictions.jsonl").read_text() == first
    with pytest.raises(SystemExit, match="different"):
        run(ScriptedModel([]), exs, exs, out, n=3, fewshot_k=0, seed=0, max_turns=8)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["n"] == 3 and cfg["excluded_test_ids"] == []
    assert (out / "sample_ids.txt").read_text().count("\n") == 3
    md = compare_markdown([out])
    assert "| r | gold-oracle | 0 | 3 | 100.0%" in md


def test_wilson_interval_and_summary_math():
    lo, hi = wilson_interval(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.61
    rows = [{"correct": c, "category": "correct" if c else "wrong_numbers", "num_steps": s, "tool_calls": [{}],
             "invalid_calls": 0, "latency_s": lat, "prompt_tokens": 10, "completion_tokens": 2}
            for c, s, lat in [(True, 1, 1.0), (False, 2, 2.0), (True, 4, 3.0), (True, 1, 10.0)]]
    s = summarize(rows)
    assert s["accuracy"] == 0.75 and s["accuracy_by_steps"]["3+"] == {"n": 1, "accuracy": 1.0}
    assert s["latency_s"] == {"p50": 2.0, "p95": 10.0}
