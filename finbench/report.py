"""Summaries for one run, and a comparison table across runs.

    python -m finbench.report results/run-a results/run-b > docs/results/stage1.md
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

from finbench.scoring import CATEGORIES


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion. With n=200 the interval is
    roughly +/-7 points, so differences smaller than that are not claimed."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return s[idx]


def summarize(rows: list[dict]) -> dict:
    model_errors = sum(r["category"] == "model_error" for r in rows)
    rows = [r for r in rows if r["category"] != "model_error"]  # infrastructure failures are not scored
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    lo, hi = wilson_interval(k, n)
    cats = Counter(r["category"] for r in rows)
    total_calls = sum(len(r["tool_calls"]) for r in rows)
    by_steps = {}
    for bucket in ("1", "2", "3+"):
        sub = [r for r in rows if ("3+" if r["num_steps"] >= 3 else str(r["num_steps"])) == bucket]
        if sub:
            by_steps[bucket] = {"n": len(sub), "accuracy": round(sum(r["correct"] for r in sub) / len(sub), 4)}
    lat = [r["latency_s"] for r in rows]
    return {
        "n": n,
        "excluded_model_errors": model_errors,
        "accuracy": round(k / n, 4) if n else 0.0,
        "accuracy_95ci": [round(lo, 4), round(hi, 4)],
        "lenient_accuracy": round(sum(r.get("lenient_correct", r["correct"]) for r in rows) / n, 4) if n else 0.0,
        "accuracy_by_steps": by_steps,
        "error_categories": {c: cats.get(c, 0) for c in CATEGORIES},
        "invalid_tool_call_rate": round(sum(r["invalid_calls"] for r in rows) / total_calls, 4) if total_calls else 0.0,
        "latency_s": {"p50": round(percentile(lat, 0.5), 3), "p95": round(percentile(lat, 0.95), 3)},
        "avg_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / n, 1) if n else 0.0,
        "avg_completion_tokens": round(sum(r["completion_tokens"] for r in rows) / n, 1) if n else 0.0,
    }


def compare_markdown(run_dirs: list[Path]) -> str:
    runs = [(d.name, json.loads((d / "summary.json").read_text()),
             json.loads((d / "config.json").read_text())) for d in run_dirs]
    head = ("| run | model | few-shot | n | accuracy (95% CI) | lenient accuracy | invalid calls | "
            "p50 / p95 latency (s) | hardware |\n|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for name, s, c in runs:
        lo, hi = s["accuracy_95ci"]
        lines.append(f"| {name} | {c['model']} | {c['fewshot_k']} | {s['n']} | "
                     f"{s['accuracy']:.1%} ({lo:.1%}–{hi:.1%}) | {s.get('lenient_accuracy', 0):.1%} | "
                     f"{s['invalid_tool_call_rate']:.1%} | "
                     f"{s['latency_s']['p50']} / {s['latency_s']['p95']} | {c.get('hardware') or 'not recorded'} |")
    lines.append("\n**Where the errors are** (count of items per category)\n")
    lines.append("| run | " + " | ".join(CATEGORIES) + " |")
    lines.append("|---|" + "---|" * len(CATEGORIES))
    for name, s, _ in runs:
        lines.append(f"| {name} | " + " | ".join(str(s["error_categories"][c]) for c in CATEGORIES) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(compare_markdown([Path(p) for p in sys.argv[1:]]))
