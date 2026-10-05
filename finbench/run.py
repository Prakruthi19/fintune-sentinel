"""CLI: run one model on a fixed FinQA sample and write results.

    python -m finbench.run --model qwen3.5:4b --base-url http://localhost:11434/v1 \
        --n 200 --fewshot 0 --run-name qwen35-4b-zeroshot

Writes results/<run-name>/{config.json, predictions.jsonl, summary.json}.
Resumable: items already in predictions.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

from finbench.agent import solve
from finbench.data import Example, load_examples, stratified_sample
from finbench.fewshot import pick_fewshot
from finbench.model import ChatModel, OpenAICompatibleModel
from finbench.report import summarize
from finbench.scoring import answers_match, classify, text_answer
from finbench.tools import execute_gold_program


def usable(examples: list[Example]) -> tuple[list[Example], list[str]]:
    """Keep only examples whose gold program our tools reproduce exactly."""
    keep, dropped = [], []
    for ex in examples:
        try:
            ok = answers_match(execute_gold_program(ex), ex.gold)
        except Exception:
            ok = False
        (keep if ok else dropped).append(ex if ok else ex.id)
    return keep, dropped


def evaluate_one(model: ChatModel, ex: Example, fewshot: list[dict], max_turns: int) -> dict:
    traj = solve(model, ex, fewshot, max_turns=max_turns)
    calc_args = [c.arguments for c in traj.tool_calls
                 if c.name == "calculate" and not c.error and c.arguments]
    prior_results = [c.result for c in traj.tool_calls if not c.error and c.name != "final_answer"]
    verdict = classify(ex.program, ex.gold, traj.final_answer, calc_args,
                       any_tool_called=bool(traj.tool_calls), prior_results=prior_results,
                       model_error=traj.stop_reason == "model_error")
    last_text = traj.turns[-1].content if traj.turns else None
    lenient = verdict.correct or (traj.final_answer is None and answers_match(text_answer(last_text), ex.gold))
    return {
        "id": ex.id,
        "num_steps": ex.num_steps,
        "gold": ex.gold,
        "gold_program": ex.program,
        "prediction": traj.final_answer,
        "correct": verdict.correct,
        "lenient_correct": bool(lenient),
        "category": verdict.category,
        "stop_reason": traj.stop_reason,
        "error": traj.error,
        "tool_calls": [asdict(c) for c in traj.tool_calls],
        "invalid_calls": traj.invalid_calls,
        "turns": len(traj.turns),
        "latency_s": round(traj.latency_s, 3),
        "prompt_tokens": traj.prompt_tokens,
        "completion_tokens": traj.completion_tokens,
        "final_text": traj.turns[-1].content if traj.turns else None,
    }


MAX_CONSECUTIVE_MODEL_ERRORS = 3


def load_rows(path: Path) -> list[dict]:
    """Rows from predictions.jsonl, keeping the LAST row per id (a retried
    item replaces its earlier failed attempt)."""
    if not path.exists():
        return []
    latest: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["id"]] = r
    return list(latest.values())


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run(model: ChatModel, test: list[Example], train: list[Example], out_dir: Path,
        n: int, fewshot_k: int, seed: int, max_turns: int, extra_config: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    test_ok, dropped = usable(test)
    train_ok, _ = usable(train) if fewshot_k else (train, [])
    sample = stratified_sample(test_ok, n, seed)
    fewshot, fewshot_ids = pick_fewshot(train_ok, fewshot_k, seed)

    config = {
        "model": model.name, "n": len(sample), "fewshot_k": fewshot_k, "seed": seed,
        "max_turns": max_turns, "fewshot_ids": fewshot_ids,
        "excluded_test_ids": dropped, "git_sha": git_sha(),
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "cpu_count": os.cpu_count()},
        **(extra_config or {}),
    }
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        # Resuming: refuse to mix predictions from a different model or setup.
        old = json.loads(cfg_path.read_text())
        changed = [k for k in ("model", "n", "fewshot_k", "seed", "max_turns") if old.get(k) != config[k]]
        if changed:
            raise SystemExit(f"{out_dir} holds a run with different {changed}; use a new --run-name")
    else:
        cfg_path.write_text(json.dumps(config, indent=2))
    (out_dir / "sample_ids.txt").write_text("\n".join(e.id for e in sample) + "\n")

    pred_path = out_dir / "predictions.jsonl"
    done = {r["id"] for r in load_rows(pred_path) if r["category"] != "model_error"}  # retry failed calls
    consecutive_errors = 0
    with pred_path.open("a") as f:
        for i, ex in enumerate(sample, 1):
            if ex.id in done:
                continue
            row = evaluate_one(model, ex, fewshot, max_turns)
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            print(f"[{i}/{len(sample)}] {ex.id}: {row['category']} ({row['latency_s']}s)", flush=True)
            consecutive_errors = consecutive_errors + 1 if row["category"] == "model_error" else 0
            if consecutive_errors >= MAX_CONSECUTIVE_MODEL_ERRORS:
                raise SystemExit(f"stopping: {consecutive_errors} model calls failed in a row. "
                                 f"Last error: {row['error']}\nFix the model/server and rerun the same "
                                 f"command; failed items are retried.")

    rows = load_rows(pred_path)
    summary = summarize(rows)
    summary["config"] = {k: config[k] for k in ("model", "n", "fewshot_k", "seed", "git_sha")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://localhost:11434/v1", help="any OpenAI-compatible endpoint")
    p.add_argument("--api-key-env", default=None, help="env var holding the API key (never pass keys as args)")
    p.add_argument("--data-dir", default="data/finqa")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--fewshot", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--run-name", required=True)
    p.add_argument("--hardware", default="", help="free text, e.g. 'M2 laptop CPU' or 'Kaggle T4'")
    a = p.parse_args(argv)

    api_key = os.environ.get(a.api_key_env, "") if a.api_key_env else "not-needed"
    model = OpenAICompatibleModel(a.model, a.base_url, api_key=api_key or "not-needed")
    data = Path(a.data_dir)
    summary = run(model, load_examples(data / "test.json"), load_examples(data / "train.json"),
                  Path("results") / a.run_name, a.n, a.fewshot, a.seed, a.max_turns,
                  extra_config={"base_url": a.base_url, "hardware": a.hardware})
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
