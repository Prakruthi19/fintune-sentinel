"""Benchmarks GGUF models on the held-out FinQA test set, using llama.cpp on CPU.

Replaces the 3-prompt eyeball comparison with numbers you can defend:
  * execution accuracy: final number within 1% of the gold answer (handles 0.133 vs 13.3%)
  * unit compliance and trade-advice rate, from the same Policy-Sentinel audit the app uses
  * throughput: prompt + generation tokens/sec and latency on this machine's CPU

    python -m scripts.evaluate --finqa data/test.json --n 200 \
        --model base=models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
        --model sentinel_v1=models/sentinel-v1.Q4_K_M.gguf \
        --model sentinel_v2=runs/v2/gguf/unsloth.Q4_K_M.gguf
"""
import argparse
import json
import re
import statistics
import time


from sentinel.data import SYSTEM_PROMPT, from_finqa
from sentinel.governance import audit

NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*\s*%?")


def parse_number(s):
    s = s.replace("$", "").replace(",", "").strip()
    pct = s.endswith("%")
    try:
        return float(s.rstrip("%").strip()), pct
    except ValueError:
        return None, pct


def final_number(text):
    tail = text.split("Answer:")[-1]
    found = NUM.findall(tail) or NUM.findall(text)
    return parse_number(found[-1]) if found else (None, False)


def is_correct(pred_text, gold):
    pred, _ = final_number(pred_text)
    gold_val, _ = parse_number(str(gold))
    if pred is None or gold_val is None:
        return False
    # Accept either scale for percentages: 13.3% vs 0.133.
    for cand in (pred, pred / 100, pred * 100):
        if abs(cand - gold_val) <= max(0.01 * abs(gold_val), 1e-3):
            return True
    return False


def run(model_path, examples, threads, n_ctx):
    from llama_cpp import Llama  # imported here so tests and CI don't need the inference runtime

    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=threads, n_gpu_layers=0, verbose=False)
    rows = []
    for ex in examples:
        msgs = ex["messages"][:2]  # system + user; the assistant turn is the reference
        t0 = time.perf_counter()
        out = llm.create_chat_completion(messages=msgs, max_tokens=384, temperature=0.0)
        dt = time.perf_counter() - t0
        text = out["choices"][0]["message"]["content"]
        a = audit(text)
        rows.append({
            "id": ex["id"], "gold": ex["gold"], "output": text,
            "correct": is_correct(text, ex["gold"]),
            "trade_advice": a.redacted, "unitless": any("units" in f for f in a.findings),
            "latency_s": dt, "gen_tokens": out["usage"]["completion_tokens"],
        })
    del llm
    return rows


def summarize(rows):
    n = len(rows)
    return {
        "n": n,
        "exec_accuracy": sum(r["correct"] for r in rows) / n,
        "unitless_rate": sum(r["unitless"] for r in rows) / n,
        "trade_advice_rate": sum(r["trade_advice"] for r in rows) / n,
        "median_latency_s": statistics.median(r["latency_s"] for r in rows),
        "gen_tokens_per_s": sum(r["gen_tokens"] for r in rows) / sum(r["latency_s"] for r in rows),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--finqa", default="data/test.json")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--model", action="append", required=True, help="name=path/to/model.gguf")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--out", default="eval_results.json")
    args = p.parse_args()

    examples = []
    for e in json.load(open(args.finqa)):
        msgs = from_finqa(e)
        if msgs:
            msgs[0]["content"] = SYSTEM_PROMPT + " End with 'Answer: <number>'."
            examples.append({"id": e.get("id"), "gold": e["qa"].get("exe_ans", e["qa"]["answer"]), "messages": msgs})
    examples = examples[: args.n]

    report = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        rows = run(path, examples, args.threads, args.n_ctx)
        report[name] = {"summary": summarize(rows), "rows": rows}
        print(name, json.dumps(report[name]["summary"], indent=2))
    json.dump(report, open(args.out, "w"), indent=2)

    print("\n| Model | FinQA exec acc | Unit-less rate | Trade-advice rate | Median latency | Gen tok/s |")
    print("|---|---|---|---|---|---|")
    for name, r in report.items():
        s = r["summary"]
        print(f"| {name} | {s['exec_accuracy']:.1%} | {s['unitless_rate']:.1%} | {s['trade_advice_rate']:.1%} "
              f"| {s['median_latency_s']:.1f}s | {s['gen_tokens_per_s']:.1f} |")


if __name__ == "__main__":
    main()
