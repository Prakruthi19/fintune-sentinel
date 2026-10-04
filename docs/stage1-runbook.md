# Stage 1 runbook: running the baseline

Everything here runs for free. Commit `results/<run-name>/` after each run;
the predictions file is the evidence behind every number in the README.

## 0. Setup

```bash
pip install -e ".[dev]"
python scripts/download_finqa.py      # pinned commit + checksums
pytest                                # no model needed
```

## 1. Small models on your laptop (Ollama, CPU)

```bash
# Prompts are ~1.5-3k tokens (more with few-shot). Ollama's default context
# is smaller and it truncates SILENTLY, which would corrupt the results.
export OLLAMA_CONTEXT_LENGTH=16384
ollama serve &
ollama pull <model-tag>

python -m finbench.run --model <model-tag> --base-url http://localhost:11434/v1 \
  --n 200 --fewshot 0 --run-name <short-name>-0shot --hardware "<your CPU/RAM>"
python -m finbench.run --model <model-tag> --base-url http://localhost:11434/v1 \
  --n 200 --fewshot 3 --run-name <short-name>-3shot --hardware "<your CPU/RAM>"
```

Start with `--n 10` to check the setup and timing before a full run. Runs
are resumable: rerun the same command after an interruption.

Candidate models (check the exact tag in the Ollama library on the day you
run, and confirm it has the `tools` tag): a current Qwen small model
(~4B), Gemma 4 E4B, Phi-4-mini, plus `llama3.2:3b` as the old baseline the
project started from.

If a model "thinks" before answering (some Qwen versions do by default),
record whether thinking was on in the run name: it changes both accuracy
and latency.

## 2. Same models on a free Kaggle GPU (vLLM)

In a Kaggle notebook with a T4 GPU:

```bash
pip install vllm
vllm serve <hf-model-id> --max-model-len 16384 --enable-auto-tool-choice \
  --tool-call-parser <parser for that model family> &
python -m finbench.run --model <hf-model-id> --base-url http://localhost:8000/v1 \
  --n 200 --fewshot 0 --run-name <short-name>-0shot-t4 --hardware "Kaggle T4"
```

The tool-call parser differs by model family; see vLLM's tool-calling docs.
Download the `results/` folder from the notebook and commit it.

## 3. Large-model baseline (free API tier)

```bash
export LARGE_MODEL_KEY=...           # never pass keys on the command line
python -m finbench.run --model <provider-model-id> --base-url <provider OpenAI-compatible URL> \
  --api-key-env LARGE_MODEL_KEY --n 200 --fewshot 0 --run-name large-0shot --hardware "API"
```

Free tiers are rate-limited; the run resumes where it stopped. API
latency on a free tier is not representative of paid tiers.

## 4. Compare

```bash
python -m finbench.report results/* > docs/results/stage1.md
```

## Before writing any conclusion

1. Open 10 wrong predictions per model and check the error category by hand.
2. Treat differences under ~10 points as noise at n=200 (see ADR 001).
3. Write down which model and setup (0-shot or 3-shot) becomes the
   baseline to beat, and the one or two error categories worth fixing.
