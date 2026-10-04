# FinTune-Sentinel

**When is a small fine-tuned model worth it over a large API model?**
This project answers that for one narrow, high-volume task: numerical
questions over company financial filings. Every number in this README
comes from a script in this repo and states its setup. Anything not yet
measured is marked as planned.

## Status

| Stage | What it answers | Status |
|---|---|---|
| v2 | Clean training pipeline, compliance checks, CPU app, direct-answer eval script | code done; **no eval results recorded yet** |
| 1 | How well do current small models already do this with tools, and where do they fail? | **harness built and tested; model runs pending** |
| 2 | Training data: tool-use traces distilled from a large model, leakage-checked, versioned | planned |
| 3 | Fine-tuning: SFT, then GRPO with "answer is correct" as the reward | planned |
| 4 | Quantization, CPU/GPU serving, load tests, cost per 1,000 requests vs an API | planned |
| 5 | MLOps: experiment tracking, model registry with an eval gate, CI, rollback | planned |

Fine-tuning a new model (stage 3) happens only if stage 1 shows a gap that
prompting alone does not close. If prompting is enough, that is the result.

## Two ways to answer, both measured

| | Direct answer (`scripts/evaluate.py`) | Tool use (`finbench/`) |
|---|---|---|
| How the model answers | Free text ending in `Answer: <number>` | Calls `calculate` / `table_aggregate`, then `final_answer` |
| Who does the arithmetic | The model | Code |
| Runs on | llama.cpp, CPU, GGUF files | Any OpenAI-compatible server: Ollama, llama.cpp, vLLM, hosted APIs |
| Extra checks | Unit-less numbers, trade advice (Policy-Sentinel) | Error category per item |

Comparing the two is part of stage 1. Their scoring rules differ today
(see [ADR 001](docs/adr/001-stage1-baseline-design.md)) and will be unified
before any side-by-side number is published.

## Stage 1: the tool-use harness (`finbench/`)

The model reads a filing excerpt and table from
[FinQA](https://github.com/czyssrs/FinQA) and must use tools for all
arithmetic. Each answer is checked against FinQA's gold answer, and each
failure is sorted into one category: no tool use, no final answer, scale
error (percent vs decimal), wrong numbers, wrong operations, or wrong
arrangement.

Verified so far, with no model involved:
- The tools reproduce FinQA's own answers for 1147/1147 test and 883/883
  dev programs (2/6251 train programs excluded; see ADR 001).
- A perfect "oracle" model replaying gold solutions scores 100% through the
  full harness on a 200-item stratified sample.
- Unit tests for finbench and sentinel run in CI on Python 3.11 and 3.12.

Run it: [docs/stage1-runbook.md](docs/stage1-runbook.md).
Results: *none yet*.

## v2: training pipeline, compliance checks, app

- `sentinel/data.py` converts the three source datasets to one chat format.
  It fixes two bugs in the original notebook's data: Llama-2 records were
  parsed so the answer landed in the user turn, and a literal `</s>` was
  left in training targets.
- `scripts/train_sft.py`: LoRA / QLoRA SFT with Unsloth. FinQA train split
  only (dev/test held out), loss on assistant tokens only, over-length
  examples dropped instead of truncated, GGUF export.
- `sentinel/governance.py` (Policy-Sentinel): flags numbers without units
  and redacts trade recommendations, using phrase-level patterns so
  "buyback", "shareholders" and "selling, general and administrative" are
  not mistaken for advice.
- `app.py`: Gradio chat on a 4-bit GGUF via llama.cpp, using the model's
  own chat template, with a Policy-Sentinel verdict on every answer.
- `scripts/quant_demo.py`: a small re-implementation of Q4_0 and NF4
  quantization, for explaining how 4-bit weights work.

## v1 (2025): the original demo

Fine-tuned Llama 3.2 3B Instruct with QLoRA on three Kaggle datasets,
exported to GGUF (Q4_K_M), served on CPU in a Gradio app. v1 was not
evaluated on a held-out set, so earlier claims about speed, accuracy and
comparisons with FinGPT were removed. Llama 3.2 3B is now a 2024 model;
stage 1 includes it as the old baseline next to current small models.

Training data (v1 and v2):
1. [Financial Q&A-10k](https://www.kaggle.com/datasets/yousefsaeedian/financial-q-and-a-10k) (Yousef Saeedian)
2. [Finance-Llama2-1k](https://www.kaggle.com/datasets/yousefsaeedian/finance-llama2-1k) (Yousef Saeedian)
3. [Question Answering on Financial Data](https://www.kaggle.com/datasets/visalakshiiyer/question-answering-financial-data) (Visalakshi Iyer)
4. FinQA train split (v2)

## Data and license notes

FinQA: Chen et al., *FinQA: A Dataset of Numerical Reasoning over Financial
Data*, EMNLP 2021, MIT license. Downloaded at a pinned commit with checksums
by `scripts/download_finqa.py`; not committed. `tests/fixtures/finqa_sample.json`
contains six FinQA test items for unit tests.
