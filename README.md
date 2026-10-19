# 🛡️ Sentinel: a free financial health check for small businesses

[![tests](https://github.com/Prakruthi19/fintune-sentinel/actions/workflows/tests.yml/badge.svg)](https://github.com/Prakruthi19/fintune-sentinel/actions/workflows/tests.yml)

> Type in (or paste) a few numbers from your profit & loss and balance sheet and get a health score, rated growth, profit, cash and debt checks with the exact calculation behind each one, and a plain-English summary with questions for your accountant. No sign-up, nothing saved, no investment advice.

**Who it's for:** small business owners and freelancers who want to understand their own numbers, and finance students (an *Ask a question* mode explains concepts with the working shown).

**How it works:** every ratio is computed exactly in Python (`sentinel/health.py`). A Llama 3.2 3B model, fine-tuned on SEC 10-K and FinQA data and quantized to 4-bit GGUF so it runs on a plain CPU, only writes the explanation, and every word it writes passes a compliance layer that blocks investment advice.

**Live demo:** https://fintune-sentinel-245006759125.us-east4.run.app (Google Cloud Run, CPU only; the first request after idle takes ~30 s to wake up)

---

## Why

General-purpose LLMs answering questions about 10-K filings tend to (1) do arithmetic loosely, (2) drop units ("revenue grew 1.133"), and (3) drift into investment advice, which is a liability for any regulated firm. FinTune-Sentinel targets all three on hardware an analyst already has: no GPU at inference time.

## Architecture

```mermaid
flowchart LR
    A[10-K Q&A<br/>Finance theory<br/>FinQA tables] --> B[sentinel/data.py<br/>one chat template]
    B --> C[QLoRA SFT<br/>Unsloth, T4 GPU]
    C --> D[Merge + quantize<br/>GGUF Q4_K_M, ~2 GB]
    D --> E[llama.cpp on CPU<br/>mmap + AVX2]
    E --> F[Policy-Sentinel<br/>compliance audit]
    F --> G[React UI<br/>streamed over SSE]
```

| Component | What it does | Code |
| :--- | :--- | :--- |
| Data pipeline | Converts three datasets into one chat format; renders FinQA tables as markdown and its gold calculation program as step-by-step targets | `sentinel/data.py` |
| Training | QLoRA (r=32) on all attention + MLP projections, loss on assistant tokens only, held-out eval split, GGUF export | `scripts/train_sft.py` |
| Inference API | `llama-cpp-python`, 4-bit Q4_K_M weights, model's own Llama-3 chat template; FastAPI streams tokens over server-sent events, redacting trade advice as it is generated | `app.py` |
| Health check | Type the numbers in or upload a statement (PDF, Excel, Word, CSV, read in memory), statement parser, exact ratio math, good/watch/risk ratings and a health score; the model explains the results | `sentinel/health.py` |
| Web app | React + TypeScript + Tailwind: health check form and report, Ask mode, light/dark, print to PDF | `web/` |
| Policy-Sentinel | Redacts trade recommendations; flags financial figures with no unit | `sentinel/governance.py` |
| Evaluation | FinQA execution accuracy, unit compliance, trade-advice rate, CPU latency and tokens/sec | `scripts/evaluate.py` |
| Quantization explainer | From-scratch Q4_0 and NF4 quantizers plus a GGUF header writer/reader | `scripts/quant_demo.py` |

## Results

**Quality** — `scripts/evaluate.py` on 300 held-out FinQA test questions, identical Q4_K_M GGUF files run through `llama.cpp`, temperature 0. (Scores don't depend on hardware, so this runs on a Kaggle T4 for speed.)

| Model | FinQA exec. accuracy | Answers with unit-less figures | Trade-advice rate |
| :--- | :--- | :--- | :--- |
| Llama 3.2 3B Instruct (base) | 47.7% | 63.7% | 0.0% |
| Sentinel v1 | 18.3% | 13.3% | 0.0% |
| **Sentinel v2** | **66.0%** | 95.0% | 0.0% |

v2 gets 18.3 points more right than the base model. v1 scored *below* the base model because of the training bugs listed under [What I learned building v2](#what-i-learned-building-v2). The unit-less column is inflated by the benchmark itself: every prompt asks the model to end with `Answer: <number>`, and v2 follows that format most closely, so its bare final numbers get flagged.

**Speed on CPU only** (Q4_K_M, ~2 GB):

| Hardware | Generation speed | Example |
| :--- | :--- | :--- |
| Laptop, Intel i5-1334U (15 W, 2 P-cores), 16 GB RAM | ~3 tokens/s | short answer in ~13 s |
| Google Cloud Run, 2 vCPU / 4 GB | _pending_ | YoY growth question answered in ~21 s |

## Policy-Sentinel

Every response passes through `audit()` before it is shown:

1. **Trade advice → `REJECTED`.** Phrase-level patterns ("strong buy rating", "I recommend selling", "price target") are redacted and a disclaimer is appended. Accounting vocabulary such as *selling, general & administrative*, *shareholders* and *buyback* is deliberately not matched (the v1 substring check flagged all of them).
2. **Unit-less figures → `FLAGGED`.** Decimals and values ≥ 100 with no `$`, `%`, or scale word are reported; years, list numbers and labels like "Item 1A" are ignored.
3. Otherwise **`APPROVED`.** The status is shown under every answer.

## What I learned building v2

Auditing the v1 training run turned up real bugs. Fixing them was most of the work:

- **Questions and answers were swapped** for the finance-theory dataset: the parser put the answer in the user turn and left the assistant turn empty.
- **Leaked Llama-2 tokens:** `</s>` was left inside training targets, so the fine-tuned model printed a literal `</s>`.
- **Truncation cut off answers:** long 10-K contexts were truncated from the end, removing the very text the model was supposed to learn. v2 drops over-length examples instead.
- **Loss was dominated by the prompt:** v2 masks system and user tokens so only the analyst's answer is learned.
- **FinQA had no reasoning targets:** v2 turns the gold program (`subtract(510, 450), divide(#0, 450)`) into numbered calculation steps, and keeps dev/test out of training.
- **Duplicate BOS / hand-built prompts:** training and inference now share the tokenizer's chat template.
- **Eyeball evaluation:** three hand-picked prompts were replaced by a scored benchmark.

## Limitations

- Policy-Sentinel is rule-based (regex), not a learned classifier: it is transparent and fast, but paraphrased advice can slip through.
- The unit check does not verify that a number is *correct* or present in the source filing; grounding against the provided context is the next planned check.
- 3B parameters limit multi-step reasoning on long tables; the model is an analyst's assistant, not a replacement.

## Training data

1. [Financial Q&A-10k](https://www.kaggle.com/datasets/yousefsaeedian/financial-q-and-a-10k): questions grounded in 10-K passages.
2. [Finance-Llama2-1k](https://www.kaggle.com/datasets/yousefsaeedian/finance-llama2-1k): finance concepts (YTM, D/E, hedging).
3. [FinQA](https://www.kaggle.com/datasets/visalakshiiyer/question-answering-financial-data): numerical reasoning over filing tables, with gold calculation programs.

## Quick start

Requires Python 3.10–3.12 (prebuilt CPU wheels for `llama-cpp-python`) and Node 20+ for the UI.

```bash
git clone https://github.com/Prakruthi19/fintune-sentinel.git
cd fintune-sentinel
pip install -r requirements.txt
(cd web && npm ci && npm run build)   # React UI -> web/dist, served by app.py
python app.py                         # downloads the GGUF (~2 GB); open http://localhost:7860
```

UI development with hot reload: keep `python app.py` running and start `cd web && npm run dev` (http://localhost:5173, proxies `/api` to the server).

Docker:

```bash
docker build -t sentinel . && docker run -p 8080:8080 sentinel
```

Other entry points:

```bash
python -m pytest -q                                   # unit tests (no model needed)
python scripts/quant_demo.py                          # quantization walkthrough
python -m scripts.evaluate --finqa data/test.json --n 100 --model v1=models/sentinel-v1.Q4_K_M.gguf
python -m scripts.train_sft --load-in-4bit --out runs/v2   # GPU; see notebooks/train_v2_kaggle.ipynb
```
