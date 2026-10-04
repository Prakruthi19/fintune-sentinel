# ADR 001: How the stage 1 baseline is measured

Status: accepted. Date: 2026-10-04.

## Question this stage answers

Before fine-tuning anything: how well do current small open models already
solve FinQA questions when they must use tools for the arithmetic, and
*where* do they fail? Fine-tuning is only justified for a failure that
prompting does not fix.

## Decisions

**1. The model never does arithmetic.** Tools: `calculate` (add, subtract,
multiply, divide, exp, greater), `table_aggregate` (sum/average/max/min of a
table row), `final_answer`. Small models are bad at exact arithmetic; that
weakness is removed by design so the eval measures understanding (which
numbers, which operations), not mental math.

**2. No cell-lookup tool yet.** Copying a number from the prompt is a real
failure mode. Stage 1 measures how often it happens (`wrong_numbers`). A
lookup tool is added only if that category dominates. Adding it now would
hide the very error we want to see.

**3. Tools are validated against FinQA itself.** Every gold program is run
through the same tool code the model uses. Result: 1147/1147 test and
883/883 dev programs reproduce FinQA's answer; 2/6251 train programs do not
(duplicate row labels) and are excluded from few-shot selection. Two parsing
rules were needed: percent literals in programs mean decimals ("8.75%" =
0.0875), and table cells carry a repeated value in brackets
("$ -61.1 ( 61.1 )"). Exclusions are written into every run's config.json.

**4. Sample: 200 test items, stratified, fixed seed.** Stratified by number
of program steps so the 57% / 36% / 7% mix of 1-, 2- and 3+-step questions
matches the full split. n=200 keeps free-tier and CPU runs feasible. The
95% Wilson interval at n=200 is about ±7 points, so **differences under ~10
points between runs are not claimed as real**. Larger n is a flag away.

**5. Matching: 0.5% relative tolerance.** FinQA answers are rounded to 5
decimals; models round further when submitting. 0.5% accepts 0.0986 for
0.09864 but rejects 0.10. Answers 100x/1000x off are counted separately as
`scale_error` (percent vs decimal, thousands vs millions) because the fix
differs from a wrong calculation.

**6. Error taxonomy is heuristic.** Each wrong item gets one category by
comparing tool calls with the gold program: `no_tool_use`,
`no_final_answer`, `scale_error`, `wrong_numbers`, `wrong_operations`,
`wrong_arrangement`. It can mislabel individual items (e.g. a valid
alternative solution path). It is for spotting *patterns* across 200 items,
and a sample of labels should be checked by hand before acting on it.

**7. One model interface, OpenAI-compatible.** Ollama, llama.cpp server,
vLLM and most hosted APIs expose the same chat-completions API with tools,
so one adapter covers laptop CPU, Kaggle GPU and API baselines. The eval
depends only on a small `ChatModel` protocol, which is why it is fully
tested with fakes and no network.

**8. Few-shot examples come from the train split** (k items spanning 1-,
2- and 3+-step programs, fixed seed, ids recorded). They use the exact
conversation format that stage 2 SFT data will use, so prompting and
fine-tuning are compared on identical structure.

## Not decided here

Which models to run is left to the runbook, because the best current small
models change monthly. Whatever is run, its exact tag, quantization,
hardware and whether "thinking" mode was on must be recorded with the run.

## Relation to `scripts/evaluate.py` (v2)

`scripts/evaluate.py` measures the other design: the model answers in free
text and the last number is parsed. Keeping both gives the "tools vs direct
answer" comparison. Two differences must be resolved before their numbers
are put side by side:
- evaluate.py accepts 13.3 for a gold of 0.133 (either scale counts as
  correct); finbench counts that as `scale_error`. Same data, different
  accuracy.
- evaluate.py uses a 1% tolerance and the first n test items; finbench uses
  0.5% and a stratified sample.
The plan is to report evaluate.py results through finbench's scoring and
sample, so both modes are judged by one rule.

## Rejected

- *LLM-as-judge scoring*: FinQA answers are numbers; exact checking is
  cheaper, deterministic and not biased toward any model family.
- *Free-text answers parsed with regex*: hides format failures, which are
  exactly what tool-use fine-tuning would fix.
