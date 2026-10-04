"""QLoRA fine-tune of Llama-3.2-3B-Instruct for FinTune-Sentinel (v2).

Built for a single GCP GPU VM (L4 24 GB / A100). Run from the repo root:
    python -m scripts.train_sft --data-dir data --out runs/v2

Fixes over the notebook run: FinQA from train.json (dev/test kept for evaluation), tables
rendered as markdown, gold calculation steps as targets, Llama-2 tokens stripped, one chat
template for all data, loss only on assistant tokens, over-length examples dropped instead
of truncated (truncation cut off the answers), and a held-out eval split.
"""
import argparse
import json
import os

import pandas as pd
from unsloth import FastLanguageModel, is_bfloat16_supported  # must be imported before trl
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset, concatenate_datasets
from trl import SFTConfig, SFTTrainer

from sentinel.data import from_10k, from_finqa, from_llama2


def build_dataset(data_dir, tokenizer, max_len, seed, max_samples=None):
    convs = {
        "10k": [from_10k(r) for r in pd.read_csv(os.path.join(data_dir, "Financial-QA-10k.csv")).to_dict("records")],
        "theory": [from_llama2(t) for t in pd.read_csv(os.path.join(data_dir, "finance-llama2-1k.csv"))["text"]],
        "finqa": [from_finqa(e) for e in json.load(open(os.path.join(data_dir, "train.json")))],
    }
    parts = []
    for source, rows in convs.items():
        rows = [r for r in rows if r][:max_samples]
        # The trainer's tokenizer adds BOS itself; strip the template's copy to avoid a double BOS.
        texts = [tokenizer.apply_chat_template(r, tokenize=False).removeprefix(tokenizer.bos_token) for r in rows]
        lengths = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts]
        keep = [t for t, n in zip(texts, lengths) if n <= max_len]
        print(f"{source:>7}: {len(rows)} parsed, {len(keep)} kept (<= {max_len} tokens)")
        parts.append(Dataset.from_dict({"text": keep, "source": [source] * len(keep)}))
    return concatenate_datasets(parts).shuffle(seed=seed).train_test_split(test_size=0.05, seed=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="runs/v2")
    p.add_argument("--base", default="unsloth/Llama-3.2-3B-Instruct")
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--load-in-4bit", action="store_true", help="QLoRA; default is 16-bit LoRA (fits on an L4)")
    p.add_argument("--batch-size", type=int, default=4, help="use 1-2 on a 16 GB T4")
    p.add_argument("--max-samples", type=int, default=None, help="per source; e.g. 50 for a cheap smoke test")
    p.add_argument("--max-steps", type=int, default=-1, help="e.g. 20 for a smoke test; -1 = full epochs")
    p.add_argument("--skip-gguf", action="store_true", help="skip the slow GGUF export (smoke tests)")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--report-to", default="none", help="e.g. wandb or tensorboard")
    args = p.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=args.max_len, dtype=None, load_in_4bit=args.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )

    ds = build_dataset(args.data_dir, tokenizer, args.max_len, args.seed, args.max_samples)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_len,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=16 // args.batch_size,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.01,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            optim="adamw_8bit",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            output_dir=os.path.join(args.out, "checkpoints"),
            report_to=args.report_to,
            seed=args.seed,
        ),
    )
    # Mask system/user tokens: the long 10-K and FinQA contexts no longer dominate the loss.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    stats = trainer.train()
    metrics = {**stats.metrics, **trainer.evaluate()}
    print(metrics)
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    json.dump(trainer.state.log_history, open(os.path.join(args.out, "log_history.json"), "w"), indent=2)

    model.save_pretrained(os.path.join(args.out, "lora"))
    tokenizer.save_pretrained(os.path.join(args.out, "lora"))
    if not args.skip_gguf:
        # One merge, two quantizations: Q4_K_M to serve, Q8_0 to measure quantization loss against.
        model.save_pretrained_gguf(os.path.join(args.out, "gguf"), tokenizer, quantization_method=["q4_k_m", "q8_0"])


if __name__ == "__main__":
    main()
