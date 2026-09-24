"""Converts the three source datasets into chat-message examples.

Every example is a list of {"role", "content"} dicts. Rendering to text is left to
tokenizer.apply_chat_template so training and llama.cpp inference share one template
(no hand-written special tokens, no duplicate <|begin_of_text|>).
"""
import re

SYSTEM_PROMPT = (
    "You are FinTune-Sentinel, a senior financial analyst auditing SEC filings. "
    "Answer only from the provided context when one is given, show calculations step by step, "
    "always state units ($, %, millions), and never give buy/sell/hold recommendations."
)

LLAMA2_TOKENS = re.compile(r"</?s>|<<SYS>>.*?<</SYS>>", re.DOTALL)
LLAMA2_TURN = re.compile(r"\[INST\](?P<q>.*?)\[/INST\](?P<a>.*)", re.DOTALL)


def _chat(user: str, assistant: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]


def from_10k(row: dict):
    if not (row.get("question") and row.get("answer") and row.get("context")):
        return None
    return _chat(f"Context:\n{row['context']}\n\nQuestion: {row['question']}", row["answer"])


def from_llama2(text: str):
    """Parses '<s>[INST] question [/INST] answer </s>' and strips Llama-2 control tokens.

    The original notebook split on '[/INST]' and took parts[1] as the *instruction*, which
    puts the answer in the user turn and leaves the assistant turn empty, and it left '</s>'
    in the targets (the fine-tuned model now emits a literal '</s>').
    """
    m = LLAMA2_TURN.search(text or "")
    if not m:
        return None
    q = LLAMA2_TOKENS.sub("", m.group("q")).strip()
    a = LLAMA2_TOKENS.sub("", m.group("a")).strip()
    if not q or not a:
        return None
    return _chat(q, a)


def table_to_markdown(table: list) -> str:
    if not table:
        return ""
    width = max(len(r) for r in table)
    rows = [[str(c).strip() for c in r] + [""] * (width - len(r)) for r in table]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _fmt_num(x) -> str:
    try:
        return f"{float(x):,.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(x)


def finqa_reasoning(qa: dict) -> str:
    """Turns FinQA's gold program (e.g. subtract(510, 450), divide(#0, 450)) into steps."""
    steps = qa.get("steps") or []
    lines = []
    for i, s in enumerate(steps, 1):
        op = re.sub(r"[\d-]+$", "", s.get("op", ""))  # FinQA ops look like "minus2-1"
        lines.append(f"Step {i}: {op}({s.get('arg1')}, {s.get('arg2')}) = {_fmt_num(s.get('res'))}")
    if not lines and qa.get("program"):
        lines.append(f"Program: {qa['program']}")
    lines.append(f"Answer: {qa['answer']}")
    return "\n".join(lines)


def from_finqa(entry: dict):
    qa = entry.get("qa") or {}
    if not (qa.get("question") and qa.get("answer") not in (None, "")):
        return None
    context = "\n".join(
        part for part in (
            " ".join(entry.get("pre_text", [])),
            table_to_markdown(entry.get("table", [])),
            " ".join(entry.get("post_text", [])),
        ) if part
    )
    return _chat(f"Context:\n{context}\n\nQuestion: {qa['question']}", finqa_reasoning(qa))
