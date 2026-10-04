import os

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from sentinel.data import SYSTEM_PROMPT
from sentinel.governance import audit

# 1. Configuration (override via env vars on Cloud Run / HF Spaces)
REPO_ID = os.environ.get("MODEL_REPO", "Prakruthi/FinTune-Sentinel-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "Llama-3.2-3B-Instruct.Q4_K_M.gguf")
N_CTX = int(os.environ.get("N_CTX", "4096"))
N_THREADS = int(os.environ.get("N_THREADS", str(os.cpu_count() or 2)))

# 2. Load model (CPU). MODEL_PATH lets a container use a baked-in or mounted file.
model_path = os.environ.get("MODEL_PATH") or hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
llm = Llama(model_path=model_path, n_ctx=N_CTX, n_threads=N_THREADS, n_gpu_layers=0, verbose=False)
print(f"Model loaded: {model_path} (n_ctx={N_CTX}, threads={N_THREADS})")


def _to_messages(history):
    """Accepts both Gradio history formats: openai-style dicts or (user, bot) tuples."""
    messages = []
    for item in history or []:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            user, bot = item
            messages += [{"role": "user", "content": user}, {"role": "assistant", "content": bot}]
    return messages


def predict(message, history):
    # create_chat_completion applies the GGUF's own Llama-3 chat template, so there is no
    # hand-built prompt, no duplicate BOS, and generation stops on <|eot_id|> only.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *_to_messages(history)[-6:],
                {"role": "user", "content": message}]
    out = llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.1)
    result = audit(out["choices"][0]["message"]["content"].strip())
    footer = f"\n\n---\nPolicy-Sentinel: **{result.status}**"
    if result.findings:
        footer += " - " + "; ".join(result.findings)
    return result.text + footer


# 3. Launch UI (0.0.0.0 + $PORT so it is reachable inside Docker / Cloud Run)
gr.ChatInterface(
    predict,
    title="FinTune-Sentinel (CPU Optimized)",
    description="Financial analysis on a 4-bit GGUF model, with every answer audited by Policy-Sentinel.",
).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
