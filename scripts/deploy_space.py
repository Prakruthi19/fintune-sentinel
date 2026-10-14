"""Publishes the Gradio app to a free CPU Hugging Face Space.

    huggingface-cli login            # token with write access
    python -m scripts.deploy_space --space Prakruthi/FinTune-Sentinel

To serve a different GGUF (e.g. v2), pass --model-repo / --model-file; they become Space variables.
"""
import argparse
import os
import tempfile

from huggingface_hub import HfApi

SPACE_README = """---
title: FinTune-Sentinel
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
python_version: "3.10"
app_file: app.py
pinned: false
short_description: 3B financial analyst LLM on CPU with a compliance auditor
---

Source: https://github.com/Prakruthi19/fintune-sentinel
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--space", required=True, help="e.g. Prakruthi/FinTune-Sentinel")
    p.add_argument("--model-repo", default=None)
    p.add_argument("--model-file", default=None)
    args = p.parse_args()

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    for key, value in (("MODEL_REPO", args.model_repo), ("MODEL_FILE", args.model_file)):
        if value:
            api.add_space_variable(args.space, key, value)

    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "README.md"), "w", encoding="utf-8").write(SPACE_README)
        # The Space pins Gradio via sdk_version, so drop it from requirements to avoid a conflict.
        reqs = [l for l in open("requirements.txt").read().splitlines() if not l.startswith("gradio")]
        open(os.path.join(tmp, "requirements.txt"), "w").write("\n".join(reqs) + "\n")
        for path in ("README.md", "requirements.txt"):
            api.upload_file(path_or_fileobj=os.path.join(tmp, path), path_in_repo=path,
                            repo_id=args.space, repo_type="space")
    api.upload_file(path_or_fileobj="app.py", path_in_repo="app.py", repo_id=args.space, repo_type="space")
    api.upload_folder(folder_path="sentinel", path_in_repo="sentinel", repo_id=args.space, repo_type="space",
                      ignore_patterns=["__pycache__/*"])
    print(f"Deployed: https://huggingface.co/spaces/{args.space} (first build takes ~5-10 min)")


if __name__ == "__main__":
    main()
