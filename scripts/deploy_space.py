"""Publishes the app (FastAPI + React UI, built by the Dockerfile) to a free CPU Hugging Face Space.

    huggingface-cli login            # token with write access
    python -m scripts.deploy_space --space Prakruthi/FinTune-Sentinel

To serve a different GGUF (e.g. v2), pass --model-repo / --model-file; they become Space variables,
which Docker Spaces pass to the image build as build args.
"""
import argparse

from huggingface_hub import HfApi

SPACE_README = """---
title: FinTune-Sentinel
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8080
pinned: false
short_description: Free financial health check for small businesses
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
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    for key, value in (("MODEL_REPO", args.model_repo), ("MODEL_FILE", args.model_file)):
        if value:
            api.add_space_variable(args.space, key, value)

    api.upload_file(path_or_fileobj=SPACE_README.encode(), path_in_repo="README.md",
                    repo_id=args.space, repo_type="space")
    for path in ("Dockerfile", ".dockerignore", "app.py", "requirements.txt"):
        api.upload_file(path_or_fileobj=path, path_in_repo=path, repo_id=args.space, repo_type="space")
    for folder in ("sentinel", "web"):
        api.upload_folder(folder_path=folder, path_in_repo=folder, repo_id=args.space, repo_type="space",
                          ignore_patterns=["__pycache__/*", "node_modules/*", "dist/*"])
    print(f"Deployed: https://huggingface.co/spaces/{args.space} (first build takes ~5-10 min)")


if __name__ == "__main__":
    main()
