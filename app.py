"""FinTune-Sentinel server: a streaming chat API over the GGUF model, plus the React UI in web/dist.

    python app.py                       # http://localhost:7860 (UI + API)
    cd web && npm run dev               # UI dev server on :5173, proxies /api here
"""
import json
import os
import threading
import time

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sentinel.data import SYSTEM_PROMPT
from sentinel.documents import DocumentError, extract_text
from sentinel.governance import TRADE_ADVICE, audit
from sentinel.health import EXPLAIN_INSTRUCTIONS, compute, explain_prompt, parse_statement, report

# 1. Configuration (override via env vars on Cloud Run / HF Spaces)
REPO_ID = os.environ.get("MODEL_REPO", "Prakruthi/FinTune-Sentinel-v2-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "Llama-3.2-3B-Instruct.Q4_K_M.gguf")
N_CTX = int(os.environ.get("N_CTX", "4096"))
# More threads is slower past ~4: on a 2P+8E laptop, 12 threads generated at 1.2 tok/s vs 2.9 with 4.
N_THREADS = int(os.environ.get("N_THREADS", str(min(4, os.cpu_count() or 2))))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
EXPLAIN_TOKENS = 260  # a summary plus three questions; caps a rambling answer on a slow CPU
WEB_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "dist")


def load_llm():
    """CPU model. MODEL_PATH lets a container use a baked-in or mounted file."""
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    path = os.environ.get("MODEL_PATH") or hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
    llm = Llama(model_path=path, n_ctx=N_CTX, n_threads=N_THREADS, n_gpu_layers=0, verbose=False)
    print(f"Model loaded: {path} (n_ctx={N_CTX}, threads={N_THREADS})")
    # Warm-up: pages the weights in and leaves the health-check instructions in the KV cache, so the
    # first visitor's summary starts in seconds instead of re-reading the whole prompt.
    llm.create_chat_completion(messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                         {"role": "user", "content": EXPLAIN_INSTRUCTIONS}], max_tokens=1)
    return llm, os.path.basename(path)


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Turn]


class ParseRequest(BaseModel):
    text: str


class HealthRequest(BaseModel):
    fields: dict[str, float | None]


def _event(name, data):
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def create_app(llm, model_name):
    app = FastAPI(title="FinTune-Sentinel")
    lock = threading.Lock()  # one llama.cpp context: generate one answer at a time
    explained = {}  # same figures -> same explanation (temperature is near 0), so repeats are instant

    @app.get("/api/info")
    def info():
        return {"model": model_name, "threads": N_THREADS, "context": N_CTX, "max_tokens": MAX_TOKENS}

    @app.post("/api/parse")
    def parse(req: ParseRequest):
        return {"fields": parse_statement(req.text)}

    @app.post("/api/upload")
    async def upload(file: UploadFile):
        try:
            text = extract_text(file.filename, await file.read())
        except DocumentError as err:
            raise HTTPException(422, str(err))
        return {"fields": parse_statement(text), "filename": file.filename}

    @app.post("/api/health")
    def health(req: HealthRequest):
        return report(req.fields)

    @app.post("/api/explain")
    def explain(req: HealthRequest):
        metrics = compute(req.fields)
        if not metrics:
            raise HTTPException(400, "Enter at least revenue and one other figure.")
        key = tuple(sorted((k, v) for k, v in req.fields.items() if v is not None))
        return _sse([{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": explain_prompt(metrics)}], EXPLAIN_TOKENS, key)

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        turns = [t.model_dump() for t in req.messages if t.role in ("user", "assistant")]
        return _sse([{"role": "system", "content": SYSTEM_PROMPT}, *turns[-7:]])

    def _sse(messages, max_tokens=MAX_TOKENS, cache_key=None):
        # create_chat_completion applies the GGUF's own Llama-3 chat template, so there is no
        # hand-built prompt, no duplicate BOS, and generation stops on <|eot_id|> only.
        def stream():
            if cache_key in explained:
                done = explained[cache_key]
                yield _event("token", {k: done[k] for k in ("text", "tokens", "elapsed", "first")})
                yield _event("done", done)
                return
            with lock:
                start, first, tokens, text = time.perf_counter(), None, 0, ""
                for chunk in llm.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                                        temperature=0.1, stream=True):
                    delta = chunk["choices"][0]["delta"].get("content")
                    if not delta:
                        continue
                    first = first or time.perf_counter() - start
                    tokens += 1
                    text += delta
                    # Redact while streaming so advice is never on screen, even before the final audit.
                    yield _event("token", {"text": TRADE_ADVICE.sub("[redacted]", text), "tokens": tokens,
                                           "elapsed": time.perf_counter() - start, "first": first})
                result = audit(text.strip())
                done = {"text": result.text, "status": result.status, "findings": result.findings,
                        "tokens": tokens, "elapsed": time.perf_counter() - start, "first": first}
                if cache_key is not None and tokens:
                    if len(explained) >= 256:
                        explained.pop(next(iter(explained)))
                    explained[cache_key] = done
                yield _event("done", done)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if os.path.isdir(WEB_DIST):
        app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIST, "assets")), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            file = os.path.realpath(os.path.join(WEB_DIST, path))
            inside = file.startswith(os.path.realpath(WEB_DIST) + os.sep)
            return FileResponse(file if inside and os.path.isfile(file) else os.path.join(WEB_DIST, "index.html"))

    return app


# 2. Launch (0.0.0.0 + $PORT so it is reachable inside Docker / Cloud Run)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(*load_llm()), host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
