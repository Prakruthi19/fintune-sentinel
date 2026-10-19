# 1. Build the React UI
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# 2. Python server: FastAPI + llama.cpp on CPU
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the 2 GB GGUF into the image so Cloud Run cold starts don't re-download it.
ARG MODEL_REPO=Prakruthi/FinTune-Sentinel-v2-GGUF
ARG MODEL_FILE=Llama-3.2-3B-Instruct.Q4_K_M.gguf
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('${MODEL_REPO}', '${MODEL_FILE}', local_dir='/models')"
ENV MODEL_PATH=/models/${MODEL_FILE}

COPY sentinel/ sentinel/
COPY app.py .
COPY --from=web /web/dist web/dist

ENV PORT=8080
EXPOSE 8080

CMD ["python", "app.py"]
