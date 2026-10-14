# ShikshaEval-Indic: multi-stage, GPU-ready production image
# Target platform: AWS ECS/Fargate with g5/g6 GPU task definitions.
#
#   docker build -t shikshaeval-api:1.0.0 .
#   docker run --gpus all -p 8000:8000 \
#       -e SHIKSHA_MODEL_ID=unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit \
#       -e HF_TOKEN=$HF_TOKEN \
#       -v hf_cache:/root/.cache/huggingface shikshaeval-api:1.0.0

# ---------------- Stage 1: builder ----------------
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-dev python3-venv build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# Install into an isolated venv so runtime copy is atomic & lean.
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# ---------------- Stage 2: runtime ----------------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    HF_HOME=/models/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 libgomp1 libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/ ./src/
# Pre-bake weights at build time (or mount the hf_cache volume at run time):
#   RUN /opt/venv/bin/python -c "from huggingface_hub import snapshot_download; \
#       snapshot_download('unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit')"

RUN mkdir -p /models/cache && chown -R appuser:appuser /models /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Single GPU worker: uvicorn workers>1 would duplicate the 2.2GB replica.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
