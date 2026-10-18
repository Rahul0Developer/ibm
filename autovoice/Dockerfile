# Placeholder API image. PHASE 8 replaces this with the final multi-stage build:
#   python:3.11-slim base, CPU-only torch, non-root user, healthcheck, PORT honored.
FROM python:3.11-slim AS base
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY . .
EXPOSE 8080
CMD ["echo", "AutoVoice API Dockerfile finalized in Phase 8"]
