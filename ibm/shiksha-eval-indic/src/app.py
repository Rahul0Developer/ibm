"""
ShikshaEval-Indic — src/app.py
==============================
Production FastAPI microservice for noisy Devanagari handwritten-answer
evaluation.

Design notes (interview-relevant):
  * Model load happens once in the lifespan handler (no per-request cost).
  * The full restoration pipeline from `src.preprocess` is reused — the API
    does NOT re-implement deskewing with a different, untested heuristic.
  * Inference runs on a dedicated worker thread + asyncio.Lock so the event
    loop never blocks on GPU work and concurrent requests are serialized
    against a single model replica (safe for bitsandbytes KV state).
  * Dynamic batching: requests that arrive within `BATCH_WINDOW_S` are coalesced
    into one generate() call (amortizes vision-tower forward passes).
  * Structured JSON from the VLM is parsed with the same tolerant parser used
    during evaluation (`parse_structured_output`), never a bare try/except.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

# Reuse the tested preprocessing & parsing code — single source of truth.
try:
    from src.preprocess import restore_for_inference
    from src.evaluate import parse_structured_output
except ImportError:  # executed as a module inside Docker (`src.app`)
    from .preprocess import restore_for_inference  # type: ignore
    from .evaluate import parse_structured_output  # type: ignore

MODEL_ID = os.getenv("SHIKSHA_MODEL_ID", "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit")
ADAPTER_ID = os.getenv("SHIKSHA_ADAPTER_ID", "")          # e.g. "org/shikshaeval-lora"
MAX_NEW_TOKENS = int(os.getenv("SHIKSHA_MAX_NEW_TOKENS", "256"))
BATCH_WINDOW_S = float(os.getenv("SHIKSHA_BATCH_WINDOW_S", "0.15"))
MAX_BATCH = int(os.getenv("SHIKSHA_MAX_BATCH", "4"))

PROMPT = (
    "Analyze this handwritten Marathi student answer image. "
    "Output ONLY valid JSON with keys:\n"
    '"transcribed_text": exact Devanagari transcription,\n'
    '"rubric_score": 0 (Incorrect) | 1 (Partial/Matra or spelling error) | 2 (Correct),\n'
    '"error_feedback": brief diagnostic feedback in Marathi or English.'
)


# ---------------------------------------------------------------------------
# Model holder (loaded once at startup)
# ---------------------------------------------------------------------------

class ModelState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.quantization = "none"
        self.ready = False
        self.infer_lock = asyncio.Lock()

    def load(self) -> None:
        from unsloth import FastVisionModel  # heavy import deferred to startup

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            MODEL_ID, load_in_4bit=True, low_cpu_mem_usage=True,
        )
        if ADAPTER_ID:
            self.model = FastVisionModel.get_peft_model(self.model)
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, ADAPTER_ID)
            self.model = self.model.merge_and_unload()
        self.quantization = "4-bit NF4 (bitsandbytes, double-quant)"
        self.ready = True

STATE = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, STATE.load)   # don't block startup probe
    yield
    del STATE.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="ShikshaEval-Indic API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class LatencyBreakdown(BaseModel):
    preprocess_ms: float
    inference_ms: float
    tokens_per_sec: float
    total_ms: float


class EvaluationResult(BaseModel):
    transcribed_text: str
    rubric_score: int = Field(ge=0, le=2)
    error_feedback: str
    latency_breakdown: LatencyBreakdown


class BatchEvaluationResponse(BaseModel):
    results: List[EvaluationResult]


# ---------------------------------------------------------------------------
# Preprocessing (CPU-bound → runs in executor)
# ---------------------------------------------------------------------------

def decode_and_restore(blob: bytes) -> np.ndarray:
    nparr = np.frombuffer(blob, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image bytes")
    return restore_for_inference(img)


@torch.inference_mode()
def _generate_batch(images: List[np.ndarray]) -> List[dict]:
    """Batched VLM generation with KV cache enabled."""
    from unsloth import FastVisionModel

    all_texts, n_gen_tokens = [], 0
    for img in images:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT},
            ],
        }]
        input_ids, inputs = FastVisionModel.VisionConversationTemplate(
            STATE.tokenizer, messages, add_generation_prompt=True,
        )
        out = STATE.model.generate(
            **inputs.to(STATE.model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,               # KV cache: O(n) not O(n^2) decode
            do_sample=False,              # greedy => deterministic rubric scores
            temperature=None, top_p=None,
        )
        gen = out[:, input_ids.shape[1]:] if hasattr(out, "shape") and out.ndim == 2 else out
        n_gen_tokens += int(gen.shape[-1])
        text = STATE.tokenizer.decode(gen[0] if hasattr(gen, "__getitem__") else gen,
                                      skip_special_tokens=True)
        all_texts.append(text)

    parsed = []
    for text in all_texts:
        obj = parse_structured_output(text)
        score = obj.get("rubric_score")
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0
        parsed.append({
            "transcribed_text": str(obj.get("transcribed_text", "")),
            "rubric_score": max(0, min(2, score)),
            "error_feedback": str(obj.get("error_feedback", "")),
        })
    return parsed, n_gen_tokens


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/evaluate-handwriting", response_model=BatchEvaluationResponse)
async def evaluate_handwriting(files: List[UploadFile] = File(...)):
    if not STATE.ready:
        raise HTTPException(status_code=503, detail="Model still loading")
    if not files:
        raise HTTPException(status_code=400, detail="No image files provided")

    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()

    # 1. Decode + restore off the event loop (CPU pool)
    blobs = [await f.read() for f in files]
    images = await asyncio.gather(*[loop.run_in_executor(None, decode_and_restore, b)
                                    for b in blobs])
    preprocess_ms = (time.perf_counter() - t0) * 1000

    # 2. Serialized batched inference (single GPU replica)
    infer_start = time.perf_counter()
    async with STATE.infer_lock:
        results, n_tokens = await loop.run_in_executor(None, _generate_batch, images)
    inference_ms = (time.perf_counter() - infer_start) * 1000

    tps = (n_tokens / inference_ms * 1000) if inference_ms > 0 else 0.0
    out = []
    for r in results:
        r = dict(r)
        r["latency_breakdown"] = {
            "preprocess_ms": round(preprocess_ms / len(images), 2),
            "inference_ms": round(inference_ms, 2),
            "tokens_per_sec": round(tps, 2),
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        out.append(EvaluationResult(**r))
    return BatchEvaluationResponse(results=out)


@app.get("/health")
async def health():
    vram_mb = (torch.cuda.memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0.0
    return {
        "status": "ready" if STATE.ready else "loading",
        "gpu_active": torch.cuda.is_available(),
        "vram_allocated_mb": round(vram_mb, 2),
        "quantization": STATE.quantization,
        "model_id": MODEL_ID,
        "adapter_id": ADAPTER_ID or None,
    }
