"""
ShikshaEval-Indic — src/benchmark_inference.py
==============================================
Latency / throughput / VRAM benchmark across quantization regimes:

    fp16  : AutoModelForVision2Seq in torch.float16 (baseline)
    nf4   : Unsloth QLoRA checkpoint merged, served via bitsandbytes 4-bit NF4
    onnx  : ONNX-Runtime exported *vision encoder* + quantized text decode
            (CPU/CUDA EP; the realistic edge path — full VLM ONNX export of
             KV-cache generation is not yet stable, so we bench the tower and
             extrapolate end-to-end with measured tokens/sec).

Usage:
    python -m src.benchmark_inference --mode nf4 --batch 2 --iters 30
    python -m src.benchmark_inference --mode all --report benchmarks.md

Edge notes (Android): ExecuTorch or ONNX Runtime Mobile with a 4-bit
weight-packed QNN delegate; expect ~2-3x CPU decode latency vs T4 and a
<=2GB memory budget → ship the 0.5B/3B vision tower INT8 + small LM head.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class BenchResult:
    mode: str
    batch_size: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    peak_vram_mb: float
    tokens_per_sec: float
    images_per_sec: float
    model_size_mb: Optional[float] = None


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _peak_vram_mb(reset: bool = False) -> float:
    if not torch.cuda.is_available():
        return 0.0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return peak


def _model_size_mb(model) -> float:
    total = sum(p.numel() for p in model.parameters())
    # bitsandbytes Params4bit store packed uint8 (0.5 bytes/param for NF4)
    bpe = 0.5 if any("Params4bit" in type(p.data).__name__ or
                     getattr(getattr(p, "data", None), "quant_state", None) is not None
                     for p in model.parameters()) else 2.0
    return total * bpe / 1024 ** 2


def load_model(mode: str, model_id: str):
    """Return (model, processor_or_tokenizer, generate_fn)."""
    if mode == "nf4":
        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            model_id, load_in_4bit=True)
        return model, tokenizer, "unsloth"
    if mode == "fp16":
        from transformers import AutoProcessor, AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor, "hf"
    if mode == "onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(
            f"exported/vision_encoder.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        return sess, None, "onnx"
    raise ValueError(f"unknown mode {mode}")


def bench_fp_or_nf4(mode: str, model_id: str, batch_size: int, iters: int,
                    max_new_tokens: int = 64) -> BenchResult:
    model, tok, backend = load_model(mode, model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_px = torch.randn(batch_size, 3, 512, 512, device=device,
                           dtype=torch.float16)
    input_ids = torch.ones(batch_size, 16, dtype=torch.long, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    latencies, gen_tokens = [], []
    for i in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids, pixel_values=dummy_px,
                max_new_tokens=max_new_tokens, use_cache=True, do_sample=False)
        _sync(device)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)
        gen_tokens.append(max(out.shape[-1] - input_ids.shape[-1], 1))
        # warmup excluded from stats below
    latencies = np.asarray(latencies[max(1, iters // 10):])  # drop first 10% warmup
    gen_tokens = np.asarray(gen_tokens[max(1, iters // 10):])

    p50, p95 = float(np.percentile(latencies, 50)), float(np.percentile(latencies, 95))
    tps = float(gen_tokens.mean() / (p50 / 1000.0))
    return BenchResult(
        mode=mode, batch_size=batch_size, p50_ms=round(p50, 1),
        p95_ms=round(p95, 1), mean_ms=round(float(latencies.mean()), 1),
        peak_vram_mb=round(_peak_vram_mb(), 1), tokens_per_sec=round(tps, 1),
        images_per_sec=round(batch_size * 1000.0 / p50, 2),
        model_size_mb=round(_model_size_mb(model), 1))


def bench_onnx(iters: int, batch_size: int) -> BenchResult:
    """Vision-encoder-only ONNX bench (fixed-shape 512x512 patchified input)."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession("exported/vision_encoder.onnx", opts,
                                providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    dummy = np.random.randn(batch_size, 3, 512, 512).astype(np.float32)
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: dummy})
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.asarray(lat)
    p50, p95 = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))
    size_mb = __import__("os").path.getsize("exported/vision_encoder.onnx") / 1024**2
    return BenchResult(mode="onnx(vision-tower)", batch_size=batch_size,
                       p50_ms=round(p50, 1), p95_ms=round(p95, 1),
                       mean_ms=round(float(lat.mean()), 1), peak_vram_mb=0.0,
                       tokens_per_sec=0.0,
                       images_per_sec=round(batch_size * 1000.0 / p50, 2),
                       model_size_mb=round(size_mb, 1))


def markdown_table(results: list[BenchResult]) -> str:
    cols = ["mode", "batch_size", "model_size_mb", "p50_ms", "p95_ms",
            "peak_vram_mb", "tokens_per_sec", "images_per_sec"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in results:
        d = asdict(r)
        lines.append("| " + " | ".join(str(d[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit")
    ap.add_argument("--mode", choices=["fp16", "nf4", "onnx", "all"], required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--report", default="", help="write markdown table to this file")
    args = ap.parse_args()

    results: list[BenchResult] = []
    modes = ["fp16", "nf4", "onnx"] if args.mode == "all" else [args.mode]
    for m in modes:
        try:
            if m == "onnx":
                results.append(bench_onnx(args.iters, args.batch))
            else:
                mid = args.model if m == "nf4" else "Qwen/Qwen2.5-VL-3B-Instruct"
                results.append(bench_fp_or_nf4(m, mid, args.batch, args.iters))
        except FileNotFoundError as e:
            print(f"[skip] {m}: {e} (run export script first)")
        except Exception as e:
            print(f"[fail] {m}: {type(e).__name__}: {e}")

    table = markdown_table(results)
    print(table)
    for r in results:
        print(json.dumps(asdict(r)))
    if args.report:
        with open(args.report, "w") as fh:
            fh.write("# Inference Benchmark\n\n" + table + "\n")
