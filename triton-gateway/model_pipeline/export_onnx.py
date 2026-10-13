"""Export a Hugging Face transformer to ONNX with dynamic batch/sequence axes.

Mirrors the production pipeline used to feed NVIDIA Triton's ONNX Runtime
backend: dynamic axes avoid static-shape reallocation so Triton can pack
requests of arbitrary length into unified GPU batches (see docs/TRITON_TUNING.md).

Usage:
    python export_onnx.py --model-id BAAI/bge-small-en-v1.5 \
        --output model_repository/text_embeddings/1/model.onnx --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def build_dummy_inputs(tokenizer, device: str) -> dict[str, torch.Tensor]:
    """Tokenize a representative sentence and move tensors to ``device``."""
    enc = tokenizer(
        "Sample text for export covering a typical query length.",
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def DYNAMIC_AXES() -> dict:  # noqa: N802 - kept simple/testable
    """Dynamic axes: dim0 = batch, dim1 = sequence for every tensor."""
    return {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "sentence_embedding": {0: "batch_size"},
    }


def export_model(
    model_id: str,
    output_path: Path,
    device: str = "cpu",
    opset_version: int = 17,
) -> Path:
    """Load ``model_id``, trace it and write an ONNX graph to ``output_path``."""
    from transformers import AutoModel, AutoTokenizer  # heavy import, local on purpose

    if device == "cuda" and not torch.cuda.is_available():
        print("[export_onnx] CUDA requested but unavailable; falling back to CPU")
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()
    if device == "cuda":
        model = model.cuda()

    dummy = build_dummy_inputs(tokenizer, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["sentence_embedding"],
            dynamic_axes=DYNAMIC_AXES(),
            opset_version=opset_version,
            do_constant_folding=True,
        )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_repository/text_embeddings/1/model.onnx"),
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    path = export_model(args.model_id, args.output, args.device, args.opset)
    size_mb = path.stat().st_size / 1e6
    print(f"[export_onnx] wrote {path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
