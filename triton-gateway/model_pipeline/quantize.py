"""FP16 post-export conversion of an ONNX graph.

Halves memory-bandwidth pressure and unlocks tensor-core throughput on T4/A10G,
which is what lets Triton hold the p99 < 30 ms budget at 3k+ QPS per GPU node.

Usage:
    python quantize.py --input model.onnx --output model_fp16.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def convert_float32_to_float16(input_path: Path, output_path: Path) -> Path:
    """Cast the whole graph to float16 via onnxconverter_common."""
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(input_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    # Triton expects FP16 embeddings declared in config.pbtxt; strip redundant
    # doc string so graphs stay deterministic across conversions.
    model_fp16.doc_string = ""
    onnx.save(model_fp16, str(output_path))
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("model_repository/text_embeddings/1/model.onnx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_repository/text_embeddings/1/model_fp16.onnx"),
    )
    args = parser.parse_args(argv)

    out = convert_float32_to_float16(args.input, args.output)
    before, after = args.input.stat().st_size, out.stat().st_size
    print(f"[quantize] {before/1e6:.1f} MB -> {after/1e6:.1f} MB "
          f"({100*(1-after/before):.0f}% reduction)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
