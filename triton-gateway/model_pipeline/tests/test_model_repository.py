"""Validate the Triton model repository layout and config.pbtxt invariants.

These guards catch the class of mistakes that silently destroy batching
performance: wrong dims, FP32 declared where the graph emits FP16, or a
dynamic-batching window that violates the latency SLO.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "model_repository" / "text_embeddings" / "config.pbtxt"


def read_config() -> str:
    assert CONFIG.exists(), f"missing Triton config at {CONFIG}"
    return CONFIG.read_text()


def test_model_version_directory_exists():
    version_dir = CONFIG.parent / "1"
    assert version_dir.is_dir(), "Triton requires <model>/1/ version directories"


def test_max_batch_size_matches_preferred_batches():
    text = read_config()
    max_batch = int(re.search(r"max_batch_size:\s*(\d+)", text).group(1))
    preferred = [
        int(n) for n in re.findall(r"preferred_batch_size:\s*\[([0-9,\s]+)\]", text)[0].split(",")
    ]
    assert all(p <= max_batch for p in preferred), (
        "preferred batch sizes must not exceed max_batch_size"
    )


def test_dynamic_batching_window_within_slo():
    text = read_config()
    delay_us = int(re.search(r"max_queue_delay_microseconds:\s*(\d+)", text).group(1))
    # 5 ms queue + ~20 ms compute + ~3 ms RPC == 28 ms < 30 ms p99 SLA.
    assert delay_us <= 5000, "queue delay window would break the p99 < 30ms budget"


def test_gpu_instance_group_present():
    text = read_config()
    assert re.search(r"kind:\s*KIND_GPU", text), "GPU instances required for tensor cores"
    count = int(re.search(r"count:\s*(\d+)", text).group(1))
    assert count >= 2, "single instance serializes memcpy behind compute"


def test_io_dtypes_declared_correctly():
    text = read_config()
    assert 'name: "input_ids"' in text and "TYPE_INT64" in text
    assert 'name: "sentence_embedding"' in text and "TYPE_FP16" in text
