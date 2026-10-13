"""Integration test: Triton server <-> Python client over gRPC.

Skipped automatically when no Triton endpoint is reachable, so CI on GPUs-less
runners stays green while GPU boxes run the full contract suite:

    TRITON_GRPC_ADDR=localhost:8001 pytest tests -q -m integration
"""

from __future__ import annotations

import os
import socket

import numpy as np
import pytest

TRITON_ADDR = os.environ.get("TRITON_GRPC_ADDR", "localhost:8001")


def triton_reachable(addr: str) -> bool:
    host, port = addr.rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not triton_reachable(TRITON_ADDR),
    reason=f"no Triton server listening on {TRITON_ADDR}",
)


@pytest.fixture(scope="module")
def triton_client():
    tritonclient = pytest.importorskip("tritonclient.grpc")
    return tritonclient.InferenceServerClient(url=TRITON_ADDR)


def _payload(batch: int, seq: int):
    input_ids = np.random.randint(0, 50257, size=(batch, seq), dtype=np.int64)
    attention_mask = np.ones((batch, seq), dtype=np.int64)
    return input_ids, attention_mask


def test_server_and_model_ready(triton_client):
    assert triton_client.is_server_live()
    assert triton_client.is_model_ready("text_embeddings")


def test_single_infer_returns_384_dim_embedding(triton_client):
    ids, mask = _payload(1, 16)
    result = triton_client.infer(
        "text_embeddings",
        inputs=[
            _input("input_ids", ids.shape, np.int64, ids.tobytes()),
            _input("attention_mask", mask.shape, np.int64, mask.tobytes()),
        ],
        outputs=[_output("sentence_embedding")],
    )
    emb = np.frombuffer(result.as_raw_output_contents()[0], dtype=np.float16)
    assert emb.shape == (384,)


def _input(name, shape, dtype, raw):
    tritonclient = pytest.importorskip("tritonclient.grpc")
    inp = tritonclient.InferInput(name, list(shape), "INT64")
    inp.set_data_from_numpy(np.frombuffer(raw, dtype=dtype).reshape(shape))
    return inp


def _output(name):
    tritonclient = pytest.importorskip("tritonclient.grpc")
    return tritonclient.InferRequestedOutput(name)
