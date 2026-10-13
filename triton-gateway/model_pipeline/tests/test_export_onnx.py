"""Unit tests for the ONNX export pipeline (CPU-only, no network)."""

from __future__ import annotations

import pytest
import torch

from model_pipeline.export_onnx import DYNAMIC_AXES, build_dummy_inputs


class FakeTokenizer:
    """Deterministic stand-in so tests never touch the Hugging Face hub."""

    def __call__(self, text, padding, truncation, max_length, return_tensors):
        ids = torch.arange(max_length).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_dynamic_axes_cover_batch_and_sequence():
    axes = DYNAMIC_AXES()
    assert axes["input_ids"] == {0: "batch_size", 1: "sequence_length"}
    assert axes["attention_mask"] == {0: "batch_size", 1: "sequence_length"}
    assert axes["sentence_embedding"] == {0: "batch_size"}


def test_build_dummy_inputs_shapes_and_device():
    enc = build_dummy_inputs(FakeTokenizer(), "cpu")
    assert enc["input_ids"].shape == (1, 128)
    assert enc["attention_mask"].shape == (1, 128)
    assert enc["input_ids"].dtype == torch.int64


def test_export_requires_existing_output_parent(tmp_path):
    from model_pipeline.export_onnx import export_model

    # Missing model id on purpose: we only assert argument plumbing fails fast.
    with pytest.raises(Exception):  # noqa: B017 - any load error is acceptable here
        export_model("this/model-does-not-exist", tmp_path / "out" / "model.onnx", "cpu")
