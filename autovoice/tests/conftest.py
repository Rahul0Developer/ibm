"""Shared pytest fixtures for AutoVoice tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the src/ layout importable without installing the package (dev convenience;
# CI uses `uv sync` which installs the project).
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove project env vars so tests never read a developer's local .env."""
    prefixes = ("BQ_", "GCP_", "LLM_", "CORS_", "NER_", "CLF_", "HF_", "GEMINI_", "CHROMA_")
    for key in list(os.environ):
        if key.isupper() and key.startswith(prefixes):
            monkeypatch.delenv(key, raising=False)
