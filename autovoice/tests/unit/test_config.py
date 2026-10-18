"""Phase-1 smoke tests: config defaults/overrides and structured logging."""

from __future__ import annotations

import json

from autovoice.config import Settings
from autovoice.logging_conf import configure_logging, get_logger, request_id_var


def test_settings_defaults(clean_env) -> None:
    s = Settings(_env_file=None)
    assert s.llm_provider == "none"
    assert s.max_text_length == 5000
    assert s.rate_limit_per_minute == 60
    assert s.gcp_project_id is None


def test_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10")
    s = Settings(_env_file=None)
    assert s.llm_provider == "gemini"
    assert s.rate_limit_per_minute == 10


def test_cors_string_is_split(monkeypatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example , https://b.example")
    s = Settings(_env_file=None)
    assert s.cors_origins == ["https://a.example", "https://b.example"]


def test_invalid_llm_provider_rejected(monkeypatch) -> None:
    import pytest
    from pydantic import ValidationError

    monkeypatch.setenv("LLM_PROVIDER", "gpt-4o")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_json_logs_include_request_id(capsys) -> None:
    configure_logging(level="INFO", json_logs=True)
    token = request_id_var.set("abc-123")
    try:
        log = get_logger("test")
        log.info("hello", answer=42)
    finally:
        request_id_var.reset(token)
    out = capsys.readouterr().out.strip().splitlines()[-1]
    rec = json.loads(out)
    assert rec["event"] == "hello"
    assert rec["request_id"] == "abc-123"
    assert rec["answer"] == 42
    assert "timestamp" in rec or "time" in rec  # structlog TimeStamper field
