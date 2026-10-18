"""Structured JSON logging with request-id support (structlog).

Library code must use ``get_logger(__name__)`` and keyword arguments -- never
``print``.  Raw user text is NOT logged by default (privacy/responsible-AI);
set ``LOG_RAW_TEXT=true`` only during local debugging.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from typing import Any

import structlog

# Per-request id propagated through middleware -> log records.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_CONFIGURED = False


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure stdlib + structlog. Idempotent; call once at process start."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_request_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer(serializer=json.dumps))
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        # sys.stdout is resolved lazily so captured/redirected streams (pytest)
        # never leave a closed-file handle behind.
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def _add_request_id(_logger: Any, _method: str, event_dict: dict) -> dict:
    event_dict["request_id"] = request_id_var.get()
    return event_dict


def get_logger(name: str = "autovoice") -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to ``name``."""
    return structlog.get_logger(name)
