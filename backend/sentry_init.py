"""
sentry_init.py — optional Sentry error monitoring for the FastAPI backend.

No-op unless SENTRY_DSN is set, so local dev and CI run unaffected. Call
init_sentry() once, as early as possible (before the FastAPI app is created),
so the FastAPI/Starlette integration can hook request handling.

Env vars:
  SENTRY_DSN          — project DSN from sentry.io (required to enable)
  SENTRY_ENVIRONMENT  — e.g. "production" / "dev" (default: "development")
  SENTRY_TRACES_SAMPLE_RATE — 0.0–1.0 perf tracing sample (default: 0.0)
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def init_sentry() -> bool:
    """Initialize Sentry if SENTRY_DSN is configured. Returns True if enabled."""
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        log.info("Sentry disabled (no SENTRY_DSN set).")
        return False

    try:
        import sentry_sdk
    except ImportError:
        log.warning("SENTRY_DSN set but sentry-sdk not installed; skipping.")
        return False

    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
        # Don't send request bodies / user questions by default (privacy).
        send_default_pii=False,
    )
    log.info("Sentry initialized.")
    return True
