"""Langfuse tracing wrapper.

Separate from metrics.py because Langfuse has a different initialisation lifecycle
and can be disabled without affecting metric collection.

Usage:
    lf = get_langfuse()           # None if LANGFUSE_SECRET_KEY not set
    handler = get_callback(lf)    # None if lf is None
    # pass handler in invoke_config["callbacks"] = [handler]

Phase 3 interns: tracing is orthogonal to failure injection.
You can disable it by unsetting LANGFUSE_SECRET_KEY.
"""

from __future__ import annotations

import os
from typing import Optional


def get_langfuse():
    """Return a Langfuse client, or None if keys are not configured."""
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not secret_key:
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            secret_key=secret_key,
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception:
        return None


def get_callback(langfuse_client) -> Optional[object]:
    """Return a LangfuseCallbackHandler for use in LangGraph invoke config, or None."""
    if langfuse_client is None:
        return None
    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler(client=langfuse_client)
    except Exception:
        return None
