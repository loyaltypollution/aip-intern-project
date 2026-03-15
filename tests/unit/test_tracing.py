import pytest
from aip_intern.core.tracing import get_langfuse, get_callback


def test_get_langfuse_returns_none_when_keys_absent(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    assert get_langfuse() is None


def test_get_callback_returns_none_when_client_is_none():
    assert get_callback(None) is None
