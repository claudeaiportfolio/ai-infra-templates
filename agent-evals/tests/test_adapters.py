"""Tests for the adapter modules.

These tests verify the offline behaviour (no env vars -> no-op) and the
ImportError shape when the SDK isn't installed. They do not exercise live
posting to Langfuse or Braintrust — those require credentials and network
access, and are out of scope for unit tests.
"""
from __future__ import annotations

import importlib
import sys

import pytest

# Reload modules between tests so the env-var-driven branches re-evaluate.


def _reload_adapter(name: str):
    """Import or reload the named adapter module."""
    full_name = f"agent_evals.adapters.{name}"
    if full_name in sys.modules:
        return importlib.reload(sys.modules[full_name])
    return importlib.import_module(full_name)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip Langfuse/Braintrust env vars before every test."""
    for k in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "BRAINTRUST_API_KEY",
        "BRAINTRUST_PROJECT",
        "BRAINTRUST_API_URL",
    ):
        monkeypatch.delenv(k, raising=False)


class TestLangfuseAdapter:
    def test_is_enabled_false_without_env(self) -> None:
        lf = _reload_adapter("langfuse")
        assert lf.is_enabled() is False

    def test_post_run_is_no_op_without_env(self) -> None:
        lf = _reload_adapter("langfuse")
        assert lf.post_run([], "test") is None

    def test_is_enabled_true_when_both_keys_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        lf = _reload_adapter("langfuse")
        assert lf.is_enabled() is True

    def test_is_enabled_false_with_only_public_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        lf = _reload_adapter("langfuse")
        assert lf.is_enabled() is False

    def test_raises_import_error_when_sdk_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        lf = _reload_adapter("langfuse")
        with pytest.raises(ImportError, match="langfuse SDK not installed"):
            lf.post_run([], "test")


class TestBraintrustAdapter:
    def test_is_enabled_false_without_env(self) -> None:
        bt = _reload_adapter("braintrust")
        assert bt.is_enabled() is False

    def test_post_run_is_no_op_without_env(self) -> None:
        bt = _reload_adapter("braintrust")
        assert bt.post_run([], "test") is None

    def test_is_enabled_true_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAINTRUST_API_KEY", "bt-test")
        bt = _reload_adapter("braintrust")
        assert bt.is_enabled() is True

    def test_raises_import_error_when_sdk_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAINTRUST_API_KEY", "bt-test")
        bt = _reload_adapter("braintrust")
        with pytest.raises(ImportError, match="braintrust SDK not installed"):
            bt.post_run([], "test")
