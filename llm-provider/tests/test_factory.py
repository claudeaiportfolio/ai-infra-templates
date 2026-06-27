import pytest

from llm_provider import get_provider


def test_get_provider_selects_by_name(monkeypatch):
    # Dummy keys so the SDK clients construct without network or real creds.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert get_provider("anthropic").name == "anthropic"
    assert get_provider("openai").name == "openai"


def test_get_provider_reads_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert get_provider().name == "anthropic"


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        get_provider("gemini")
