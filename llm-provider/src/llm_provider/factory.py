"""Provider selection — the whole swap is one config value.

``get_provider()`` reads ``LLM_PROVIDER`` (default ``anthropic``). Imports are
lazy so selecting one provider never constructs the other's client or requires
its API key in the environment.
"""

from __future__ import annotations

import os

from llm_provider.base import LLMProvider

PROVIDERS = ("anthropic", "openai")


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or os.environ.get("LLM_PROVIDER", "anthropic")).lower()
    if name == "anthropic":
        from llm_provider.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if name == "openai":
        from llm_provider.openai_provider import OpenAIProvider

        return OpenAIProvider()
    raise ValueError(
        f"unknown LLM_PROVIDER {name!r} (expected one of {', '.join(PROVIDERS)})"
    )
