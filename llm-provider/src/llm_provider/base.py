"""The seam itself: one protocol both providers satisfy.

Two methods, because they are the two shapes a consumer needs: ``complete`` for
a single grounded answer (the RAG service's only LLM call), and ``stream`` for
token-by-token output (an agent loop, piece 2). Everything provider-specific is
in the concrete classes; consumers depend only on this protocol and the neutral
types.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from llm_provider.types import (
    Completion,
    Message,
    ProviderConfig,
    StreamEvent,
    ToolSpec,
)


@runtime_checkable
class LLMProvider(Protocol):
    """Provider-neutral chat interface. Implementations: ``AnthropicProvider``,
    ``OpenAIProvider``. Select one with ``llm_provider.get_provider()``."""

    #: short, stable id ("anthropic" | "openai") — handy for tracing/logging.
    name: str

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        config: ProviderConfig | None = None,
    ) -> Completion:
        """One request → one response (text and/or tool calls), with usage."""
        ...

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        config: ProviderConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Same request, yielded as ``StreamEvent``s ending in ``MessageDone``."""
        ...
