"""llm-provider — a thin seam over Claude + OpenAI.

Abstracts the coupling surface a loop actually shares with a provider — messages,
the tool round-trip, the stream-event contract, a completion with usage — and
passes provider-*differentiating* features (caching, structured outputs, extended
thinking) through ``ProviderConfig.extra`` as opaque config. Swap providers with
one value: ``LLM_PROVIDER=anthropic|openai``.

    from llm_provider import get_provider, Message, ProviderConfig

    provider = get_provider()  # reads LLM_PROVIDER (default: anthropic)
    answer = await provider.complete(
        [Message(role="user", content="ground this in the chunks…")],
        config=ProviderConfig(model="claude-sonnet-4-5", system="You are…"),
    )
"""

from __future__ import annotations

from llm_provider.anthropic_provider import AnthropicProvider
from llm_provider.base import LLMProvider
from llm_provider.factory import PROVIDERS, get_provider
from llm_provider.openai_provider import OpenAIProvider
from llm_provider.types import (
    Completion,
    ContentBlock,
    Message,
    MessageDone,
    ProviderConfig,
    Role,
    StopReason,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    ToolUseDelta,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "PROVIDERS",
    "AnthropicProvider",
    "Completion",
    "ContentBlock",
    "LLMProvider",
    "Message",
    "MessageDone",
    "OpenAIProvider",
    "ProviderConfig",
    "Role",
    "StopReason",
    "StreamEvent",
    "TextBlock",
    "TextDelta",
    "ToolResultBlock",
    "ToolSpec",
    "ToolUseBlock",
    "ToolUseDelta",
    "Usage",
    "__version__",
    "get_provider",
]
