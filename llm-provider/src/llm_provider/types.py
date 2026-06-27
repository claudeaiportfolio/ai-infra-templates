"""Provider-neutral request/response types for the LLM seam.

These model the *coupling surface* between a loop and a provider: messages, the
tool round-trip (tool-use + tool-result), the streamed event contract, and a
completion with usage. This is where Claude (content blocks, ``tool_use`` /
``tool_result``) and OpenAI (``tool_calls`` + ``role:"tool"`` messages) actually
diverge, so a naive ``complete(str) -> str`` would be too thin to be real.

Provider-*differentiating* features (prompt caching, extended thinking,
structured-output modes) are deliberately NOT modelled here. They ride through
``ProviderConfig.extra`` as opaque config that the concrete implementation
interprets and the other implementation ignores — abstracting them would cost
more than the lock-in it saves.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop", "other"]


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    """A model's request to call a tool (provider-neutral)."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """The caller's reply to a ``ToolUseBlock``, keyed by ``tool_use_id``."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


class Message(BaseModel):
    """One turn. ``content`` is either plain text or a list of blocks; the
    helpers below normalise so translators don't branch on the union everywhere.
    """

    role: Role
    content: str | list[ContentBlock]

    def blocks(self) -> list[ContentBlock]:
        if isinstance(self.content, str):
            return [TextBlock(text=self.content)]
        return self.content

    def text(self) -> str:
        return "".join(b.text for b in self.blocks() if isinstance(b, TextBlock))


class ToolSpec(BaseModel):
    """A tool the model may call. ``input_schema`` is JSON Schema (the common
    shape both providers accept, modulo wrapper keys the translators add)."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class Completion(BaseModel):
    """A single non-streamed response. ``tool_calls`` is empty unless the model
    asked to use tools (``stop_reason == "tool_use"``)."""

    model_config = ConfigDict(protected_namespaces=())

    text: str = ""
    tool_calls: list[ToolUseBlock] = Field(default_factory=list)
    stop_reason: StopReason = "end_turn"
    model: str = ""
    usage: Usage = Field(default_factory=Usage)


# --- streamed event contract -------------------------------------------------
class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolUseDelta(BaseModel):
    """Incremental tool-call: ``id``/``name`` arrive on the opening event, then
    ``partial_json`` accumulates the arguments across subsequent events."""

    type: Literal["tool_use_delta"] = "tool_use_delta"
    id: str = ""
    name: str = ""
    partial_json: str = ""


class MessageDone(BaseModel):
    type: Literal["message_done"] = "message_done"
    stop_reason: StopReason = "end_turn"
    usage: Usage = Field(default_factory=Usage)


StreamEvent = TextDelta | ToolUseDelta | MessageDone


class ProviderConfig(BaseModel):
    """Request knobs common to both providers, plus an opaque ``extra`` bag for
    provider-specific config. The seam never reads ``extra``; the concrete
    provider does (Claude impl reads e.g. ``cache_control`` / ``thinking``;
    OpenAI impl reads e.g. ``response_format``) and ignores what isn't its own.
    """

    model_config = ConfigDict(protected_namespaces=())

    model: str
    max_tokens: int = 1024
    temperature: float = 0.0
    system: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
