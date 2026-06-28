"""Anthropic (Claude) implementation of the seam.

The translation functions are module-level and pure so the swap test can assert
the neutral→Claude request shape and the Claude→neutral response shape without a
network call. ``cache_control`` is the worked example of "Claude reads its own
opaque config": it's lifted off ``ProviderConfig.extra`` and applied to the
system prompt; every other key in ``extra`` is forwarded as a top-level request
param.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from anthropic import NOT_GIVEN, AsyncAnthropic

from llm_provider.types import (
    Completion,
    Message,
    MessageDone,
    ProviderConfig,
    StopReason,
    StreamEvent,
    TextDelta,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    ToolUseDelta,
    Usage,
)

_STOP: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop",
}


def to_anthropic_tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in (tools or [])
    ]


def _block_to_anthropic(block: Any) -> dict[str, Any]:
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultBlock):
        out: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
        }
        if block.is_error:
            out["is_error"] = True
        return out
    return {"type": "text", "text": block.text}


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Map neutral messages to Anthropic's. ``system`` turns are dropped here
    (Anthropic takes system as a top-level param — see ``system_param``); a
    ``tool`` turn becomes a ``user`` turn carrying ``tool_result`` blocks."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        role = "user" if m.role == "tool" else m.role
        out.append({"role": role, "content": [_block_to_anthropic(b) for b in m.blocks()]})
    return out


def system_param(
    messages: list[Message], config: ProviderConfig | None, cache: bool
) -> Any:
    """Collect ``config.system`` + any system-role message text. When ``cache``
    is set, return the block form with ``cache_control`` so the prefix is cached;
    otherwise a plain string (or ``NOT_GIVEN`` when there is no system text)."""
    parts: list[str] = []
    if config and config.system:
        parts.append(config.system)
    parts += [m.text() for m in messages if m.role == "system" and m.text()]
    text = "\n\n".join(parts)
    if not text:
        return NOT_GIVEN
    if cache:
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    return text


def from_anthropic_response(resp: Any) -> Completion:
    text = "".join(b.text for b in resp.content if b.type == "text")
    tool_calls = [
        ToolUseBlock(id=b.id, name=b.name, input=dict(b.input))
        for b in resp.content
        if b.type == "tool_use"
    ]
    return Completion(
        text=text,
        tool_calls=tool_calls,
        stop_reason=_STOP.get(resp.stop_reason or "", "other"),
        model=resp.model,
        usage=Usage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            # Optional on the SDK usage object (absent/None when caching is off);
            # pass through as-is so None ("not reported") stays distinct from 0.
            cache_creation_input_tokens=getattr(
                resp.usage, "cache_creation_input_tokens", None
            ),
            cache_read_input_tokens=getattr(
                resp.usage, "cache_read_input_tokens", None
            ),
        ),
    )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        # Defer client construction so importing the module needs no API key.
        self._client = client or AsyncAnthropic()

    def _request_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        config: ProviderConfig | None,
    ) -> dict[str, Any]:
        cfg = config or ProviderConfig(model="claude-sonnet-4-5")
        extra = dict(cfg.extra)
        cache = bool(extra.pop("cache_control", False))
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "system": system_param(messages, cfg, cache),
            "messages": to_anthropic_messages(messages),
            **extra,
        }
        anthropic_tools = to_anthropic_tools(tools)
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        return kwargs

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        config: ProviderConfig | None = None,
    ) -> Completion:
        resp = await self._client.messages.create(
            **self._request_kwargs(messages, tools, config)
        )
        return from_anthropic_response(resp)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        config: ProviderConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        kwargs = self._request_kwargs(messages, tools, config)
        stop: StopReason = "end_turn"
        usage = Usage()
        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                etype = getattr(event, "type", "")
                if etype == "content_block_start" and event.content_block.type == "tool_use":
                    yield ToolUseDelta(id=event.content_block.id, name=event.content_block.name)
                elif etype == "content_block_delta":
                    d = event.delta
                    if d.type == "text_delta":
                        yield TextDelta(text=d.text)
                    elif d.type == "input_json_delta":
                        yield ToolUseDelta(partial_json=d.partial_json)
                elif etype == "message_delta":
                    if event.delta.stop_reason:
                        stop = _STOP.get(event.delta.stop_reason, "other")
                    usage.output_tokens = event.usage.output_tokens
            final = await stream.get_final_message()
            usage.input_tokens = final.usage.input_tokens
            # Cache tokens are only on the final aggregated usage, not the
            # per-event message_delta usage — read them once at the end.
            usage.cache_creation_input_tokens = getattr(
                final.usage, "cache_creation_input_tokens", None
            )
            usage.cache_read_input_tokens = getattr(
                final.usage, "cache_read_input_tokens", None
            )
        yield MessageDone(stop_reason=stop, usage=usage)
