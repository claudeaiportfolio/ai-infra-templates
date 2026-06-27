"""OpenAI implementation of the seam.

Where Claude carries the tool round-trip as content blocks inside user/assistant
turns, OpenAI splits it: the assistant turn gets a ``tool_calls`` array and each
result is its own ``role:"tool"`` message keyed by ``tool_call_id``. That
divergence is exactly what the neutral types insulate the consumer from.

``cache_control`` is Claude's knob; this impl pops and ignores it (the worked
example of "the other implementation ignores what isn't its own"). Other
``extra`` keys (e.g. ``response_format``) are forwarded as request params.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from llm_provider.types import (
    Completion,
    Message,
    MessageDone,
    ProviderConfig,
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

_STOP: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "other",
}


def to_openai_tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in (tools or [])
    ]


def system_text(messages: list[Message], config: ProviderConfig | None) -> str:
    parts: list[str] = []
    if config and config.system:
        parts.append(config.system)
    parts += [m.text() for m in messages if m.role == "system" and m.text()]
    return "\n\n".join(parts)


def to_openai_messages(messages: list[Message], system: str) -> list[dict[str, Any]]:
    """Flatten neutral messages to OpenAI's chat format. An assistant turn with
    tool calls becomes one message with a ``tool_calls`` array; a ``tool`` turn
    fans out into one ``role:"tool"`` message per result block."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "tool":
            for b in m.blocks():
                if isinstance(b, ToolResultBlock):
                    out.append(
                        {"role": "tool", "tool_call_id": b.tool_use_id, "content": b.content}
                    )
            continue
        text = "".join(b.text for b in m.blocks() if isinstance(b, TextBlock))
        tool_calls = [
            {
                "id": b.id,
                "type": "function",
                "function": {"name": b.name, "arguments": json.dumps(b.input)},
            }
            for b in m.blocks()
            if isinstance(b, ToolUseBlock)
        ]
        msg: dict[str, Any] = {"role": m.role, "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        out.append(msg)
    return out


def from_openai_response(resp: Any) -> Completion:
    choice = resp.choices[0]
    msg = choice.message
    tool_calls = [
        ToolUseBlock(
            id=tc.id,
            name=tc.function.name,
            input=json.loads(tc.function.arguments) if tc.function.arguments else {},
        )
        for tc in (msg.tool_calls or [])
    ]
    return Completion(
        text=msg.content or "",
        tool_calls=tool_calls,
        stop_reason=_STOP.get(choice.finish_reason or "", "other"),
        model=resp.model,
        usage=Usage(
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
        ),
    )


class OpenAIProvider:
    name = "openai"

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI()

    def _request_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        config: ProviderConfig | None,
    ) -> dict[str, Any]:
        cfg = config or ProviderConfig(model="gpt-4o-mini")
        extra = dict(cfg.extra)
        extra.pop("cache_control", None)  # Claude-only; OpenAI ignores it.
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "messages": to_openai_messages(messages, system_text(messages, cfg)),
            **extra,
        }
        openai_tools = to_openai_tools(tools)
        if openai_tools:
            kwargs["tools"] = openai_tools
        return kwargs

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        config: ProviderConfig | None = None,
    ) -> Completion:
        resp = await self._client.chat.completions.create(
            **self._request_kwargs(messages, tools, config)
        )
        return from_openai_response(resp)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        config: ProviderConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        kwargs = self._request_kwargs(messages, tools, config)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        stop: StopReason = "end_turn"
        usage = Usage()
        async for chunk in await self._client.chat.completions.create(**kwargs):
            if chunk.usage:
                usage = Usage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                )
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta and delta.content:
                yield TextDelta(text=delta.content)
            for tc in getattr(delta, "tool_calls", None) or []:
                fn = tc.function
                yield ToolUseDelta(
                    id=tc.id or "",
                    name=(fn.name or "") if fn else "",
                    partial_json=(fn.arguments or "") if fn else "",
                )
            if choice.finish_reason:
                stop = _STOP.get(choice.finish_reason, "other")
        yield MessageDone(stop_reason=stop, usage=usage)
