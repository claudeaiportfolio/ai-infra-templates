"""Loop tests for the llm-provider routing (network-free).

The loop drives REAL `AnthropicProvider` / `OpenAIProvider` instances with FAKE
SDK clients (no API calls), so the neutral translation is exercised end to end.
Covers: the anthropic path (behaviour preserved), the openai path, a provider
swap yielding the same `LoopResult`, the duck-typed MCP fallback, and
`_make_provider` selection + api-key injection.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from llm_provider import ToolSpec
from llm_provider.anthropic_provider import AnthropicProvider
from llm_provider.openai_provider import OpenAIProvider

from agent_core import AgentLoop, Tracer

TOOLS = [
    {
        "name": "search",
        "description": "search the corpus",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
]


# --- fakes -------------------------------------------------------------------
class _FakeMCP:
    """Mimics MCPClient: neutral tool_specs() + tools_for_anthropic() + call_tool()."""

    def __init__(self, tools=None, results=None):
        self._tools = tools or []
        self._results = results or {}
        self.calls: list[tuple[str, dict]] = []

    def tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t["input_schema"],
            )
            for t in self._tools
        ]

    def tools_for_anthropic(self):
        return self._tools

    async def call_tool(self, name, arguments, **_kw):
        self.calls.append((name, arguments))
        return self._results[name]


class _DuckTypedMCP:
    """A consumer that duck-types MCPClient with ONLY tools_for_anthropic()."""

    def __init__(self, tools=None, results=None):
        self._tools = tools or []
        self._results = results or {}

    def tools_for_anthropic(self):
        return self._tools

    async def call_tool(self, name, arguments, **_kw):
        return self._results[name]


def _anthropic_resp(
    *, text="", tool_calls=(), stop_reason="end_turn", model="claude-sonnet-4-5",
    input_tokens=10, output_tokens=5, cache_creation=0, cache_read=0,
):
    content = []
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    for tc in tool_calls:
        content.append(
            SimpleNamespace(type="tool_use", id=tc["id"], name=tc["name"], input=tc["input"])
        )
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        ),
    )


def _openai_resp(
    *, text=None, tool_calls=(), finish_reason="stop", model="gpt-4o-mini",
    prompt_tokens=10, completion_tokens=5,
):
    tcs = [
        SimpleNamespace(
            id=tc["id"],
            function=SimpleNamespace(name=tc["name"], arguments=json.dumps(tc["input"])),
        )
        for tc in tool_calls
    ]
    msg = SimpleNamespace(content=text, tool_calls=tcs or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class _ScriptedAnthropicClient:
    def __init__(self, responses):
        self._responses = list(responses)

        async def create(**_kwargs):
            return self._responses.pop(0)

        self.messages = SimpleNamespace(create=create)


class _ScriptedOpenAIClient:
    def __init__(self, responses):
        self._responses = list(responses)

        async def create(**_kwargs):
            return self._responses.pop(0)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _loop(provider, mcp, tmp_path, monkeypatch, **kw):
    loop = AgentLoop(mcp=mcp, tracer=Tracer(trace_dir=tmp_path, mirror_to_stderr=False), **kw)
    monkeypatch.setattr(loop, "_make_provider", lambda: provider)
    return loop


def _events(trace_path):
    return [json.loads(line) for line in Path(trace_path).read_text().splitlines()]


# --- anthropic path (behaviour preserved) ------------------------------------
async def test_anthropic_single_turn_end_turn(tmp_path, monkeypatch):
    provider = AnthropicProvider(
        client=_ScriptedAnthropicClient(
            [_anthropic_resp(text="grounded answer", cache_read=900)]
        )
    )
    loop = _loop(provider, _FakeMCP(), tmp_path, monkeypatch)
    result = await loop.run("question?")

    assert result.final_text == "grounded answer"
    assert result.turns == 1
    assert result.tool_calls == 0
    assert result.stop_reason == "end_turn"
    assert result.prompt_version == "v1"
    # Cache tokens flow from the neutral Usage into the trace.
    resp_evt = next(e for e in _events(result.trace_path) if e["kind"] == "claude_response")
    assert resp_evt["cache_read_input_tokens"] == 900
    assert resp_evt["cache_creation_input_tokens"] == 0


async def test_anthropic_tool_use_then_end_turn(tmp_path, monkeypatch):
    provider = AnthropicProvider(
        client=_ScriptedAnthropicClient(
            [
                _anthropic_resp(
                    tool_calls=[{"id": "t1", "name": "search", "input": {"q": "x"}}],
                    stop_reason="tool_use",
                ),
                _anthropic_resp(text="final answer"),
            ]
        )
    )
    mcp = _FakeMCP(tools=TOOLS, results={"search": "found it"})
    loop = _loop(provider, mcp, tmp_path, monkeypatch)
    result = await loop.run("question?")

    assert result.final_text == "final answer"
    assert result.turns == 2
    assert result.tool_calls == 1
    assert result.stop_reason == "end_turn"
    assert mcp.calls == [("search", {"q": "x"})]


async def test_abnormal_stop_reason_surfaced(tmp_path, monkeypatch):
    provider = AnthropicProvider(
        client=_ScriptedAnthropicClient(
            [_anthropic_resp(text="partial", stop_reason="max_tokens")]
        )
    )
    loop = _loop(provider, _FakeMCP(), tmp_path, monkeypatch)
    result = await loop.run("question?")
    assert result.stop_reason == "max_tokens"
    assert result.final_text.startswith("[loop ended with stop_reason=max_tokens]")


async def test_bail_on_repeated_tool_errors(tmp_path, monkeypatch):
    provider = AnthropicProvider(
        client=_ScriptedAnthropicClient(
            [
                _anthropic_resp(
                    tool_calls=[{"id": "t1", "name": "search", "input": {}}],
                    stop_reason="tool_use",
                ),
                _anthropic_resp(
                    tool_calls=[{"id": "t2", "name": "search", "input": {}}],
                    stop_reason="tool_use",
                ),
            ]
        )
    )
    mcp = _FakeMCP(tools=TOOLS, results={"search": "ERROR: boom"})
    loop = _loop(provider, mcp, tmp_path, monkeypatch)
    result = await loop.run("question?")
    assert result.stop_reason == "bailed_on_repeated_errors"


# --- openai path -------------------------------------------------------------
async def test_openai_single_turn(tmp_path, monkeypatch):
    provider = OpenAIProvider(
        client=_ScriptedOpenAIClient([_openai_resp(text="grounded answer")])
    )
    loop = _loop(provider, _FakeMCP(), tmp_path, monkeypatch, model="gpt-4o-mini")
    result = await loop.run("question?")
    assert result.final_text == "grounded answer"
    assert result.turns == 1
    assert result.stop_reason == "end_turn"
    # OpenAI doesn't report cache tokens → coerced to 0 in the trace, not an error.
    resp_evt = next(e for e in _events(result.trace_path) if e["kind"] == "claude_response")
    assert resp_evt["cache_read_input_tokens"] == 0
    assert resp_evt["cache_creation_input_tokens"] == 0


async def test_openai_tool_use_then_end(tmp_path, monkeypatch):
    provider = OpenAIProvider(
        client=_ScriptedOpenAIClient(
            [
                _openai_resp(
                    tool_calls=[{"id": "t1", "name": "search", "input": {"q": "x"}}],
                    finish_reason="tool_calls",
                ),
                _openai_resp(text="final answer"),
            ]
        )
    )
    mcp = _FakeMCP(tools=TOOLS, results={"search": "found it"})
    loop = _loop(provider, mcp, tmp_path, monkeypatch, model="gpt-4o-mini")
    result = await loop.run("question?")
    assert result.final_text == "final answer"
    assert result.turns == 2
    assert result.tool_calls == 1
    assert mcp.calls == [("search", {"q": "x"})]


# --- the swap: same scripted conversation, same LoopResult -------------------
async def test_provider_swap_same_loop_result(tmp_path, monkeypatch):
    anthropic = AnthropicProvider(
        client=_ScriptedAnthropicClient(
            [
                _anthropic_resp(
                    tool_calls=[{"id": "t1", "name": "search", "input": {"q": "x"}}],
                    stop_reason="tool_use",
                ),
                _anthropic_resp(text="grounded answer"),
            ]
        )
    )
    openai = OpenAIProvider(
        client=_ScriptedOpenAIClient(
            [
                _openai_resp(
                    tool_calls=[{"id": "t1", "name": "search", "input": {"q": "x"}}],
                    finish_reason="tool_calls",
                ),
                _openai_resp(text="grounded answer"),
            ]
        )
    )
    results = {"search": "found it"}

    a = await _loop(
        anthropic, _FakeMCP(tools=TOOLS, results=results), tmp_path / "a", monkeypatch
    ).run("question?")
    o = await _loop(
        openai, _FakeMCP(tools=TOOLS, results=results), tmp_path / "o", monkeypatch,
        model="gpt-4o-mini",
    ).run("question?")

    # The neutral result is identical across providers (trace_path aside).
    assert (a.final_text, a.turns, a.tool_calls, a.stop_reason) == (
        "grounded answer", 2, 1, "end_turn",
    )
    assert (a.final_text, a.turns, a.tool_calls, a.stop_reason) == (
        o.final_text, o.turns, o.tool_calls, o.stop_reason,
    )


# --- duck-typed MCP fallback (tools_for_anthropic only) ----------------------
async def test_tool_specs_fallback_for_duck_typed_mcp(tmp_path, monkeypatch):
    provider = AnthropicProvider(
        client=_ScriptedAnthropicClient([_anthropic_resp(text="ok")])
    )
    mcp = _DuckTypedMCP(tools=TOOLS)
    loop = _loop(provider, mcp, tmp_path, monkeypatch)
    result = await loop.run("question?")
    assert result.final_text == "ok"
    # The loop derived neutral specs from tools_for_anthropic(): tool name logged.
    start_evt = next(e for e in _events(result.trace_path) if e["kind"] == "loop_start")
    assert start_evt["tools"] == ["search"]


# --- provider selection + api-key injection ----------------------------------
def test_make_provider_defaults_to_anthropic_and_injects_key(tmp_path, monkeypatch):
    captured = {}

    class _FakeAsyncAnthropic:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr("anthropic.AsyncAnthropic", _FakeAsyncAnthropic)
    loop = AgentLoop(
        mcp=_FakeMCP(),
        tracer=Tracer(trace_dir=tmp_path, mirror_to_stderr=False),
        api_key="secret-123",
    )
    provider = loop._make_provider()
    assert provider.name == "anthropic"
    assert captured["api_key"] == "secret-123"


def test_make_provider_selects_openai_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    # OpenAIProvider imports AsyncOpenAI at module load, so patch it there.
    monkeypatch.setattr("llm_provider.openai_provider.AsyncOpenAI", lambda *a, **k: object())
    loop = AgentLoop(
        mcp=_FakeMCP(), tracer=Tracer(trace_dir=tmp_path, mirror_to_stderr=False)
    )
    assert loop._make_provider().name == "openai"


def test_default_temperature_preserves_prior_behaviour():
    # The loop used to omit temperature (Anthropic server default 1.0); the
    # explicit default must match so behaviour is preserved.
    loop = AgentLoop(mcp=_FakeMCP(), tracer=Tracer(mirror_to_stderr=False))
    assert loop.temperature == 1.0


@pytest.mark.parametrize("enabled", [True, False])
async def test_prompt_caching_flag_passed_through(tmp_path, monkeypatch, enabled):
    # Capture the request kwargs the Anthropic SDK client receives to prove the
    # cached system block is built only when caching is enabled.
    seen = {}

    class _CapturingClient:
        def __init__(self):
            async def create(**kwargs):
                seen.update(kwargs)
                return _anthropic_resp(text="ok")

            self.messages = SimpleNamespace(create=create)

    provider = AnthropicProvider(client=_CapturingClient())
    loop = _loop(
        provider, _FakeMCP(), tmp_path, monkeypatch, enable_prompt_caching=enabled
    )
    await loop.run("question?")
    if enabled:
        assert isinstance(seen["system"], list)
        assert seen["system"][0]["cache_control"] == {"type": "ephemeral"}
    else:
        assert isinstance(seen["system"], str)
    # Temperature is sent explicitly at the preserved default.
    assert seen["temperature"] == 1.0
