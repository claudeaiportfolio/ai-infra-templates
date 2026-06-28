"""Pure (network-free) translation: neutral → provider request, provider →
neutral response. This is where Claude and OpenAI diverge, so it's where the
seam earns its keep."""

from types import SimpleNamespace

from llm_provider import anthropic_provider as ap
from llm_provider import openai_provider as op
from llm_provider.types import (
    Message,
    ProviderConfig,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

TOOLS = [
    ToolSpec(
        name="search",
        description="search docs",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
]

CONVO = [
    Message(role="user", content="hi"),
    Message(role="assistant", content=[ToolUseBlock(id="t1", name="search", input={"q": "x"})]),
    Message(role="tool", content=[ToolResultBlock(tool_use_id="t1", content="found")]),
]


def test_tool_shapes():
    a = ap.to_anthropic_tools(TOOLS)[0]
    assert set(a) == {"name", "description", "input_schema"}
    o = op.to_openai_tools(TOOLS)[0]
    assert o["type"] == "function"
    assert o["function"]["parameters"] == TOOLS[0].input_schema


def test_tool_roundtrip_diverges_per_provider():
    a = ap.to_anthropic_messages(CONVO)
    # Anthropic carries the result inside a user turn as a tool_result block.
    assert a[-1]["role"] == "user"
    assert a[-1]["content"][0]["type"] == "tool_result"
    assert a[-1]["content"][0]["tool_use_id"] == "t1"

    o = op.to_openai_messages(CONVO, system="")
    # OpenAI: assistant turn has tool_calls; result is its own role:"tool" msg.
    assert o[1]["tool_calls"][0]["function"]["name"] == "search"
    assert o[-1]["role"] == "tool"
    assert o[-1]["tool_call_id"] == "t1"


def test_system_is_split_out_of_messages():
    msgs = [Message(role="system", content="S"), Message(role="user", content="u")]
    assert op.system_text(msgs, None) == "S"
    assert all(m["role"] != "system" for m in ap.to_anthropic_messages(msgs))


def test_anthropic_reads_cache_control_openai_ignores_it():
    cfg = ProviderConfig(model="m", system="SYS", extra={"cache_control": True})
    sysp = ap.system_param([], cfg, cache=True)
    assert isinstance(sysp, list)
    assert sysp[0]["cache_control"] == {"type": "ephemeral"}
    # OpenAI pops + drops cache_control rather than forwarding an invalid param.
    prov = op.OpenAIProvider(client=SimpleNamespace())
    kwargs = prov._request_kwargs([Message(role="user", content="u")], None, cfg)
    assert "cache_control" not in kwargs


def test_from_anthropic_response():
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="tool_use", id="t1", name="search", input={"q": "x"}),
        ],
        stop_reason="tool_use",
        model="claude-x",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    c = ap.from_anthropic_response(resp)
    assert c.text == "hello"
    assert c.stop_reason == "tool_use"
    assert c.tool_calls[0].name == "search"
    assert c.usage.input_tokens == 10


def test_from_anthropic_response_populates_cache_tokens():
    # When Claude reports cache usage, both fields ride through onto neutral Usage.
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        stop_reason="end_turn",
        model="claude-x",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=200,
            cache_read_input_tokens=1800,
        ),
    )
    c = ap.from_anthropic_response(resp)
    assert c.usage.cache_creation_input_tokens == 200
    assert c.usage.cache_read_input_tokens == 1800


def test_from_anthropic_response_cache_tokens_absent_is_none():
    # Older/uncached responses may not carry the attrs at all → None, not 0.
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        stop_reason="end_turn",
        model="claude-x",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    c = ap.from_anthropic_response(resp)
    assert c.usage.cache_creation_input_tokens is None
    assert c.usage.cache_read_input_tokens is None


def test_from_openai_response():
    msg = SimpleNamespace(
        content="hello",
        tool_calls=[
            SimpleNamespace(
                id="t1", function=SimpleNamespace(name="search", arguments='{"q": "x"}')
            )
        ],
    )
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        model="gpt-x",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    c = op.from_openai_response(resp)
    assert c.text == "hello"
    assert c.stop_reason == "tool_use"
    assert c.tool_calls[0].input == {"q": "x"}
    assert c.usage.output_tokens == 5
    # OpenAI doesn't surface discrete cache-token counts: the impl leaves both None.
    assert c.usage.cache_creation_input_tokens is None
    assert c.usage.cache_read_input_tokens is None
