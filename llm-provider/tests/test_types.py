from llm_provider.types import (
    Message,
    ProviderConfig,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def test_message_normalises_str_and_blocks():
    assert Message(role="user", content="hi").blocks() == [TextBlock(text="hi")]
    m = Message(role="assistant", content=[TextBlock(text="a"), TextBlock(text="b")])
    assert m.text() == "ab"


def test_tool_blocks_round_trip():
    tu = ToolUseBlock(id="t1", name="search", input={"q": "x"})
    assert tu.type == "tool_use"
    assert ToolResultBlock(tool_use_id="t1", content="r").is_error is False


def test_provider_config_extra_is_opaque():
    cfg = ProviderConfig(
        model="m", extra={"cache_control": True, "response_format": {"type": "json"}}
    )
    assert cfg.extra["cache_control"] is True
    assert cfg.max_tokens == 1024  # sane default
