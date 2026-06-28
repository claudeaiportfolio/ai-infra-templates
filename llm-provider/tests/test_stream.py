"""Streamed-path coverage for the cache-token fields (network-free).

The non-streamed path is covered in test_translation; the stream path reads
cache tokens off the *final aggregated* message usage, so it needs its own fake.
"""

from types import SimpleNamespace

from llm_provider import MessageDone
from llm_provider.anthropic_provider import AnthropicProvider
from llm_provider.openai_provider import OpenAIProvider
from llm_provider.types import Message, ProviderConfig

CONVO = [Message(role="user", content="hi")]


class _FakeAnthropicStream:
    """Mimics AsyncAnthropic .messages.stream(...) as an async context manager."""

    def __init__(self, final_usage):
        self._final_usage = final_usage

        def stream(**kwargs):
            final_usage = self._final_usage

            class _Ctx:
                async def __aenter__(self_inner):
                    return self_inner

                async def __aexit__(self_inner, *exc):
                    return False

                async def __aiter__(self_inner):
                    # One text delta, then the message_delta carrying output usage.
                    yield SimpleNamespace(
                        type="content_block_delta",
                        delta=SimpleNamespace(type="text_delta", text="hello"),
                    )
                    yield SimpleNamespace(
                        type="message_delta",
                        delta=SimpleNamespace(stop_reason="end_turn"),
                        usage=SimpleNamespace(output_tokens=7),
                    )

                async def get_final_message(self_inner):
                    return SimpleNamespace(usage=final_usage)

            return _Ctx()

        self.messages = SimpleNamespace(stream=stream)


async def test_anthropic_stream_surfaces_cache_tokens():
    client = _FakeAnthropicStream(
        SimpleNamespace(
            input_tokens=42,
            cache_creation_input_tokens=100,
            cache_read_input_tokens=900,
        )
    )
    provider = AnthropicProvider(client=client)
    events = [
        e
        async for e in provider.stream(CONVO, config=ProviderConfig(model="claude-sonnet-4-5"))
    ]
    done = events[-1]
    assert isinstance(done, MessageDone)
    assert done.usage.input_tokens == 42
    assert done.usage.output_tokens == 7
    assert done.usage.cache_creation_input_tokens == 100
    assert done.usage.cache_read_input_tokens == 900


async def test_anthropic_stream_cache_tokens_absent_is_none():
    client = _FakeAnthropicStream(SimpleNamespace(input_tokens=42))
    provider = AnthropicProvider(client=client)
    events = [
        e
        async for e in provider.stream(CONVO, config=ProviderConfig(model="claude-sonnet-4-5"))
    ]
    done = events[-1]
    assert done.usage.cache_creation_input_tokens is None
    assert done.usage.cache_read_input_tokens is None


class _FakeOpenAIStream:
    """Mimics AsyncOpenAI .chat.completions.create(stream=True) -> async iterator."""

    def __init__(self):
        async def create(**kwargs):
            async def gen():
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="hello", tool_calls=None),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                )
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=None, tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=42, completion_tokens=7),
                )

            return gen()

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


async def test_openai_stream_leaves_cache_tokens_none():
    provider = OpenAIProvider(client=_FakeOpenAIStream())
    events = [e async for e in provider.stream(CONVO, config=ProviderConfig(model="gpt-4o-mini"))]
    done = events[-1]
    assert isinstance(done, MessageDone)
    assert done.usage.input_tokens == 42
    assert done.usage.cache_creation_input_tokens is None
    assert done.usage.cache_read_input_tokens is None
