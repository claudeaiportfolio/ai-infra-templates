"""The swap test (network-free arm).

Same scripted conversation through both providers with fake clients; assert the
*neutral* result is identical. This proves the seam yields the same behaviour
regardless of provider — "the same call passes on both arms", not "the interface
compiles". The live arm (real eval green on both LLM_PROVIDER values) runs in the
consuming repo's groundedness eval; this is the deterministic CI guarantee.
"""

from types import SimpleNamespace

from llm_provider import Message, ProviderConfig
from llm_provider.anthropic_provider import AnthropicProvider
from llm_provider.openai_provider import OpenAIProvider


class _FakeAnthropic:
    """Mimics AsyncAnthropic: .messages.create(**kwargs) -> response object."""

    def __init__(self, text: str):
        self._text = text

        async def create(**kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=self._text)],
                stop_reason="end_turn",
                model=kwargs["model"],
                usage=SimpleNamespace(input_tokens=42, output_tokens=7),
            )

        self.messages = SimpleNamespace(create=create)


class _FakeOpenAI:
    """Mimics AsyncOpenAI: .chat.completions.create(**kwargs) -> response."""

    def __init__(self, text: str):
        async def create(**kwargs):
            msg = SimpleNamespace(content=text, tool_calls=[])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                model=kwargs["model"],
                usage=SimpleNamespace(prompt_tokens=42, completion_tokens=7),
            )

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


CONVO = [Message(role="user", content="What does the policy say? Ground it in the chunks.")]


async def test_swap_same_behaviour_on_both_arms():
    claude = AnthropicProvider(client=_FakeAnthropic("grounded answer"))
    gpt = OpenAIProvider(client=_FakeOpenAI("grounded answer"))

    a = await claude.complete(CONVO, config=ProviderConfig(model="claude-sonnet-4-5"))
    o = await gpt.complete(CONVO, config=ProviderConfig(model="gpt-4o-mini"))

    assert a.text == o.text == "grounded answer"
    assert a.stop_reason == o.stop_reason == "end_turn"
    assert a.usage.input_tokens == o.usage.input_tokens == 42
    assert a.usage.output_tokens == o.usage.output_tokens == 7
    # Only the model id should differ between arms.
    assert a.model == "claude-sonnet-4-5"
    assert o.model == "gpt-4o-mini"
