# llm-provider

A **thin** provider seam: one `LLMProvider` protocol with **Claude** and
**OpenAI** implementations. Swap providers with a single config value —
`LLM_PROVIDER=anthropic|openai`.

> "I abstracted the loop's dependency on the provider — messages, the tool
> round-trip, the stream contract — **not** the providers' differentiating
> features. Abstracting away prompt caching or structured outputs would cost more
> than the lock-in it saves."

## The boundary (what this does and doesn't abstract)

**Abstracted** (the real coupling, where the SDKs diverge):

- **messages** — text and the tool round-trip. Claude carries `tool_use` /
  `tool_result` as content blocks inside user/assistant turns; OpenAI splits it
  into a `tool_calls` array plus separate `role:"tool"` messages. The neutral
  `Message` / `ToolUseBlock` / `ToolResultBlock` types insulate the consumer.
- **the completion** — text, tool calls, stop reason, and **token usage** (so a
  caller can build a cost panel without provider-specific plumbing).
- **the stream-event contract** — `TextDelta` / `ToolUseDelta` / `MessageDone`.

**Passed through opaque** (provider-*differentiating* config — never abstracted):
model name, `max_tokens`, `temperature`, and a `ProviderConfig.extra` bag. The
Claude impl reads its own keys (worked example: `cache_control` → cached system
prefix); the OpenAI impl ignores them and reads its own (e.g. `response_format`).

## Use

```python
from llm_provider import get_provider, Message, ProviderConfig

provider = get_provider()  # reads LLM_PROVIDER (default: anthropic)

completion = await provider.complete(
    [Message(role="user", content="Answer only from these chunks: …")],
    config=ProviderConfig(model="claude-sonnet-4-5", system="You are a grounded assistant."),
)
print(completion.text, completion.usage.input_tokens, completion.usage.output_tokens)
```

`Usage` also carries optional prompt-caching counts —
`cache_creation_input_tokens` and `cache_read_input_tokens` (both `int | None`).
The Claude impl populates them from the SDK; the OpenAI impl leaves them `None`
(`None` = "not reported", distinct from `0` = "no cache activity"), so a caller
can build a cache-aware cost panel without provider-specific plumbing.

Streaming yields `StreamEvent`s ending in `MessageDone`:

```python
async for event in provider.stream(messages, config=cfg):
    ...
```

## The proof the swap is real

`tests/test_swap.py` runs the **same** scripted conversation through both
providers and asserts an **identical neutral result** — "the same call passes on
both arms", not "the interface compiles". The live arm (a real groundedness eval
green under both `LLM_PROVIDER=anthropic` and `=openai`) runs in the consuming
repo (`rag-retrieval-service`), where the seam powers the single answer-generation
call.

## Consume (pinned, git-subdirectory)

```toml
dependencies = [
    "llm-provider @ git+https://github.com/claudeaiportfolio/ai-infra-templates.git@llm-provider-v0.2.0#subdirectory=llm-provider",
]
```

## Develop

```bash
uv sync --extra dev
uv run ruff check src tests
uv run mypy src
uv run pytest          # live-marked tests are skipped without API keys
```
