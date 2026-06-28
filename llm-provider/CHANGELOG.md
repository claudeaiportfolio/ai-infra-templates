# Changelog

All notable changes to `llm-provider` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Releases
are tagged `llm-provider-vX.Y.Z`.

## [0.2.0] - 2026-06-28

### Added
- `Usage` now carries optional prompt-caching token counts:
  `cache_creation_input_tokens` and `cache_read_input_tokens` (both
  `int | None`, default `None`). The `AnthropicProvider` populates them from the
  SDK usage object on both `complete()` and `stream()`; the `OpenAIProvider`
  leaves them `None` (OpenAI's automatic caching isn't surfaced as discrete
  token counts). `None` means "not reported" and stays distinct from `0`
  ("reported, no cache activity"), so a caller can build a cache-aware cost
  panel without provider-specific plumbing.

### Notes
- **Backward compatible.** The change is purely additive: existing `Usage`
  fields (`input_tokens`, `output_tokens`) are unchanged, and the new fields
  default to `None`, so existing consumers and serialised payloads keep working.

## [0.1.0] - 2026-06

### Added
- Initial release: the `LLMProvider` protocol with `AnthropicProvider` and
  `OpenAIProvider` implementations; neutral `Message` / `ToolUseBlock` /
  `ToolResultBlock` / `ToolSpec` / `Completion` / `Usage` types; the
  `TextDelta` / `ToolUseDelta` / `MessageDone` stream contract; and
  `get_provider()` selection via the `LLM_PROVIDER` env var. Provider-
  differentiating features (prompt caching, structured outputs) ride through
  `ProviderConfig.extra` as opaque config.

[0.2.0]: https://github.com/claudeaiportfolio/ai-infra-templates/releases/tag/llm-provider-v0.2.0
[0.1.0]: https://github.com/claudeaiportfolio/ai-infra-templates/releases/tag/llm-provider-v0.1.0
