# Changelog

All notable changes to `agent-core` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Releases
are tagged `agent-core-vX.Y.Z`.

## [0.2.0] - 2026-06-28

### Changed
- The agent loop now routes its provider call through the **`llm-provider`**
  seam (`get_provider()`, selected by the `LLM_PROVIDER` env var, default
  `anthropic`) instead of the raw Anthropic SDK directly. The loop is now
  provider-portable. Anthropic prompt caching is preserved via
  `ProviderConfig.extra` (`cache_control`), and cache tokens surface through the
  neutral `Usage`. **The public API is unchanged** (`AgentLoop`, `run`,
  `LoopResult` signatures/return shapes) and the anthropic path's behaviour is
  preserved.

### Added
- `MCPClient.tool_specs()` and `ToolSchema.to_spec()` — a provider-neutral
  `ToolSpec` accessor for the seam. `tools_for_anthropic()` is unchanged and
  still supported. The loop prefers `tool_specs()` and falls back to building
  specs from `tools_for_anthropic()`, so a consumer that duck-types `MCPClient`
  with only the Anthropic accessor keeps working.
- `AgentLoop.temperature` (default `1.0`). Forwarded on every provider request.
  The default matches the prior behaviour exactly: the loop used to omit
  `temperature`, so requests ran at the Anthropic API's server default of `1.0`.

### Dependencies
- Adds `llm-provider>=0.2.0`, pinned to the git-subdirectory tag
  `llm-provider-v0.2.0` (cut by a human post-merge). `anthropic` stays an
  explicit dependency because the loop constructs an `AsyncAnthropic` client
  directly to preserve the injected-API-key path.

### Backward compatibility
- Public API (`AgentLoop` fields/`run`/`LoopResult`) is unchanged; new fields
  are additive with behaviour-preserving defaults.
- The anthropic path is behaviour-preserving. Two scoped normalisations follow
  from routing through the neutral seam, both irrelevant to the normal
  end_turn/tool_use/max_turns/bail flow: (1) **abnormal** stop reasons are
  mapped to the neutral set (`refusal`/`pause_turn` → `other`, `stop_sequence` →
  `stop`); end_turn/tool_use/max_tokens are unchanged. (2) the final answer
  joins multiple text blocks with `""` rather than `"\n"` (assistant turns emit
  a single text block in practice).
- The only consumer, legacy `rag-ingestion-platform`, is pinned to
  `agent-core-v0.1.2`; tag-pinning means it is unaffected until it re-pins.

## [0.1.2] - 2026-06

### Added
- Initial published line: hand-written bounded plan/act/observe loop
  (`AgentLoop`), `MCPClient` (streamable-HTTP MCP + Auth0 M2M), `Tracer`
  (JSONL + OTel GenAI/MCP spans), `SkillLoader` progressive disclosure,
  Anthropic prompt caching with cache-token accounting, and direct
  API-key injection.

[0.2.0]: https://github.com/claudeaiportfolio/ai-infra-templates/releases/tag/agent-core-v0.2.0
[0.1.2]: https://github.com/claudeaiportfolio/ai-infra-templates/releases/tag/agent-core-v0.1.2
