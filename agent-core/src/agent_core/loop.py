"""The agent loop.

This is intentionally a manual tool loop. No agent framework — every decision
(when to stop, how to handle errors, what to log) is visible. The architecture
round will probe these details; they should not be hidden behind a library.

The provider call goes through the ``llm-provider`` seam (``get_provider()``,
``LLM_PROVIDER`` env, default ``anthropic``) rather than the raw Anthropic SDK,
so the loop is provider-portable. Provider-*differentiating* config (Anthropic
prompt caching) rides through ``ProviderConfig.extra``; the neutral ``Usage``
surfaces cache tokens. The loop shape, stop handling, and tracing are unchanged
— on the anthropic path the behaviour is preserved.

Loop shape (one of the two canonical patterns), now in neutral types:

    messages = [Message(role="user", content=question)]
    while True:
        completion = await provider.complete(messages, tools=tools, config=config)
        messages.append(Message(role="assistant", content=[*text, *completion.tool_calls]))

        if completion.stop_reason == "end_turn":
            return final_text
        if completion.stop_reason != "tool_use":
            return f"unexpected stop_reason: {completion.stop_reason}"

        results = []
        for call in completion.tool_calls:
            result = await call_tool(call.name, call.input)
            results.append(ToolResultBlock(tool_use_id=call.id, content=result))
        messages.append(Message(role="tool", content=results))

Key choices made and the reasoning:

  - max_turns is a hard cap. Without one, a misbehaving prompt can spin.
    8 is generous for this domain; if the agent needs more it's a bug.
  - Tool errors are returned to the model as tool_result blocks with
    is_error=True, not raised. The model gets one shot to recover. A
    second consecutive error from the same tool bails out — common AI
    portfolio anti-pattern is to retry indefinitely.
  - All requests, responses, and tool calls are traced. The trace
    file is the input to the eval suite next weekend.
  - Every chat completion and tool invocation opens an OTel span with
    GenAI/MCP semantic-convention attributes (gen_ai.provider.name,
    gen_ai.request.model, gen_ai.usage.* on chat spans; mcp.method.name,
    gen_ai.tool.name on agent-side MCP client spans). The MCP server's
    tools.py already emits server-side execute_tool spans; the agent-side
    client span closes the trace shape so a customer's APM can join
    the agent and server traces by trace ID.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from llm_provider import (
    AnthropicProvider,
    LLMProvider,
    Message,
    ProviderConfig,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    get_provider,
)
from opentelemetry import trace as otel_trace

from agent_core.telemetry import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GENAI_PROMPT_VERSION,
    LLM_TOOL_CALLS_MADE,
    get_tracer,
)

from .mcp_client import MCPClient
from .skills import SkillLoader
from .tracing import Tracer

_tracer = get_tracer(__name__)

# Generic default — consumers pass their own (e.g. rag_agent's planner/critic prompt).
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use them when needed; answer concisely and accurately, grounded in tool results."
)

# How much of each tool result to keep in the JSONL trace. Large enough that an
# eval judge can score groundedness against the actual tool output.
TOOL_RESULT_PREVIEW_CHARS = 8000


@dataclass
class LoopResult:
    final_text: str
    turns: int
    tool_calls: int
    stop_reason: str
    trace_path: str
    prompt_version: str


@dataclass
class AgentLoop:
    mcp: MCPClient
    tracer: Tracer
    model: str = "claude-sonnet-4-5"
    max_turns: int = 8
    max_tokens_per_turn: int = 4096
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    prompt_version: str = "v1"
    # When set, the loader composes the full system prompt by appending
    # the skill inventory (always) and selected skill bodies (per query).
    # When None, the bare `system_prompt` is used unchanged — useful for
    # without-skill eval baselines.
    skill_loader: SkillLoader | None = None
    # Anthropic prompt caching on the system prompt block. Caching is the
    # cost mitigation for the +168% input-token finding (see EVAL_FINDINGS);
    # an uncached A/B arm is the natural follow-up. False here gives the
    # apples-to-apples uncached baseline. Default True is the recommended
    # production posture.
    enable_prompt_caching: bool = True
    # Sampling temperature, forwarded on every provider request. Default 1.0
    # matches the prior behaviour exactly: the loop used to omit `temperature`
    # on `messages.create`, so the request ran at the Anthropic API's server
    # default of 1.0 (also OpenAI's default). Routing through the seam means the
    # value is now sent explicitly; keeping it at 1.0 preserves byte-identical
    # behaviour while letting a caller override it.
    temperature: float = 1.0
    # Inject the Anthropic key directly (e.g. fetched from a secrets manager)
    # instead of forcing it through the process environment. Falls back to
    # ANTHROPIC_API_KEY when None, which suits local/dev and CI eval runs.
    # Applies to the anthropic provider path; other providers read their own env.
    api_key: str | None = None
    _consecutive_errors_per_tool: dict[str, int] = field(default_factory=dict)

    def _make_provider(self) -> LLMProvider:
        """Select the provider via ``LLM_PROVIDER`` (default ``anthropic``).

        On the anthropic path the SDK client is built with the injected
        ``api_key`` (falling back to ``ANTHROPIC_API_KEY``) so the
        secrets-manager injection feature is preserved without forcing the key
        through the process environment — byte-identical to the prior client
        construction. Other providers are constructed by the factory and read
        their own credentials from the environment.
        """
        name = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        if name == "anthropic":
            from anthropic import AsyncAnthropic

            return AnthropicProvider(
                client=AsyncAnthropic(
                    api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY")
                )
            )
        return get_provider(name)

    def _tool_specs(self) -> list[ToolSpec]:
        """Neutral tool list for the provider. Prefers the MCP client's neutral
        ``tool_specs()`` accessor; falls back to building specs from
        ``tools_for_anthropic()`` so a consumer that duck-types ``MCPClient``
        with only the Anthropic accessor (e.g. legacy ``rag-ingestion-platform``)
        keeps working unchanged."""
        tool_specs = getattr(self.mcp, "tool_specs", None)
        if callable(tool_specs):
            return tool_specs()
        return [
            ToolSpec(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("input_schema")
                or {"type": "object", "properties": {}},
            )
            for t in self.mcp.tools_for_anthropic()
        ]

    async def run(self, user_question: str) -> LoopResult:
        provider = self._make_provider()
        tool_specs = self._tool_specs()

        # Compose the system prompt once per query. The skill loader (if set)
        # appends the always-on skill inventory and any skill bodies its
        # selector matches. The composed prompt is stable for the full loop;
        # we don't re-compose per turn because (a) it's the same question
        # throughout and (b) we want the model to see consistent guidance
        # across turns.
        if self.skill_loader is not None:
            composed_system_prompt = self.skill_loader.compose_system_prompt(
                self.system_prompt, user_question
            )
            loaded_skill_names = [
                s.name for s in self.skill_loader.selected_bodies(user_question)
            ]
        else:
            composed_system_prompt = self.system_prompt
            loaded_skill_names = []

        self.tracer.event(
            "loop_start",
            question=user_question,
            model=self.model,
            prompt_version=self.prompt_version,
            tools=[t.name for t in tool_specs],
            max_turns=self.max_turns,
            skills_loaded=loaded_skill_names,
            system_prompt_chars=len(composed_system_prompt),
            prompt_caching_enabled=self.enable_prompt_caching,
        )

        # Provider-neutral conversation history. Each turn we hand the full list
        # to the provider, which translates it to the active SDK's wire format
        # (Anthropic content blocks / OpenAI tool_calls + role:"tool" messages).
        messages: list[Message] = [Message(role="user", content=user_question)]
        turn = 0
        tool_call_count = 0
        final_text = ""
        stop_reason = "unknown"

        while turn < self.max_turns:
            turn += 1
            self.tracer.event("turn_start", turn=turn)

            self.tracer.event(
                "claude_request",
                turn=turn,
                message_count=len(messages),
            )
            # OTel GenAI agent-span for the chat completion. The span name
            # follows the spec's "{operation} {model}" convention.
            with _tracer.start_as_current_span(f"chat {self.model}") as chat_span:
                chat_span.set_attribute(GEN_AI_OPERATION_NAME, "chat")
                chat_span.set_attribute(GEN_AI_PROVIDER_NAME, provider.name)
                chat_span.set_attribute(GEN_AI_REQUEST_MODEL, self.model)
                chat_span.set_attribute(
                    GEN_AI_REQUEST_MAX_TOKENS, self.max_tokens_per_turn
                )
                chat_span.set_attribute(GENAI_PROMPT_VERSION, self.prompt_version)
                # Prompt caching is provider-*differentiating* config: it rides
                # through ProviderConfig.extra as the opaque `cache_control` flag
                # the Anthropic impl reads (it builds the cached system block;
                # the OpenAI impl pops and ignores it). When caching is off the
                # Anthropic impl sends the bare-string system form — the exact
                # uncached A/B-arm path the loop ran before.
                config = ProviderConfig(
                    model=self.model,
                    max_tokens=self.max_tokens_per_turn,
                    temperature=self.temperature,
                    system=composed_system_prompt,
                    extra={"cache_control": True} if self.enable_prompt_caching else {},
                )
                chat_span.set_attribute(
                    "gen_ai.prompt.caching_enabled", self.enable_prompt_caching
                )
                try:
                    completion = await provider.complete(
                        messages, tools=tool_specs, config=config
                    )
                except Exception as exc:
                    chat_span.record_exception(exc)
                    chat_span.set_status(otel_trace.StatusCode.ERROR, str(exc))
                    raise

                chat_span.set_attribute(GEN_AI_RESPONSE_MODEL, completion.model)
                chat_span.set_attribute(
                    GEN_AI_USAGE_INPUT_TOKENS, completion.usage.input_tokens
                )
                chat_span.set_attribute(
                    GEN_AI_USAGE_OUTPUT_TOKENS, completion.usage.output_tokens
                )
                # Cache-token usage from the neutral Usage. The fields are
                # int | None (None = "not reported", e.g. on the OpenAI path or
                # an uncached Anthropic call); we coerce to 0 so the trace/span
                # shape is unchanged. Cache reads bill at 10% of the normal
                # input-token rate; cache creation bills at 125%. The eval
                # pipeline's cache-aware token-cost observer uses these to
                # compute the effective cost.
                cache_creation = completion.usage.cache_creation_input_tokens or 0
                cache_read = completion.usage.cache_read_input_tokens or 0
                chat_span.set_attribute("gen_ai.usage.cache_creation_input_tokens", cache_creation)
                chat_span.set_attribute("gen_ai.usage.cache_read_input_tokens", cache_read)
                chat_span.set_attribute(
                    GEN_AI_RESPONSE_FINISH_REASONS, [completion.stop_reason or "unknown"]
                )
                tool_uses_this_turn = len(completion.tool_calls)
                chat_span.set_attribute(LLM_TOOL_CALLS_MADE, tool_uses_this_turn)

            # Capture a preview of the text content (if any) for the trace.
            text_preview = completion.text[:300]

            self.tracer.event(
                "claude_response",
                turn=turn,
                stop_reason=completion.stop_reason,
                input_tokens=completion.usage.input_tokens,
                output_tokens=completion.usage.output_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                text_preview=text_preview,
            )

            # Persist the assistant turn into the neutral history: its text
            # (if any) followed by the tool-call blocks. The provider echoes
            # these back to the SDK in the right wire shape on the next turn.
            assistant_blocks: list[Any] = []
            if completion.text:
                assistant_blocks.append(TextBlock(text=completion.text))
            assistant_blocks.extend(completion.tool_calls)
            messages.append(Message(role="assistant", content=assistant_blocks))

            stop_reason = completion.stop_reason or "unknown"
            self.tracer.event("stop_reason", turn=turn, reason=stop_reason)

            if stop_reason == "end_turn":
                final_text = completion.text
                break

            if stop_reason != "tool_use":
                # max_tokens, refusal, pause_turn — surface and bail.
                # We deliberately do NOT auto-retry: a refusal or token cap
                # is signal, not noise, and the trace should show it.
                final_text = (
                    f"[loop ended with stop_reason={stop_reason}]\n" + completion.text
                )
                break

            # Execute every tool-call the model requested this turn.
            tool_results: list[ToolResultBlock] = []
            should_bail = False
            for block in completion.tool_calls:
                tool_call_count += 1
                tool_name = block.name
                tool_input = block.input or {}
                self.tracer.event(
                    "tool_use",
                    turn=turn,
                    tool=tool_name,
                    input=tool_input,
                    tool_use_id=block.id,
                )

                try:
                    raw_result = await self.mcp.call_tool(tool_name, tool_input)
                    is_error = raw_result.startswith("ERROR:")
                except Exception as e:  # noqa: BLE001 — surface everything to the model once
                    raw_result = f"ERROR: {type(e).__name__}: {e}"
                    is_error = True

                if is_error:
                    self._consecutive_errors_per_tool[tool_name] = (
                        self._consecutive_errors_per_tool.get(tool_name, 0) + 1
                    )
                    self.tracer.event(
                        "tool_error",
                        turn=turn,
                        tool=tool_name,
                        tool_use_id=block.id,
                        error=raw_result[:500],
                        consecutive=self._consecutive_errors_per_tool[tool_name],
                    )
                    if self._consecutive_errors_per_tool[tool_name] >= 2:
                        # Two consecutive errors from the same tool: bail.
                        should_bail = True
                else:
                    self._consecutive_errors_per_tool[tool_name] = 0
                    self.tracer.event(
                        "tool_result",
                        turn=turn,
                        tool=tool_name,
                        tool_use_id=block.id,
                        preview_len=len(raw_result),
                        # Carry enough of the result for downstream eval judges to
                        # score groundedness against the actual tool output (a
                        # 300-char snippet is too little for retrieval results).
                        preview=raw_result[:TOOL_RESULT_PREVIEW_CHARS],
                    )

                tool_results.append(
                    ToolResultBlock(
                        tool_use_id=block.id,
                        content=raw_result,
                        is_error=is_error,
                    )
                )

            # A neutral "tool" turn; the provider maps it to the SDK's result
            # shape (Anthropic: a user turn of tool_result blocks; OpenAI: one
            # role:"tool" message per result).
            messages.append(Message(role="tool", content=tool_results))

            if should_bail:
                final_text = (
                    "[loop bailed: two consecutive errors from the same tool — "
                    "investigate the tool, not the prompt]"
                )
                stop_reason = "bailed_on_repeated_errors"
                break

        else:
            # while-else: max_turns hit without breaking
            stop_reason = "max_turns"
            final_text = f"[loop ended: max_turns={self.max_turns} reached without end_turn]"

        self.tracer.event(
            "loop_end",
            turns=turn,
            tool_calls=tool_call_count,
            stop_reason=stop_reason,
            final_text_preview=final_text[:300],
            final_text_full=final_text,
        )
        return LoopResult(
            final_text=final_text,
            turns=turn,
            tool_calls=tool_call_count,
            stop_reason=stop_reason,
            trace_path=str(self.tracer.path),
            prompt_version=self.prompt_version,
        )
