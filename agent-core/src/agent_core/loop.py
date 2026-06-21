"""The agent loop.

This is intentionally a manual tool loop on top of the raw Anthropic SDK.
No agent framework — every decision (when to stop, how to handle errors,
what to log) is visible. The architecture round will probe these details;
they should not be hidden behind a library.

Loop shape (one of the two canonical patterns):

    messages = [{"role": "user", "content": question}]
    while True:
        response = await client.messages.create(..., messages=messages, tools=tools)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return final_text
        if response.stop_reason != "tool_use":
            return f"unexpected stop_reason: {response.stop_reason}"

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = await call_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})

Key choices made and the reasoning:

  - max_turns is a hard cap. Without one, a misbehaving prompt can spin.
    8 is generous for this domain; if the agent needs more it's a bug.
  - Tool errors are returned to the model as tool_result blocks with
    is_error=True, not raised. The model gets one shot to recover. A
    second consecutive error from the same tool bails out — common AI
    portfolio anti-pattern is to retry indefinitely.
  - All Claude requests, responses, and tool calls are traced. The trace
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

from anthropic import AsyncAnthropic
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
    _consecutive_errors_per_tool: dict[str, int] = field(default_factory=dict)

    async def run(self, user_question: str) -> LoopResult:
        client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        tools = self.mcp.tools_for_anthropic()

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
            tools=[t["name"] for t in tools],
            max_turns=self.max_turns,
            skills_loaded=loaded_skill_names,
            system_prompt_chars=len(composed_system_prompt),
            prompt_caching_enabled=self.enable_prompt_caching,
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_question}
        ]
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
                chat_span.set_attribute(GEN_AI_PROVIDER_NAME, "anthropic")
                chat_span.set_attribute(GEN_AI_REQUEST_MODEL, self.model)
                chat_span.set_attribute(
                    GEN_AI_REQUEST_MAX_TOKENS, self.max_tokens_per_turn
                )
                chat_span.set_attribute(GENAI_PROMPT_VERSION, self.prompt_version)
                # Anthropic's API accepts either a bare string for `system`
                # or a list of typed content blocks. cache_control is only
                # available on blocks — strings cannot carry cache metadata.
                # For the uncached A/B arm we pass the string form, which
                # also exercises the code path a non-caching customer would
                # run.
                if self.enable_prompt_caching:
                    system_param: Any = [
                        {
                            "type": "text",
                            "text": composed_system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                else:
                    system_param = composed_system_prompt
                chat_span.set_attribute(
                    "gen_ai.prompt.caching_enabled", self.enable_prompt_caching
                )
                try:
                    response = await client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens_per_turn,
                        system=system_param,
                        tools=tools,  # type: ignore[arg-type]  # dict tool schemas are the right runtime shape
                        messages=messages,  # type: ignore[arg-type]  # dict message blocks are the right runtime shape
                    )
                except Exception as exc:
                    chat_span.record_exception(exc)
                    chat_span.set_status(otel_trace.StatusCode.ERROR, str(exc))
                    raise

                chat_span.set_attribute(GEN_AI_RESPONSE_MODEL, response.model)
                chat_span.set_attribute(
                    GEN_AI_USAGE_INPUT_TOKENS, response.usage.input_tokens
                )
                chat_span.set_attribute(
                    GEN_AI_USAGE_OUTPUT_TOKENS, response.usage.output_tokens
                )
                # Cache-token usage. The two fields are always set in the
                # response (zero when caching wasn't hit), so we record them
                # unconditionally. Cache reads bill at 10% of the normal
                # input-token rate; cache creation bills at 125%. The eval
                # pipeline's cache-aware token-cost observer uses these to
                # compute the effective cost.
                cache_creation = getattr(
                    response.usage, "cache_creation_input_tokens", 0
                ) or 0
                cache_read = getattr(
                    response.usage, "cache_read_input_tokens", 0
                ) or 0
                chat_span.set_attribute("gen_ai.usage.cache_creation_input_tokens", cache_creation)
                chat_span.set_attribute("gen_ai.usage.cache_read_input_tokens", cache_read)
                chat_span.set_attribute(
                    GEN_AI_RESPONSE_FINISH_REASONS, [response.stop_reason or "unknown"]
                )
                tool_uses_this_turn = sum(
                    1 for b in response.content if getattr(b, "type", None) == "tool_use"
                )
                chat_span.set_attribute(LLM_TOOL_CALLS_MADE, tool_uses_this_turn)

            # Capture a preview of the text content (if any) for the trace.
            text_chunks = [
                b.text for b in response.content
                if getattr(b, "type", None) == "text"
            ]
            text_preview = (" ".join(text_chunks))[:300]

            self.tracer.event(
                "claude_response",
                turn=turn,
                stop_reason=response.stop_reason,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
                text_preview=text_preview,
            )

            # Persist assistant turn into the message history exactly as
            # the API returned it. The Messages API requires the full
            # content blocks (including tool_use blocks) to be echoed
            # back; serialising to dicts is the safe way.
            messages.append(
                {
                    "role": "assistant",
                    "content": [b.model_dump() for b in response.content],
                }
            )

            stop_reason = response.stop_reason or "unknown"
            self.tracer.event("stop_reason", turn=turn, reason=stop_reason)

            if stop_reason == "end_turn":
                final_text = "\n".join(text_chunks)
                break

            if stop_reason != "tool_use":
                # max_tokens, refusal, pause_turn — surface and bail.
                # We deliberately do NOT auto-retry: a refusal or token cap
                # is signal, not noise, and the trace should show it.
                final_text = (
                    f"[loop ended with stop_reason={stop_reason}]\n"
                    + "\n".join(text_chunks)
                )
                break

            # Execute every tool_use block in this assistant turn.
            tool_results: list[dict[str, Any]] = []
            should_bail = False
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
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
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": raw_result,
                        "is_error": is_error,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

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
