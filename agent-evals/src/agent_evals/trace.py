"""Trace file parsing.

A TraceRecord is the canonical shape consumed by the eval pipeline. Each one
holds the result of a single agent run against a single query: the query
metadata, the ordered tool calls and their inputs, per-turn token usage,
the final answer, and the loop summary.

Built from a JSONL trace file with one event per line. The agent loop must
emit these event kinds:

    query_meta (1x)
        Required fields: run_id, query_id, category, expected_tools (list),
        skills_enabled (bool).
        Optional: note.

    loop_start (1x)
        Required fields: question, model.
        Optional: prompt_version, tools (list), max_turns, skills_loaded,
        system_prompt_chars.

    claude_response (Nx)
        Required fields: turn (int), stop_reason, input_tokens, output_tokens.

    tool_use (Nx) — emitted in lockstep with the matching tool_result/tool_error
        Required fields: turn, tool, input (dict).
        Optional: tool_use_id.

    tool_result (Nx) or tool_error (Nx)
        Required fields: turn, tool.
        For tool_result: preview, preview_len.
        For tool_error: error.

    loop_end (1x)
        Required fields: turns, tool_calls, stop_reason.
        Optional: final_text_full, final_text_preview.

The parser is forgiving — missing optional fields are filled with sensible
defaults. Records with missing required events raise ValueError.

The shared package documents this schema; projects that emit it differently
can write their own loader returning TraceRecord. The deterministic checks
and report renderer are insulated from the trace format and depend only on
TraceRecord.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation by the agent during a run."""

    turn: int
    tool: str
    input: dict[str, Any]
    tool_use_id: str | None
    result_preview: str | None
    result_preview_len: int | None
    error: str | None  # None unless tool_error was emitted


@dataclass(frozen=True)
class TurnUsage:
    """Token usage for one model response."""

    turn: int
    stop_reason: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class TraceRecord:
    """The eval pipeline's view of one agent run.

    All Layer 1 checks are pure functions over this record. Layer 2 checks
    will also live here once they land; the LLM-judge reads `final_text` for
    synthesis-quality scoring, which is why the `final_text_truncated` flag
    matters — older traces emitted only a 300-char preview.
    """

    # Provenance
    path: Path
    run_id: str

    # Query metadata (from query_meta event)
    query_id: str
    category: str
    expected_tools: tuple[str, ...]
    note: str
    skills_enabled: bool

    # Loop config (from loop_start event)
    question: str
    model: str
    prompt_version: str
    available_tools: tuple[str, ...]
    max_turns: int
    skills_loaded: tuple[str, ...]
    system_prompt_chars: int

    # Tool calls in order
    tool_calls: tuple[ToolCall, ...]

    # Per-turn model usage
    turn_usage: tuple[TurnUsage, ...]

    # Final outcome (from loop_end event)
    turns: int
    tool_call_count: int
    stop_reason: str
    final_text: str
    final_text_truncated: bool  # True if only preview was available

    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.turn_usage)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.turn_usage)

    @property
    def tools_called(self) -> tuple[str, ...]:
        """Tool names in the order they were called."""
        return tuple(tc.tool for tc in self.tool_calls)

    @classmethod
    def load(cls, path: Path) -> TraceRecord:
        """Parse a JSONL trace file into a TraceRecord.

        Raises ValueError on missing required events; tolerates missing
        optional fields (final_text_full pre-dates the field's addition).
        """
        events = _read_events(path)

        query_meta = _find_one(events, "query_meta", path)
        loop_start = _find_one(events, "loop_start", path)
        loop_end = _find_one(events, "loop_end", path)

        tool_calls = _build_tool_calls(events)
        turn_usage = _build_turn_usage(events)

        # final_text_full is the post-eval-pipeline field; older traces only
        # carry the 300-char preview. Both layers should know which they got.
        final_text_full = loop_end.get("final_text_full")
        if final_text_full is not None:
            final_text = final_text_full
            truncated = False
        else:
            final_text = loop_end.get("final_text_preview", "")
            truncated = True

        return cls(
            path=path,
            run_id=query_meta["run_id"],
            query_id=query_meta["query_id"],
            category=query_meta["category"],
            expected_tools=tuple(query_meta.get("expected_tools", ())),
            note=query_meta.get("note", ""),
            skills_enabled=bool(query_meta.get("skills_enabled", False)),
            question=loop_start["question"],
            model=loop_start["model"],
            prompt_version=loop_start.get("prompt_version", ""),
            available_tools=tuple(loop_start.get("tools", ())),
            max_turns=int(loop_start.get("max_turns", 0)),
            skills_loaded=tuple(loop_start.get("skills_loaded", ())),
            system_prompt_chars=int(loop_start.get("system_prompt_chars", 0)),
            tool_calls=tool_calls,
            turn_usage=turn_usage,
            turns=int(loop_end.get("turns", 0)),
            tool_call_count=int(loop_end.get("tool_calls", 0)),
            stop_reason=loop_end.get("stop_reason", ""),
            final_text=final_text,
            final_text_truncated=truncated,
        )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _read_events(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of event dicts.

    Skips blank lines; raises ValueError on malformed JSON with line context.
    """
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: malformed JSON: {e}") from e
    return events


def _find_one(events: list[dict[str, Any]], kind: str, path: Path) -> dict[str, Any]:
    """Find the first event of a given kind, raising if absent.

    Each trace file is expected to contain exactly one query_meta, one
    loop_start, and one loop_end. If any are missing the trace is malformed
    and Layer 1 checks can't run.
    """
    for ev in events:
        if ev.get("kind") == kind:
            return ev
    raise ValueError(f"{path}: missing required event kind={kind!r}")


def _build_tool_calls(events: list[dict[str, Any]]) -> tuple[ToolCall, ...]:
    """Pair tool_use events with their matching tool_result / tool_error.

    The agent loop emits tool_use and the corresponding tool_result/tool_error
    in lockstep, in the same order as the model's content blocks. The trace
    stream therefore looks like:
        tool_use(A) → tool_result(A) → tool_use(B) → tool_result(B)

    tool_result events do NOT carry `tool_use_id` in the current trace
    schema (only the tool_use event does). That means we can't match by id;
    we match positionally within each (turn, tool) bucket.

    Worth fixing upstream: adding `tool_use_id` to tool_result emission in
    the agent's tracer would make this matching robust to future re-ordering.
    For now, positional matching is safe given the loop's known structure.
    """
    out: list[ToolCall] = []

    pending: list[dict[str, Any]] = []

    def _pop_match(turn: Any, tool: str) -> dict[str, Any] | None:
        for i, u in enumerate(pending):
            if u.get("turn") == turn and u.get("tool") == tool:
                return pending.pop(i)
        return None

    pairs: list[tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]] = []
    use_index: dict[int, int] = {}

    for ev in events:
        kind = ev.get("kind")
        if kind == "tool_use":
            pending.append(ev)
            pairs.append((ev, None, None))
            use_index[id(ev)] = len(pairs) - 1
        elif kind == "tool_result":
            matched = _pop_match(ev.get("turn"), ev.get("tool"))
            if matched is not None:
                idx = use_index[id(matched)]
                u, _, e = pairs[idx]
                pairs[idx] = (u, ev, e)
        elif kind == "tool_error":
            matched = _pop_match(ev.get("turn"), ev.get("tool"))
            if matched is not None:
                idx = use_index[id(matched)]
                u, r, _ = pairs[idx]
                pairs[idx] = (u, r, ev)

    for use, result, error in pairs:
        turn = use.get("turn")
        out.append(
            ToolCall(
                turn=int(turn) if turn is not None else 0,
                tool=str(use.get("tool")),
                input=dict(use.get("input") or {}),
                tool_use_id=use.get("tool_use_id"),
                result_preview=(result or {}).get("preview"),
                result_preview_len=(result or {}).get("preview_len"),
                error=(error or {}).get("error"),
            )
        )
    return tuple(out)


def _build_turn_usage(events: list[dict[str, Any]]) -> tuple[TurnUsage, ...]:
    """Extract token usage per model response."""
    out: list[TurnUsage] = []
    for ev in events:
        if ev.get("kind") != "claude_response":
            continue
        out.append(
            TurnUsage(
                turn=int(ev.get("turn", 0)),
                stop_reason=str(ev.get("stop_reason", "")),
                input_tokens=int(ev.get("input_tokens", 0)),
                output_tokens=int(ev.get("output_tokens", 0)),
            )
        )
    return tuple(out)


def load_directory(trace_dir: Path) -> list[TraceRecord]:
    """Load every *.jsonl trace from a directory, sorted by query_id.

    Malformed traces are skipped with a warning to stderr — one bad file
    shouldn't kill scoring on the rest.
    """
    import sys

    records: list[TraceRecord] = []
    for p in sorted(trace_dir.glob("*.jsonl")):
        try:
            records.append(TraceRecord.load(p))
        except ValueError as e:
            print(f"WARN: skipping {p.name}: {e}", file=sys.stderr)
    records.sort(key=lambda r: r.query_id)
    return records
