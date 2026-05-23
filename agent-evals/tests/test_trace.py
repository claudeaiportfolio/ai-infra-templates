"""Tests for trace parsing."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_evals.trace import TraceRecord, load_directory

FIXTURES = Path(__file__).parent / "fixtures"


def test_two_tool_chain_parses() -> None:
    """The basic happy path: two distinct tool calls in two turns."""
    r = TraceRecord.load(FIXTURES / "valid_two_tool_chain.jsonl")
    assert r.query_id == "test-01"
    assert r.category == "tool_selection"
    assert r.expected_tools == ("search_db", "format_results")
    assert r.skills_enabled is True
    assert r.tools_called == ("search_db", "format_results")
    assert r.tool_call_count == 2
    assert r.turns == 3
    assert r.stop_reason == "end_turn"
    assert r.final_text_truncated is False
    assert "markdown" in r.final_text


def test_two_tool_chain_token_totals() -> None:
    r = TraceRecord.load(FIXTURES / "valid_two_tool_chain.jsonl")
    assert r.total_input_tokens == 1200 + 1450 + 1500
    assert r.total_output_tokens == 80 + 30 + 120


def test_two_tool_chain_tool_inputs() -> None:
    r = TraceRecord.load(FIXTURES / "valid_two_tool_chain.jsonl")
    search, fmt = r.tool_calls
    assert search.tool == "search_db"
    assert search.input == {"query": "X", "limit": 10}
    assert search.result_preview_len == 250
    assert search.tool_use_id == "toolu_a1"
    assert fmt.tool == "format_results"
    assert fmt.input == {"format": "markdown"}


def test_boundary_refusal_parses() -> None:
    r = TraceRecord.load(FIXTURES / "valid_boundary_refusal.jsonl")
    assert r.category == "boundary"
    assert r.expected_tools == ()
    assert r.tool_call_count == 0
    assert r.tools_called == ()
    assert r.turns == 1
    assert r.stop_reason == "end_turn"


def test_same_tool_twice_positional_matching() -> None:
    """When the same tool is called twice in one turn, positional matching
    must pair use[0] with result[0] and use[1] with result[1], NOT collapse
    both to whichever result happens to match first.
    """
    r = TraceRecord.load(FIXTURES / "valid_same_tool_twice.jsonl")
    assert r.tool_call_count == 2
    first, second = r.tool_calls
    assert first.input == {"query": "A"}
    assert first.result_preview_len == 100
    assert first.result_preview == "found A"
    assert second.input == {"query": "B"}
    assert second.result_preview_len == 0
    assert second.result_preview == ""


def test_missing_query_meta_raises(tmp_path: Path) -> None:
    bad = tmp_path / "broken.jsonl"
    bad.write_text(
        '{"kind": "loop_start", "question": "?", "model": "x"}\n'
        '{"kind": "loop_end", "turns": 0, "tool_calls": 0, "stop_reason": "end_turn"}\n'
    )
    with pytest.raises(ValueError, match="query_meta"):
        TraceRecord.load(bad)


def test_malformed_json_raises_with_line_context(tmp_path: Path) -> None:
    bad = tmp_path / "malformed.jsonl"
    bad.write_text(
        '{"kind": "query_meta", "run_id": "x", "query_id": "y", "category": "z"}\n'
        'not valid json\n'
    )
    with pytest.raises(ValueError, match="malformed JSON"):
        TraceRecord.load(bad)


def test_legacy_trace_without_final_text_full(tmp_path: Path) -> None:
    """Old traces only have final_text_preview; loader should flag truncation."""
    trace = tmp_path / "legacy.jsonl"
    trace.write_text(
        '{"kind": "query_meta", "run_id": "r1", "query_id": "q1", "category": "x", "expected_tools": [], "skills_enabled": false}\n'
        '{"kind": "loop_start", "question": "?", "model": "x"}\n'
        '{"kind": "claude_response", "turn": 1, "stop_reason": "end_turn", "input_tokens": 10, "output_tokens": 5}\n'
        '{"kind": "loop_end", "turns": 1, "tool_calls": 0, "stop_reason": "end_turn", "final_text_preview": "short answer"}\n'
    )
    r = TraceRecord.load(trace)
    assert r.final_text == "short answer"
    assert r.final_text_truncated is True


def test_load_directory_sorts_by_query_id() -> None:
    """Multiple files loaded together should come out sorted by query_id."""
    records = load_directory(FIXTURES)
    query_ids = [r.query_id for r in records]
    assert query_ids == sorted(query_ids)


def test_load_directory_skips_malformed(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    (tmp_path / "good.jsonl").write_text(
        '{"kind": "query_meta", "run_id": "r1", "query_id": "q1", "category": "x", "expected_tools": [], "skills_enabled": false}\n'
        '{"kind": "loop_start", "question": "?", "model": "x"}\n'
        '{"kind": "claude_response", "turn": 1, "stop_reason": "end_turn", "input_tokens": 10, "output_tokens": 5}\n'
        '{"kind": "loop_end", "turns": 1, "tool_calls": 0, "stop_reason": "end_turn"}\n'
    )
    (tmp_path / "bad.jsonl").write_text("not json\n")

    records = load_directory(tmp_path)
    assert len(records) == 1
    assert records[0].query_id == "q1"
    captured = capsys.readouterr()
    assert "bad.jsonl" in captured.err
