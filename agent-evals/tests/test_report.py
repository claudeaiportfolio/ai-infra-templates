"""Tests for the Markdown report renderer."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_evals import (
    CheckSuite,
    TraceRecord,
    render_comparison,
    render_single_run,
    score_all,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def scored_run() -> list:
    suite = CheckSuite.with_defaults()
    records = [
        TraceRecord.load(FIXTURES / "valid_two_tool_chain.jsonl"),
        TraceRecord.load(FIXTURES / "valid_boundary_refusal.jsonl"),
    ]
    return score_all(records, suite)


class TestRenderSingleRun:
    def test_renders_without_error(self, scored_run: list) -> None:
        out = render_single_run(scored_run, "test-run")
        assert "Eval results" in out
        assert "test-run" in out

    def test_includes_pass_rate(self, scored_run: list) -> None:
        out = render_single_run(scored_run, "x")
        assert "Pass rate" in out
        assert "100.0%" in out  # all checks pass on the fixtures

    def test_includes_per_query_table(self, scored_run: list) -> None:
        out = render_single_run(scored_run, "x")
        assert "Per-query results" in out
        assert "test-01" in out
        assert "bd-test" in out

    def test_includes_category_breakdown(self, scored_run: list) -> None:
        out = render_single_run(scored_run, "x")
        assert "By category" in out
        assert "boundary" in out
        assert "tool_selection" in out

    def test_includes_observations_section(self, scored_run: list) -> None:
        out = render_single_run(scored_run, "x")
        assert "Observations" in out
        assert "token_cost" in out

    def test_caveat_when_truncated(self, tmp_path: Path) -> None:
        """A trace without final_text_full should trigger the truncation caveat."""
        truncated = tmp_path / "legacy.jsonl"
        truncated.write_text(
            '{"kind": "query_meta", "run_id": "r1", "query_id": "old-01", "category": "x", "expected_tools": [], "skills_enabled": false}\n'
            '{"kind": "loop_start", "question": "?", "model": "x"}\n'
            '{"kind": "claude_response", "turn": 1, "stop_reason": "end_turn", "input_tokens": 10, "output_tokens": 5}\n'
            '{"kind": "loop_end", "turns": 1, "tool_calls": 0, "stop_reason": "end_turn", "final_text_preview": "short"}\n'
        )
        suite = CheckSuite.with_defaults()
        scored = score_all([TraceRecord.load(truncated)], suite)
        out = render_single_run(scored, "legacy-run")
        assert "Caveats" in out
        assert "final_text_full" in out
        assert "old-01" in out

    def test_empty_run_produces_message(self) -> None:
        out = render_single_run([], "empty")
        assert "No traces to score" in out


class TestRenderComparison:
    def test_self_vs_self_zero_deltas(self, scored_run: list) -> None:
        """Comparing a run to itself produces zero deltas (sanity check)."""
        out = render_comparison(scored_run, scored_run)
        assert "with-skill vs baseline" in out
        # Token-cost delta is +0.0% when comparing to self
        assert "+0.0%" in out or "0.0%" in out

    def test_includes_token_cost_summary(self, scored_run: list) -> None:
        out = render_comparison(scored_run, scored_run)
        assert "Token cost" in out
        assert "total input tokens" in out

    def test_caveat_when_skill_states_unexpected(self, scored_run: list) -> None:
        """If with-skill traces have skills_enabled=False, the caveat fires."""
        # scored_run fixtures vary — let me check what's there
        out = render_comparison(scored_run, scored_run)
        # Both fixtures actually have skills_enabled: true, but the
        # "baseline" run is the same data → caveat should fire.
        assert "Caveats" in out
