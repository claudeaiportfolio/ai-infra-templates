"""Tests for the Layer 2 LLM-judge module.

All tests use a stub caller — no Anthropic API calls.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_evals import TraceRecord
from agent_evals.judge import (
    JudgeCriterion,
    JudgeResult,
    JudgedRun,
    parse_judge_response,
    render_judge_comparison,
    render_judge_run,
    render_trace_for_judge,
    run_judge,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def chain_record() -> TraceRecord:
    return TraceRecord.load(FIXTURES / "valid_two_tool_chain.jsonl")


@pytest.fixture
def boundary_record() -> TraceRecord:
    return TraceRecord.load(FIXTURES / "valid_boundary_refusal.jsonl")


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------


def test_parse_binary_pass() -> None:
    score, reasoning = parse_judge_response(
        "SCORE: 1\nREASONING: It satisfied the criterion clearly.",
        scale="binary",
    )
    assert score == 1
    assert reasoning == "It satisfied the criterion clearly."


def test_parse_binary_fail() -> None:
    score, reasoning = parse_judge_response(
        "SCORE: 0\nREASONING: Did not meet bar.", scale="binary"
    )
    assert score == 0
    assert "Did not meet bar" in reasoning


def test_parse_rubric() -> None:
    score, _ = parse_judge_response(
        "SCORE: 2\nREASONING: Meets bar.", scale="rubric_0_3"
    )
    assert score == 2


def test_parse_clamps_out_of_range_binary() -> None:
    # Some judges return 2 for binary by mistake — we clamp rather than fail.
    score, _ = parse_judge_response("SCORE: 2\nREASONING: x", scale="binary")
    assert score == 1


def test_parse_clamps_negative() -> None:
    score, _ = parse_judge_response("SCORE: -1\nREASONING: x", scale="binary")
    assert score == 0


def test_parse_unparseable_response() -> None:
    score, reasoning = parse_judge_response("No score here.", scale="binary")
    assert score == 0
    assert "unparseable" in reasoning


def test_parse_multiline_reasoning() -> None:
    text = """SCORE: 1
REASONING: First sentence.
Second sentence still part of reasoning."""
    _, reasoning = parse_judge_response(text, scale="binary")
    assert "First sentence" in reasoning
    assert "Second sentence" in reasoning


# ----------------------------------------------------------------------------
# Render trace for judge
# ----------------------------------------------------------------------------


def test_render_trace_includes_question_and_tools(chain_record: TraceRecord) -> None:
    rendered = render_trace_for_judge(chain_record)
    assert chain_record.question in rendered
    assert "search_db" in rendered
    assert "format_results" in rendered
    assert "Final answer" in rendered


def test_render_trace_boundary_marks_no_tools(boundary_record: TraceRecord) -> None:
    rendered = render_trace_for_judge(boundary_record)
    assert "boundary refusal or no-tools query" in rendered


# ----------------------------------------------------------------------------
# Criterion filtering
# ----------------------------------------------------------------------------


def test_criterion_applies_to_query_id_filter(chain_record: TraceRecord) -> None:
    targeted = JudgeCriterion(
        name="x",
        prompt="...",
        applies_to_query_id=("test-01",),
    )
    other = JudgeCriterion(
        name="y",
        prompt="...",
        applies_to_query_id=("not-this-one",),
    )
    assert targeted.applies_to(chain_record)
    assert not other.applies_to(chain_record)


def test_criterion_applies_to_category_filter(chain_record: TraceRecord) -> None:
    matching = JudgeCriterion(
        name="x",
        prompt="...",
        applies_to_category=("tool_selection",),
    )
    other = JudgeCriterion(
        name="y",
        prompt="...",
        applies_to_category=("output_synthesis",),
    )
    assert matching.applies_to(chain_record)
    assert not other.applies_to(chain_record)


def test_criterion_unfiltered_applies_everywhere(chain_record: TraceRecord) -> None:
    c = JudgeCriterion(name="x", prompt="...")
    assert c.applies_to(chain_record)


# ----------------------------------------------------------------------------
# Orchestration with stub caller
# ----------------------------------------------------------------------------


def _make_caller(responses: list[str]):
    """Return an async caller that yields canned responses in order."""
    iterator = iter(responses)

    async def _call(_system: str, _user: str, _model: str) -> str:
        try:
            return next(iterator)
        except StopIteration as e:
            raise RuntimeError("stub caller exhausted") from e

    return _call


def test_run_judge_single_record_single_criterion(chain_record: TraceRecord) -> None:
    criterion = JudgeCriterion(name="placeholder", prompt="Always pass.")
    caller = _make_caller(["SCORE: 1\nREASONING: OK."])

    judged = asyncio.run(
        run_judge(
            records=[chain_record],
            criteria=[criterion],
            label="test-run",
            caller=caller,
        )
    )

    assert isinstance(judged, JudgedRun)
    assert judged.label == "test-run"
    assert len(judged.results) == 1
    r = judged.results[0]
    assert r.score == 1
    assert r.max_score == 1
    assert r.record_query_id == "test-01"


def test_run_judge_skips_non_applicable_criteria(
    chain_record: TraceRecord, boundary_record: TraceRecord
) -> None:
    # criterion only applies to chain record
    chain_only = JudgeCriterion(
        name="chain_only",
        prompt="...",
        applies_to_query_id=("test-01",),
    )
    caller = _make_caller(["SCORE: 1\nREASONING: x"])

    judged = asyncio.run(
        run_judge(
            records=[chain_record, boundary_record],
            criteria=[chain_only],
            caller=caller,
        )
    )

    # Only one call should happen — the one matching test-01.
    assert len(judged.results) == 1
    assert judged.results[0].record_query_id == "test-01"


def test_run_judge_records_caller_failures_as_score_zero(
    chain_record: TraceRecord,
) -> None:
    criterion = JudgeCriterion(name="any", prompt="...")

    async def _failing(_s: str, _u: str, _m: str) -> str:
        raise RuntimeError("simulated API failure")

    judged = asyncio.run(
        run_judge(
            records=[chain_record],
            criteria=[criterion],
            caller=_failing,
        )
    )

    assert len(judged.results) == 1
    r = judged.results[0]
    assert r.score == 0
    assert "simulated API failure" in r.reasoning


def test_run_judge_accepts_scored_records(chain_record: TraceRecord) -> None:
    """ScoredRecord wrappers should work transparently."""
    from agent_evals import CheckSuite, score_all

    scored = score_all([chain_record], CheckSuite.with_defaults())
    criterion = JudgeCriterion(name="any", prompt="...")
    caller = _make_caller(["SCORE: 1\nREASONING: x"])

    judged = asyncio.run(
        run_judge(records=scored, criteria=[criterion], caller=caller)
    )

    assert len(judged.results) == 1
    assert judged.results[0].record_query_id == "test-01"


def test_run_judge_rubric_scale(chain_record: TraceRecord) -> None:
    criterion = JudgeCriterion(name="x", prompt="...", scale="rubric_0_3")
    caller = _make_caller(["SCORE: 3\nREASONING: Exceeds bar."])

    judged = asyncio.run(
        run_judge(records=[chain_record], criteria=[criterion], caller=caller)
    )
    r = judged.results[0]
    assert r.score == 3
    assert r.max_score == 3
    assert r.normalised == 1.0


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------


def test_mean_by_criterion_normalises() -> None:
    judged = JudgedRun(
        label="x",
        model="claude-haiku-4-5",
        results=(
            JudgeResult("q1", "crit", 1, 1, ""),
            JudgeResult("q2", "crit", 0, 1, ""),
            JudgeResult("q1", "rubric", 3, 3, ""),
            JudgeResult("q2", "rubric", 0, 3, ""),
        ),
    )
    means = judged.mean_by_criterion()
    assert means["crit"] == 0.5
    assert means["rubric"] == 0.5


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------


def test_render_judge_run_includes_means_and_per_query() -> None:
    judged = JudgedRun(
        label="with-skill",
        model="claude-haiku-4-5",
        results=(
            JudgeResult("q1", "crit_a", 1, 1, "good"),
            JudgeResult("q1", "crit_b", 2, 3, "ok"),
            JudgeResult("q2", "crit_a", 0, 1, "bad"),
            JudgeResult("q2", "crit_b", 3, 3, "great"),
        ),
    )
    out = render_judge_run(judged)
    assert "with-skill" in out
    assert "claude-haiku-4-5" in out
    assert "crit_a" in out
    assert "crit_b" in out
    assert "0.50" in out  # crit_a mean
    assert "0.83" in out  # crit_b mean
    assert "q1" in out
    assert "q2" in out
    assert "Reasoning" in out


def test_render_judge_comparison_shows_delta() -> None:
    ws = JudgedRun(
        label="with-skill",
        model="claude-haiku-4-5",
        results=(
            JudgeResult("q1", "crit_a", 1, 1, "..."),
            JudgeResult("q1", "crit_b", 3, 3, "..."),
        ),
    )
    bl = JudgedRun(
        label="baseline",
        model="claude-haiku-4-5",
        results=(
            JudgeResult("q1", "crit_a", 0, 1, "..."),
            JudgeResult("q1", "crit_b", 1, 3, "..."),
        ),
    )
    out = render_judge_comparison(ws, bl)
    assert "with-skill vs baseline" in out
    assert "crit_a" in out
    assert "crit_b" in out
    # delta for crit_a is +1.00, crit_b is +0.67
    assert "+1.00" in out
    assert "+0.67" in out
