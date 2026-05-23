"""Tests for Layer 1 deterministic checks."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_evals import (
    CheckResult,
    CheckSuite,
    DEFAULT_OBSERVATIONS,
    DEFAULT_UNIVERSAL,
    Outcome,
    TraceRecord,
    make_param_observer,
    score_all,
    score_record,
)
from agent_evals.deterministic import (
    check_boundary_no_tools,
    check_boundary_single_turn,
    check_no_tool_errors,
    check_stopped_cleanly,
    check_tool_count,
    check_tool_selection,
    check_tools_within_schema,
    observe_empty_results,
    observe_token_cost,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def two_tool_chain() -> TraceRecord:
    return TraceRecord.load(FIXTURES / "valid_two_tool_chain.jsonl")


@pytest.fixture
def boundary_refusal() -> TraceRecord:
    return TraceRecord.load(FIXTURES / "valid_boundary_refusal.jsonl")


@pytest.fixture
def same_tool_twice() -> TraceRecord:
    return TraceRecord.load(FIXTURES / "valid_same_tool_twice.jsonl")


# ----------------------------------------------------------------------------
# Universal pass/fail checks
# ----------------------------------------------------------------------------

class TestToolSelection:
    def test_exact_match_passes(self, two_tool_chain: TraceRecord) -> None:
        result = check_tool_selection(two_tool_chain)
        assert result.outcome == Outcome.PASS

    def test_boundary_no_tools_passes(self, boundary_refusal: TraceRecord) -> None:
        result = check_tool_selection(boundary_refusal)
        assert result.outcome == Outcome.PASS
        assert "boundary" in result.detail

    def test_wrong_tool_called_fails(self, two_tool_chain: TraceRecord) -> None:
        from dataclasses import replace
        wrong = replace(two_tool_chain, expected_tools=("totally_different_tool",))
        result = check_tool_selection(wrong)
        assert result.outcome == Outcome.FAIL


class TestToolCount:
    def test_matching_count_passes(self, two_tool_chain: TraceRecord) -> None:
        assert check_tool_count(two_tool_chain).outcome == Outcome.PASS

    def test_extra_tool_fails(self, two_tool_chain: TraceRecord) -> None:
        from dataclasses import replace
        wrong = replace(two_tool_chain, expected_tools=("search_db",))
        assert check_tool_count(wrong).outcome == Outcome.FAIL


class TestBoundary:
    def test_boundary_query_with_no_tools_passes(self, boundary_refusal: TraceRecord) -> None:
        assert check_boundary_no_tools(boundary_refusal).outcome == Outcome.PASS
        assert check_boundary_single_turn(boundary_refusal).outcome == Outcome.PASS

    def test_non_boundary_query_is_na(self, two_tool_chain: TraceRecord) -> None:
        assert check_boundary_no_tools(two_tool_chain).outcome == Outcome.NA
        assert check_boundary_single_turn(two_tool_chain).outcome == Outcome.NA


class TestStoppedCleanly:
    def test_end_turn_passes(self, two_tool_chain: TraceRecord) -> None:
        assert check_stopped_cleanly(two_tool_chain).outcome == Outcome.PASS

    def test_max_turns_reached_fails(self, two_tool_chain: TraceRecord) -> None:
        from dataclasses import replace
        stuck = replace(two_tool_chain, stop_reason="max_tokens")
        assert check_stopped_cleanly(stuck).outcome == Outcome.FAIL


class TestNoToolErrors:
    def test_clean_run_passes(self, two_tool_chain: TraceRecord) -> None:
        assert check_no_tool_errors(two_tool_chain).outcome == Outcome.PASS

    def test_errored_call_fails(self, two_tool_chain: TraceRecord) -> None:
        from dataclasses import replace
        errored_call = replace(two_tool_chain.tool_calls[0], error="connection_refused")
        broken = replace(
            two_tool_chain,
            tool_calls=(errored_call, two_tool_chain.tool_calls[1]),
        )
        result = check_no_tool_errors(broken)
        assert result.outcome == Outcome.FAIL
        assert "connection_refused" in result.detail


class TestToolsWithinSchema:
    def test_in_schema_passes(self, two_tool_chain: TraceRecord) -> None:
        assert check_tools_within_schema(two_tool_chain).outcome == Outcome.PASS

    def test_out_of_schema_fails(self, two_tool_chain: TraceRecord) -> None:
        from dataclasses import replace
        wrong = replace(two_tool_chain, available_tools=("only_this_tool",))
        result = check_tools_within_schema(wrong)
        assert result.outcome == Outcome.FAIL


# ----------------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------------

class TestObserveEmptyResults:
    def test_no_empties_returns_na(self, two_tool_chain: TraceRecord) -> None:
        assert observe_empty_results(two_tool_chain).outcome == Outcome.NA

    def test_empty_result_is_recorded(self, same_tool_twice: TraceRecord) -> None:
        # second call returned preview_len=0
        result = observe_empty_results(same_tool_twice)
        assert result.outcome == Outcome.OBSERVED
        assert "B" in result.detail  # the descriptor includes the query parameter


class TestObserveTokenCost:
    def test_records_totals(self, two_tool_chain: TraceRecord) -> None:
        result = observe_token_cost(two_tool_chain)
        assert result.outcome == Outcome.OBSERVED
        assert result.value == {
            "input": two_tool_chain.total_input_tokens,
            "output": two_tool_chain.total_output_tokens,
        }


# ----------------------------------------------------------------------------
# make_param_observer
# ----------------------------------------------------------------------------

class TestMakeParamObserver:
    def test_records_value_when_tool_called(self, two_tool_chain: TraceRecord) -> None:
        observer = make_param_observer("search_db", "limit")
        result = observer(two_tool_chain)
        assert result.outcome == Outcome.OBSERVED
        assert result.value == 10
        assert "limit=10" in result.detail

    def test_na_when_tool_not_called(self, two_tool_chain: TraceRecord) -> None:
        observer = make_param_observer("nonexistent_tool", "limit")
        result = observer(two_tool_chain)
        assert result.outcome == Outcome.NA

    def test_records_list_when_tool_called_multiple_times(self, same_tool_twice: TraceRecord) -> None:
        observer = make_param_observer("search_db", "query")
        result = observer(same_tool_twice)
        assert result.outcome == Outcome.OBSERVED
        assert result.value == ["A", "B"]

    def test_custom_observer_name(self, two_tool_chain: TraceRecord) -> None:
        observer = make_param_observer(
            "search_db", "limit", observer_name="search_limit"
        )
        result = observer(two_tool_chain)
        assert result.name == "search_limit"


# ----------------------------------------------------------------------------
# CheckSuite composition
# ----------------------------------------------------------------------------

class TestCheckSuiteWithDefaults:
    def test_defaults_only(self) -> None:
        suite = CheckSuite.with_defaults()
        assert suite.universal == DEFAULT_UNIVERSAL
        assert suite.observations == DEFAULT_OBSERVATIONS
        assert suite.query_specific == {}

    def test_extra_observations_append(self) -> None:
        extra = make_param_observer("foo", "bar")
        suite = CheckSuite.with_defaults(extra_observations=[extra])
        assert len(suite.observations) == len(DEFAULT_OBSERVATIONS) + 1
        assert suite.observations[-1] is extra

    def test_query_specific_attached(self) -> None:
        def my_check(r: TraceRecord) -> CheckResult:
            return CheckResult("my_check", Outcome.PASS, "ok")
        suite = CheckSuite.with_defaults(query_specific={"q1": my_check})
        assert suite.query_specific == {"q1": my_check}


# ----------------------------------------------------------------------------
# score_record / score_all integration
# ----------------------------------------------------------------------------

class TestScoring:
    def test_score_record_runs_all_checks(self, two_tool_chain: TraceRecord) -> None:
        suite = CheckSuite.with_defaults()
        results = score_record(two_tool_chain, suite)
        # 7 universal + 2 observations
        assert len(results) == len(DEFAULT_UNIVERSAL) + len(DEFAULT_OBSERVATIONS)

    def test_score_record_runs_query_specific_when_matching(
        self, two_tool_chain: TraceRecord
    ) -> None:
        def my_check(r: TraceRecord) -> CheckResult:
            return CheckResult("my_check", Outcome.PASS, "matched")

        suite = CheckSuite.with_defaults(query_specific={"test-01": my_check})
        results = score_record(two_tool_chain, suite)
        names = [r.name for r in results]
        assert "my_check" in names

    def test_score_record_skips_query_specific_when_not_matching(
        self, boundary_refusal: TraceRecord
    ) -> None:
        def my_check(r: TraceRecord) -> CheckResult:
            return CheckResult("my_check", Outcome.PASS, "shouldn't appear")

        suite = CheckSuite.with_defaults(query_specific={"test-01": my_check})
        results = score_record(boundary_refusal, suite)
        names = [r.name for r in results]
        assert "my_check" not in names

    def test_score_all_returns_list_in_order(
        self, two_tool_chain: TraceRecord, boundary_refusal: TraceRecord
    ) -> None:
        suite = CheckSuite.with_defaults()
        scored = score_all([two_tool_chain, boundary_refusal], suite)
        assert [s.record.query_id for s in scored] == ["test-01", "bd-test"]


class TestScoredRecordAggregates:
    def test_pass_rate_excludes_na_and_observed(self, two_tool_chain: TraceRecord) -> None:
        suite = CheckSuite.with_defaults()
        scored = score_all([two_tool_chain], suite)[0]
        # All pass/fail checks should pass; pass_rate is 1.0
        assert scored.pass_rate == 1.0
        # 3 NAs: 2 boundary checks (don't apply) + empty_results (no empties)
        assert scored.na_count == 3

    def test_pass_rate_zero_when_no_pass_fail_checks(self, two_tool_chain: TraceRecord) -> None:
        """A suite with only observations should have pass_rate == 0.0 (no denominator)."""
        suite = CheckSuite(universal=(), observations=DEFAULT_OBSERVATIONS, query_specific={})
        scored = score_all([two_tool_chain], suite)[0]
        assert scored.pass_rate == 0.0
        assert scored.pass_count == 0
        assert scored.fail_count == 0
