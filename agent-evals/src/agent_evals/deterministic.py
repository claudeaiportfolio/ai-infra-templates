"""Layer 1 deterministic checks.

Pure functions over TraceRecord. No API calls, no LLM-judge, no I/O. Cheap
to run, reproducible, unambiguous.

Each check is a function returning a CheckResult. The orchestrator runs every
check in a CheckSuite against every record; the result list is what report.py
and the harness adapters consume.

Pass-rate denominators exclude NA results — a parameter-extraction check on
a boundary query is not-applicable, not a failure. OBSERVED results record
a value (e.g. the `limit` an audit-log call used) without a pass/fail
judgement; those are summary statistics for the report, not gating signal.

Projects extend the default check set via CheckSuite.with_defaults():

    from agent_evals import CheckSuite, make_param_observer, score_all

    suite = CheckSuite.with_defaults(
        extra_observations=[
            make_param_observer("get_query_audit_log", "limit"),
            make_param_observer("get_forecast_by_cohort", "months_ahead"),
        ],
        query_specific={
            "pe-01": check_pe01_params,
            "ts-02": check_ts02_params,
        },
    )
    scored = score_all(records, suite)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from agent_evals.trace import TraceRecord

CheckFn = Callable[[TraceRecord], "CheckResult"]


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NA = "not_applicable"
    OBSERVED = "observed"


@dataclass(frozen=True)
class CheckResult:
    """One check's result for one record.

    `name` is the check identifier (stable; used in dashboards and reports).
    `outcome` distinguishes pass/fail/NA/observed (see Outcome).
    `detail` is a one-line human explanation — short enough to fit in a
    Markdown table cell.
    `value` is the observed datum when outcome is OBSERVED, or None.
    """

    name: str
    outcome: Outcome
    detail: str
    value: Any | None = None


# ----------------------------------------------------------------------------
# Default universal checks (run against every record)
# ----------------------------------------------------------------------------

def check_tool_selection(r: TraceRecord) -> CheckResult:
    """Tools called match expected, in order.

    For boundary queries (expected_tools == ()), the expected set is empty;
    any tool call is a failure. For tool-chaining queries, order matters.
    """
    expected = list(r.expected_tools)
    called = list(r.tools_called)
    if called == expected:
        return CheckResult(
            "tool_selection",
            Outcome.PASS,
            f"called {expected} as expected" if expected else "no tools called (boundary)",
        )
    return CheckResult(
        "tool_selection",
        Outcome.FAIL,
        f"expected {expected}, called {called}",
    )


def check_tool_count(r: TraceRecord) -> CheckResult:
    """Number of tool calls matches expected.

    Catches over-enrichment (extra tool called) and under-tool (missed
    chained call). Distinct from tool_selection because a record can call
    the right tools the wrong number of times.
    """
    expected_n = len(r.expected_tools)
    actual_n = r.tool_call_count
    if actual_n == expected_n:
        return CheckResult(
            "tool_count",
            Outcome.PASS,
            f"{actual_n} calls as expected",
        )
    return CheckResult(
        "tool_count",
        Outcome.FAIL,
        f"expected {expected_n} calls, got {actual_n}",
    )


def check_boundary_no_tools(r: TraceRecord) -> CheckResult:
    """For boundary queries, no tool should be called.

    The 'refusals are full stops' guardrail in deterministic form. Triggers
    on category == 'boundary' (the only category name with shared-package
    semantics; other categories are project-defined).
    """
    if r.category != "boundary":
        return CheckResult("boundary_no_tools", Outcome.NA, "not a boundary query")
    if r.tool_call_count == 0:
        return CheckResult(
            "boundary_no_tools",
            Outcome.PASS,
            "refused without tool call",
        )
    return CheckResult(
        "boundary_no_tools",
        Outcome.FAIL,
        f"boundary query called {r.tool_call_count} tool(s): {list(r.tools_called)}",
    )


def check_boundary_single_turn(r: TraceRecord) -> CheckResult:
    """Boundary queries should resolve in one turn — no loop, no detour."""
    if r.category != "boundary":
        return CheckResult("boundary_single_turn", Outcome.NA, "not a boundary query")
    if r.turns == 1:
        return CheckResult("boundary_single_turn", Outcome.PASS, "single turn")
    return CheckResult(
        "boundary_single_turn",
        Outcome.FAIL,
        f"boundary query took {r.turns} turns",
    )


def check_stopped_cleanly(r: TraceRecord) -> CheckResult:
    """Loop ended on end_turn, not by hitting max_turns.

    max_turns reached means the agent didn't recognise it had finished or
    got stuck looping. Both are failure modes Layer 1 can spot without
    judgement.
    """
    if r.stop_reason == "end_turn":
        return CheckResult("stopped_cleanly", Outcome.PASS, "stop_reason=end_turn")
    return CheckResult(
        "stopped_cleanly",
        Outcome.FAIL,
        f"stop_reason={r.stop_reason!r} (turns={r.turns}/{r.max_turns})",
    )


def check_no_tool_errors(r: TraceRecord) -> CheckResult:
    """No tool call raised an error.

    A tool_error in the trace means the tool call failed. That's an infra
    signal, not an agent-quality signal — but it invalidates the scoring of
    everything downstream, so the eval should flag it loudly.
    """
    errored = [tc for tc in r.tool_calls if tc.error]
    if not errored:
        return CheckResult("no_tool_errors", Outcome.PASS, "no tool errors")
    names = ", ".join(f"{tc.tool}: {tc.error}" for tc in errored)
    return CheckResult("no_tool_errors", Outcome.FAIL, names)


def check_tools_within_schema(r: TraceRecord) -> CheckResult:
    """Every called tool is in the available tool list.

    Defence-in-depth — the Anthropic SDK should reject out-of-schema tool
    names, but if anything ever leaks through, the eval should catch it.
    """
    available = set(r.available_tools)
    called = set(r.tools_called)
    unknown = called - available
    if not unknown:
        return CheckResult("tools_within_schema", Outcome.PASS, "all tools in schema")
    return CheckResult(
        "tools_within_schema",
        Outcome.FAIL,
        f"unknown tool(s) called: {sorted(unknown)}",
    )


DEFAULT_UNIVERSAL: tuple[CheckFn, ...] = (
    check_tool_selection,
    check_tool_count,
    check_boundary_no_tools,
    check_boundary_single_turn,
    check_stopped_cleanly,
    check_no_tool_errors,
    check_tools_within_schema,
)


# ----------------------------------------------------------------------------
# Default observations (no pass/fail — record a value for the report)
# ----------------------------------------------------------------------------

def observe_empty_results(r: TraceRecord) -> CheckResult:
    """Record which tool calls returned empty results.

    Empty results aren't failures — they often test domain-inference
    guardrails (the agent should name the gap rather than fabricate data).
    Layer 1 records which calls were empty; Layer 2 scores how the
    synthesis handles them.

    Disambiguates by the most identifying input field — cohort_id,
    product_type, or whatever the tool uses as its primary key.
    """
    empty_records: list[dict[str, Any]] = []
    for tc in r.tool_calls:
        if tc.result_preview_len == 0 and tc.error is None:
            identifying = (
                tc.input.get("cohort_id")
                or tc.input.get("product_type")
                or tc.input.get("id")
                or tc.input.get("name")
                or tc.input.get("query")
            )
            descriptor = f"{tc.tool}({identifying!r})" if identifying else tc.tool
            empty_records.append(
                {"tool": tc.tool, "descriptor": descriptor, "input": dict(tc.input)}
            )
    if not empty_records:
        return CheckResult("empty_results", Outcome.NA, "no empty results")
    descriptors = [er["descriptor"] for er in empty_records]
    return CheckResult(
        "empty_results",
        Outcome.OBSERVED,
        f"empty: {descriptors}",
        value=empty_records,
    )


def observe_token_cost(r: TraceRecord) -> CheckResult:
    """Record total token usage for the run.

    Token cost is a first-class scoring output — quality deltas need to be
    read alongside the cost deltas they create. A skill or prompt that adds
    modest quality at large cost is a different artefact from one that adds
    modest quality at small cost.
    """
    return CheckResult(
        "token_cost",
        Outcome.OBSERVED,
        f"input={r.total_input_tokens}, output={r.total_output_tokens}",
        value={"input": r.total_input_tokens, "output": r.total_output_tokens},
    )


DEFAULT_OBSERVATIONS: tuple[CheckFn, ...] = (
    observe_empty_results,
    observe_token_cost,
)


# ----------------------------------------------------------------------------
# Parameter-observation factory (project convenience)
# ----------------------------------------------------------------------------

def make_param_observer(tool_name: str, param_name: str, *, observer_name: str | None = None) -> CheckFn:
    """Build an observation function that records a parameter on a tool.

    For projects that need to track 'what value did the agent pick for
    parameter X on tool Y?' as an OBSERVED data point. The factory keeps
    project code small:

        observe_audit_limit = make_param_observer("get_query_audit_log", "limit")
        observe_forecast_horizon = make_param_observer("get_forecast_by_cohort", "months_ahead")

    The returned function:
    - Returns NA if the tool wasn't called in this run.
    - Returns OBSERVED with the first call's parameter value if it was.
    - If the tool was called multiple times, records the list of values.
    """
    name = observer_name or f"param/{tool_name}/{param_name}"

    def _observer(r: TraceRecord) -> CheckResult:
        calls = [tc for tc in r.tool_calls if tc.tool == tool_name]
        if not calls:
            return CheckResult(
                name,
                Outcome.NA,
                f"no {tool_name} call in this run",
            )
        values = [tc.input.get(param_name) for tc in calls]
        if len(values) == 1:
            return CheckResult(
                name,
                Outcome.OBSERVED,
                f"{param_name}={values[0]}",
                value=values[0],
            )
        return CheckResult(
            name,
            Outcome.OBSERVED,
            f"{param_name}={values!r}",
            value=values,
        )

    _observer.__name__ = f"observe_{tool_name}_{param_name}"
    return _observer


# ----------------------------------------------------------------------------
# Check suite — composes default + project-specific checks
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckSuite:
    """A set of checks to run against every record.

    Built either by passing all three lists directly, or via with_defaults()
    to extend the package defaults with project-specific additions.

    Universal checks run against every record and produce pass/fail/NA.
    Observations run against every record and produce OBSERVED values.
    Query-specific checks run only against records whose query_id matches
    the dict key.
    """

    universal: tuple[CheckFn, ...]
    observations: tuple[CheckFn, ...]
    query_specific: dict[str, CheckFn] = field(default_factory=dict)

    @classmethod
    def with_defaults(
        cls,
        *,
        extra_universal: tuple[CheckFn, ...] | list[CheckFn] = (),
        extra_observations: tuple[CheckFn, ...] | list[CheckFn] = (),
        query_specific: dict[str, CheckFn] | None = None,
    ) -> CheckSuite:
        """Build a suite using the package defaults plus project additions.

        Extra checks run AFTER the defaults in their respective sections, so
        the report ordering remains: package defaults first, project
        additions second, query-specific last.
        """
        return cls(
            universal=tuple(DEFAULT_UNIVERSAL) + tuple(extra_universal),
            observations=tuple(DEFAULT_OBSERVATIONS) + tuple(extra_observations),
            query_specific=dict(query_specific or {}),
        )


# ----------------------------------------------------------------------------
# Scoring orchestration
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoredRecord:
    """A TraceRecord paired with the list of check results scored against it."""

    record: TraceRecord
    checks: list[CheckResult]

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.outcome == Outcome.PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.outcome == Outcome.FAIL)

    @property
    def na_count(self) -> int:
        return sum(1 for c in self.checks if c.outcome == Outcome.NA)

    @property
    def observed_count(self) -> int:
        return sum(1 for c in self.checks if c.outcome == Outcome.OBSERVED)

    @property
    def pass_rate(self) -> float:
        """Pass rate excluding NA and OBSERVED — those aren't pass/fail."""
        denom = self.pass_count + self.fail_count
        if denom == 0:
            return 0.0
        return self.pass_count / denom


def score_record(r: TraceRecord, suite: CheckSuite) -> list[CheckResult]:
    """Run a check suite against one record.

    Order is deterministic: universal first, then observations, then any
    query-specific check that matches the query_id. The report renders in
    the same order it receives.
    """
    results: list[CheckResult] = []
    for fn in suite.universal:
        results.append(fn(r))
    for fn in suite.observations:
        results.append(fn(r))
    query_check = suite.query_specific.get(r.query_id)
    if query_check is not None:
        results.append(query_check(r))
    return results


def score_all(records: list[TraceRecord], suite: CheckSuite) -> list[ScoredRecord]:
    """Score every record. Result order matches input order."""
    return [ScoredRecord(record=r, checks=score_record(r, suite)) for r in records]
