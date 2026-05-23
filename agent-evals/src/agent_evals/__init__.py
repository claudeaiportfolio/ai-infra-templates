"""Trace-driven eval framework for agentic AI workloads.

Layer 1 (deterministic) scores trace files emitted by an agent loop against
the query bank metadata. Layer 2 (LLM-judge) scores synthesis quality on
rubric criteria — see agent_evals.judge.

Public API:

    from agent_evals import (
        TraceRecord, load_directory,
        CheckResult, Outcome, ScoredRecord,
        CheckSuite, DEFAULT_UNIVERSAL, DEFAULT_OBSERVATIONS,
        make_param_observer,
        score_all, score_record,
        render_single_run, render_comparison,
    )

    from agent_evals.judge import (
        JudgeCriterion, JudgeResult, JudgedRun,
        run_judge, render_judge_run, render_judge_comparison,
    )

Harness adapters are optional:

    from agent_evals.adapters.braintrust import post_run as post_to_braintrust
    from agent_evals.adapters.langfuse import post_run as post_to_langfuse

Each adapter no-ops when its required env vars are unset; it raises
ImportError if its SDK isn't installed. Install the relevant extra:

    pip install agent-evals[braintrust]
    pip install agent-evals[langfuse]
    pip install agent-evals[judge]
    pip install agent-evals[all]

Trace schema contract: the loader expects JSONL files with one event per
line and these event kinds:
    query_meta   (required, exactly once)
    loop_start   (required, exactly once)
    turn_start, claude_request, claude_response, stop_reason  (per turn)
    tool_use, tool_result, tool_error  (paired per call)
    loop_end     (required, exactly once)

See trace.py for the full TraceRecord shape and field expectations.
"""
from agent_evals.deterministic import (
    DEFAULT_OBSERVATIONS,
    DEFAULT_UNIVERSAL,
    CheckResult,
    CheckSuite,
    Outcome,
    ScoredRecord,
    make_param_observer,
    score_all,
    score_record,
)
from agent_evals.report import render_comparison, render_single_run
from agent_evals.trace import ToolCall, TraceRecord, TurnUsage, load_directory

__version__ = "0.2.0"

__all__ = [
    # Trace model
    "TraceRecord",
    "ToolCall",
    "TurnUsage",
    "load_directory",
    # Check primitives
    "CheckResult",
    "Outcome",
    "ScoredRecord",
    "CheckSuite",
    "DEFAULT_UNIVERSAL",
    "DEFAULT_OBSERVATIONS",
    "make_param_observer",
    "score_all",
    "score_record",
    # Reports
    "render_single_run",
    "render_comparison",
    # Version
    "__version__",
]
