"""Langfuse adapter for the eval pipeline.

Posts scored runs to Langfuse (Cloud or self-hosted) so that scores land in
a queryable surface alongside the agent traces. The same code path serves
both deployment shapes — only the env vars change:

    # Langfuse Cloud (free / paid tier)
    LANGFUSE_HOST=https://cloud.langfuse.com
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...

    # Self-hosted (Helm on existing cluster)
    LANGFUSE_HOST=https://langfuse.example.com
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...

Behaviour:

- When LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY are unset, the adapter
  is a no-op. This lets local dev and CI runs work without a Langfuse
  instance, and keeps the scorer's offline mode trivially testable.
- When the langfuse SDK is not installed, post_run() raises ImportError;
  the caller (usually the CLI) catches and proceeds without Langfuse.
- Each scored run becomes one Langfuse trace; each query within the run
  becomes a span; each Layer 1 check becomes a score on the span.

Pass/fail checks score as 1.0 / 0.0. Observed values are recorded as
metadata, not scores — they don't have a pass/fail interpretation. NA
checks are omitted entirely (Langfuse doesn't need to know about them).
"""
from __future__ import annotations

import os
import sys
from typing import Any

from agent_evals.deterministic import Outcome, ScoredRecord


def is_enabled() -> bool:
    """True iff both Langfuse keys are set in the environment.

    Reads from os.environ only — no .env loading. The invocation surface
    (Makefile, uv run --env-file, direnv) owns secret loading.
    """
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def post_run(scored: list[ScoredRecord], run_label: str) -> None:
    """Post a scored eval run to Langfuse.

    No-op if is_enabled() returns False. Raises ImportError if the
    langfuse SDK is not installed; the caller is expected to catch and
    proceed without Langfuse fan-out.

    One Langfuse trace per scored run, one span per query, one score per
    check. The trace tags include the run label and skill state so the
    Langfuse UI surfaces them filterably.
    """
    if not is_enabled():
        return

    try:
        from langfuse import Langfuse  # type: ignore
    except ImportError as e:
        raise ImportError(
            "langfuse SDK not installed. Install with: pip install agent-evals[langfuse]"
        ) from e

    client = Langfuse(
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    )

    skill_state = _infer_skill_state(scored)
    trace = client.trace(
        name=f"eval-run/{run_label}",
        tags=["eval", "layer-1", f"skill:{skill_state}"],
        metadata={
            "label": run_label,
            "record_count": len(scored),
            "skill_state": skill_state,
            "categories": sorted({s.record.category for s in scored}),
        },
    )

    for s in scored:
        _post_record(trace, s)

    try:
        client.flush()
    except Exception as e:
        print(f"warning: langfuse flush failed: {e}", file=sys.stderr)


def _post_record(trace: Any, s: ScoredRecord) -> None:
    """Post one query's span and scores to a Langfuse trace."""
    span = trace.span(
        name=f"query/{s.record.query_id}",
        metadata={
            "query_id": s.record.query_id,
            "category": s.record.category,
            "expected_tools": list(s.record.expected_tools),
            "tools_called": list(s.record.tools_called),
            "question": s.record.question,
            "model": s.record.model,
            "stop_reason": s.record.stop_reason,
            "turns": s.record.turns,
            "tool_call_count": s.record.tool_call_count,
            "total_input_tokens": s.record.total_input_tokens,
            "total_output_tokens": s.record.total_output_tokens,
            "system_prompt_chars": s.record.system_prompt_chars,
            "skills_enabled": s.record.skills_enabled,
            "skills_loaded": list(s.record.skills_loaded),
            "final_text_truncated": s.record.final_text_truncated,
            "observations": {
                c.name: c.value for c in s.checks if c.outcome == Outcome.OBSERVED
            },
        },
    )

    for c in s.checks:
        if c.outcome == Outcome.PASS:
            value = 1.0
        elif c.outcome == Outcome.FAIL:
            value = 0.0
        else:
            continue  # NA and OBSERVED don't produce scores
        span.score(name=c.name, value=value, comment=c.detail)

    span.end()


def _infer_skill_state(scored: list[ScoredRecord]) -> str:
    """Summarise the skills_enabled state across records into one tag value."""
    states = {s.record.skills_enabled for s in scored}
    if states == {True}:
        return "with-skill"
    if states == {False}:
        return "baseline"
    return "mixed"
