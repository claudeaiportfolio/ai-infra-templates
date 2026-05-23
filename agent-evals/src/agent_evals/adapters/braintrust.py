"""Braintrust adapter for the eval pipeline.

Posts scored runs to Braintrust as logged spans with attached scores.
Engineer-native API, CI-first integration, GitHub Actions feedback.

Behaviour:

- When BRAINTRUST_API_KEY is unset, the adapter is a no-op. Lets local dev
  and CI runs work without Braintrust credentials.
- When the braintrust SDK is not installed, post_run() raises ImportError;
  the caller (usually the CLI) catches and proceeds without Braintrust.
- Each scored run becomes one logger session with one root span; each
  query within the run becomes a child span; each Layer 1 check attaches
  to its query's span as a score.

Env vars:
    BRAINTRUST_API_KEY        (required)
    BRAINTRUST_PROJECT        (optional, defaults to "agent-evals")
    BRAINTRUST_API_URL        (optional, for self-hosted Braintrust)

Pass/fail checks score as 1.0 / 0.0 in the span's `scores` dict. Observed
values are recorded in the span's metadata, not as scores. NA checks are
omitted entirely.
"""
from __future__ import annotations

import os
import sys
from typing import Any

from agent_evals.deterministic import Outcome, ScoredRecord


def is_enabled() -> bool:
    """True iff the Braintrust API key is set in the environment.

    Reads from os.environ only — no .env loading. The invocation surface
    (Makefile, uv run --env-file, direnv) owns secret loading.
    """
    return bool(os.environ.get("BRAINTRUST_API_KEY"))


def post_run(scored: list[ScoredRecord], run_label: str) -> None:
    """Post a scored eval run to Braintrust.

    No-op if is_enabled() returns False. Raises ImportError if the
    braintrust SDK is not installed.

    One logger session per scored run. One root span per session, carrying
    run-level metadata (label, record count, skill state). One child span
    per query, carrying that query's input/output and per-check scores.
    """
    if not is_enabled():
        return

    try:
        import braintrust  # type: ignore
    except ImportError as e:
        raise ImportError(
            "braintrust SDK not installed. Install with: pip install agent-evals[braintrust]"
        ) from e

    # api_url is read from BRAINTRUST_API_URL by the SDK itself when set.
    # api_key comes from BRAINTRUST_API_KEY automatically too. We pass them
    # explicitly only when self-hosted, so the default cloud install is
    # zero-config beyond the api key.
    init_kwargs: dict[str, Any] = {
        "project": os.environ.get("BRAINTRUST_PROJECT", "agent-evals"),
        "api_key": os.environ["BRAINTRUST_API_KEY"],
    }
    api_url = os.environ.get("BRAINTRUST_API_URL")
    if api_url:
        init_kwargs["api_url"] = api_url

    logger = braintrust.init_logger(**init_kwargs)

    skill_state = _infer_skill_state(scored)

    # Root span scopes the whole eval run. Child spans (per query) inherit
    # via the context manager — Braintrust's tracing model is parent/child
    # by execution scope, so the `with` block is load-bearing.
    with logger.start_span(
        name=f"eval-run/{run_label}",
        metadata={
            "label": run_label,
            "record_count": len(scored),
            "skill_state": skill_state,
            "categories": sorted({s.record.category for s in scored}),
            "tags": ["eval", "layer-1", f"skill:{skill_state}"],
        },
    ) as root_span:
        for s in scored:
            _post_record(logger, s)

        # Aggregate scores on the root span for the run-level dashboard.
        root_span.log(
            scores=_aggregate_pass_rate_by_category(scored),
            metadata={
                "total_pass": sum(s.pass_count for s in scored),
                "total_fail": sum(s.fail_count for s in scored),
            },
        )

    # async_flush=True is the SDK default — explicit flush ensures the
    # one-shot CLI process exits with all writes durable.
    try:
        logger.flush()
    except Exception as e:
        print(f"warning: braintrust flush failed: {e}", file=sys.stderr)


def _post_record(logger: Any, s: ScoredRecord) -> None:
    """Post one query's span and scores to the active Braintrust logger."""
    pass_fail_scores: dict[str, float] = {}
    observations: dict[str, Any] = {}

    for c in s.checks:
        if c.outcome == Outcome.PASS:
            pass_fail_scores[c.name] = 1.0
        elif c.outcome == Outcome.FAIL:
            pass_fail_scores[c.name] = 0.0
        elif c.outcome == Outcome.OBSERVED:
            observations[c.name] = c.value
        # NA: skip entirely

    with logger.start_span(
        name=f"query/{s.record.query_id}",
    ) as span:
        span.log(
            input={
                "query_id": s.record.query_id,
                "category": s.record.category,
                "question": s.record.question,
                "expected_tools": list(s.record.expected_tools),
            },
            output={
                "tools_called": list(s.record.tools_called),
                "stop_reason": s.record.stop_reason,
                "turns": s.record.turns,
                "final_text_preview": s.record.final_text[:300],
            },
            scores=pass_fail_scores,
            metadata={
                "model": s.record.model,
                "tool_call_count": s.record.tool_call_count,
                "total_input_tokens": s.record.total_input_tokens,
                "total_output_tokens": s.record.total_output_tokens,
                "system_prompt_chars": s.record.system_prompt_chars,
                "skills_enabled": s.record.skills_enabled,
                "skills_loaded": list(s.record.skills_loaded),
                "final_text_truncated": s.record.final_text_truncated,
                "observations": observations,
                "check_details": {
                    c.name: c.detail for c in s.checks if c.outcome != Outcome.NA
                },
            },
        )


def _aggregate_pass_rate_by_category(scored: list[ScoredRecord]) -> dict[str, float]:
    """Compute pass-rate per category for the root-span score summary.

    Braintrust's dashboard surfaces root-span scores at the run level,
    which makes per-category pass-rates the right shape for at-a-glance
    monitoring. NA and OBSERVED outcomes are excluded from the denominator
    (consistent with ScoredRecord.pass_rate).
    """
    from collections import defaultdict

    by_cat_pass: dict[str, int] = defaultdict(int)
    by_cat_fail: dict[str, int] = defaultdict(int)
    for s in scored:
        by_cat_pass[s.record.category] += s.pass_count
        by_cat_fail[s.record.category] += s.fail_count

    result: dict[str, float] = {}
    for cat in by_cat_pass:
        denom = by_cat_pass[cat] + by_cat_fail[cat]
        if denom > 0:
            result[f"pass_rate/{cat}"] = by_cat_pass[cat] / denom
    return result


def _infer_skill_state(scored: list[ScoredRecord]) -> str:
    """Summarise the skills_enabled state across records into one tag value."""
    states = {s.record.skills_enabled for s in scored}
    if states == {True}:
        return "with-skill"
    if states == {False}:
        return "baseline"
    return "mixed"
