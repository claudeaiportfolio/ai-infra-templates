"""Compare two scored eval runs and render a Markdown PR-comment.

Usage:
    agent-evals-compare --with-skill out/with-skill.json --baseline out/baseline.json
    agent-evals-compare --with-skill out/with-skill.json --baseline out/baseline.json \\
        --markdown out/comparison.md

Project-specific mock-baseline generation (synthesising a baseline from the
with-skill run for development) belongs in the project's own shim script —
the shape of the mock depends on which deltas the project's SKILL/prompt
introduces. See snowflake-forecasting's scripts/compare_runs.py for an
example.

The comparison reads two JSON files emitted by `agent-evals-score --json`,
reconstructs scoring records (with token totals re-attached as a synthetic
TurnUsage), and runs render_comparison from agent_evals.report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_evals.deterministic import CheckResult, Outcome, ScoredRecord
from agent_evals.report import render_comparison
from agent_evals.trace import TraceRecord, TurnUsage


def main() -> int:
    """Entry point for the `agent-evals-compare` console script."""
    args = _parse_args()
    return run(
        with_skill_json=Path(args.with_skill),
        baseline_json=Path(args.baseline),
        markdown_path=Path(args.markdown) if args.markdown else None,
        quiet=args.quiet,
    )


def run(
    *,
    with_skill: list[ScoredRecord] | None = None,
    baseline: list[ScoredRecord] | None = None,
    with_skill_json: Path | None = None,
    baseline_json: Path | None = None,
    markdown_path: Path | None = None,
    quiet: bool = False,
    extra_prologue: str = "",
) -> int:
    """Compare two scored runs and render a Markdown report.

    Accepts either in-memory ScoredRecord lists (for project shims doing
    mock-baseline generation) or JSON file paths (for the standard two-run
    comparison flow).

    `extra_prologue` is prepended to the report — used by project shims to
    inject a mock-baseline warning banner.
    """
    if with_skill is None:
        if with_skill_json is None:
            print("error: pass --with-skill JSON or in-memory scored records", file=sys.stderr)
            return 2
        with_skill = _load_scored_json(with_skill_json)
    if baseline is None:
        if baseline_json is None:
            print("error: pass --baseline JSON or in-memory scored records", file=sys.stderr)
            return 2
        baseline = _load_scored_json(baseline_json)

    report = render_comparison(with_skill, baseline)
    if extra_prologue:
        report = extra_prologue + "\n" + report

    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(report, encoding="utf-8")
    if not quiet:
        print(report)

    return 0


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--with-skill",
        required=True,
        help="JSON file from agent-evals-score for the with-skill run",
    )
    p.add_argument(
        "--baseline",
        required=True,
        help="JSON file from agent-evals-score for the baseline run",
    )
    p.add_argument("--markdown", help="Write Markdown report to this path")
    p.add_argument("--quiet", action="store_true", help="Suppress stdout output")
    return p.parse_args()


def _load_scored_json(path: Path) -> list[ScoredRecord]:
    """Reconstruct ScoredRecord list from agent-evals-score JSON output.

    The reconstructed records hold enough state for render_comparison to
    work: query metadata, check outcomes/details, token totals. They are
    NOT full TraceRecords — trace events themselves aren't serialised, so
    re-scoring from JSON isn't possible. JSON is the durable summary;
    JSONL trace files are the source of truth for re-scoring.

    Token totals are reconstructed via a synthetic single-turn TurnUsage on
    each record, so the TraceRecord.total_input_tokens property works
    transparently — the comparison renderer doesn't need to know the data
    came from JSON.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[ScoredRecord] = []
    for rec in payload["records"]:
        fake_usage = (
            TurnUsage(
                turn=0,
                stop_reason="(from json)",
                input_tokens=rec["total_input_tokens"],
                output_tokens=rec["total_output_tokens"],
            ),
        )
        tr = TraceRecord(
            path=Path(rec["trace_path"]),
            run_id="(from json)",
            query_id=rec["query_id"],
            category=rec["category"],
            expected_tools=(),
            note="",
            skills_enabled=rec["skills_enabled"],
            question="",
            model="",
            prompt_version="",
            available_tools=(),
            max_turns=0,
            skills_loaded=(),
            system_prompt_chars=0,
            tool_calls=(),
            turn_usage=fake_usage,
            turns=0,
            tool_call_count=0,
            stop_reason="",
            final_text="",
            final_text_truncated=rec["final_text_truncated"],
        )
        checks = [
            CheckResult(
                name=c["name"],
                outcome=Outcome(c["outcome"]),
                detail=c["detail"],
                value=c["value"],
            )
            for c in rec["checks"]
        ]
        out.append(ScoredRecord(record=tr, checks=checks))
    return out


if __name__ == "__main__":
    sys.exit(main())
