"""Score a directory of traces against a CheckSuite.

Two ways to invoke:

A) Programmatically from a project's shim script:

    from agent_evals.cli.score import run
    from my_project.evals.wiring import build_suite
    sys.exit(run(suite=build_suite()))

B) As the installed `agent-evals-score` command with --suite:

    agent-evals-score traces/ --suite my_project.evals.wiring:build_suite

The script writes a Markdown report (stdout by default, optional --markdown
path) and a JSON results file (optional --json path). When the env vars
for an adapter are set and the adapter's SDK is installed, scores are also
posted to that platform.

Exit code: 1 if anything failed, 0 if all pass/fail checks passed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from agent_evals.deterministic import CheckSuite, ScoredRecord, score_all
from agent_evals.report import render_single_run
from agent_evals.trace import load_directory


def main() -> int:
    """Entry point for the `agent-evals-score` console script."""
    args = _parse_args()
    suite = _load_suite_from_string(args.suite) if args.suite else CheckSuite.with_defaults()
    return run(
        suite=suite,
        trace_dir=Path(args.trace_dir),
        label=args.label,
        json_path=Path(args.json) if args.json else None,
        markdown_path=Path(args.markdown) if args.markdown else None,
        quiet=args.quiet,
    )


def run(
    *,
    suite: CheckSuite,
    trace_dir: Path | None = None,
    label: str | None = None,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
    quiet: bool = False,
    argv: list[str] | None = None,
) -> int:
    """Score `trace_dir` against `suite` and emit reports.

    When `trace_dir` is None, parses argv (or sys.argv) for the trace
    directory argument — useful for shim scripts that want to pass a
    suite programmatically but still take the trace directory from CLI.
    """
    if trace_dir is None:
        args = _parse_args(argv)
        trace_dir = Path(args.trace_dir)
        label = label or args.label
        json_path = json_path or (Path(args.json) if args.json else None)
        markdown_path = markdown_path or (Path(args.markdown) if args.markdown else None)
        quiet = quiet or args.quiet

    if not trace_dir.is_dir():
        print(f"error: {trace_dir} is not a directory", file=sys.stderr)
        return 2

    records = load_directory(trace_dir)
    if not records:
        print(f"error: no traces found under {trace_dir}", file=sys.stderr)
        return 2

    scored = score_all(records, suite)
    resolved_label = label or _default_label(scored)

    if json_path:
        _write_json(json_path, scored, resolved_label)

    markdown = render_single_run(scored, resolved_label)
    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
    if not quiet:
        print(markdown)

    _fan_out_to_harnesses(scored, resolved_label)

    total_fail = sum(s.fail_count for s in scored)
    return 1 if total_fail else 0


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace_dir", help="Directory of *.jsonl trace files")
    p.add_argument(
        "--suite",
        help="Importable suite factory as module:attr (e.g. 'my_pkg.evals:build_suite'). "
        "Omit to use agent-evals default suite (no project-specific checks).",
    )
    p.add_argument("--label", help="Run label for the report header")
    p.add_argument("--json", help="Write scoring results as JSON to this path")
    p.add_argument("--markdown", help="Write Markdown report to this path")
    p.add_argument("--quiet", action="store_true", help="Suppress stdout output")
    return p.parse_args(argv)


def _load_suite_from_string(spec: str) -> CheckSuite:
    """Resolve 'module.path:attribute' into a CheckSuite.

    The attribute may be a CheckSuite directly or a zero-arg factory
    returning one. Anything else raises a clear error.
    """
    if ":" not in spec:
        raise ValueError(
            f"invalid suite spec {spec!r}: expected 'module.path:attribute'"
        )
    module_path, attr = spec.split(":", 1)
    module = importlib.import_module(module_path)
    obj = getattr(module, attr)
    if isinstance(obj, CheckSuite):
        return obj
    if callable(obj):
        result = obj()
        if not isinstance(result, CheckSuite):
            raise TypeError(
                f"{spec} returned {type(result).__name__}, expected CheckSuite"
            )
        return result
    raise TypeError(
        f"{spec} is {type(obj).__name__}, expected CheckSuite or factory function"
    )


def _default_label(scored: list[ScoredRecord]) -> str:
    """Infer a label from the records' skills_enabled state."""
    states = {s.record.skills_enabled for s in scored}
    if states == {True}:
        return "with-skill"
    if states == {False}:
        return "baseline"
    return "mixed"


def _write_json(path: Path, scored: list[ScoredRecord], label: str) -> None:
    """Serialise scored records to JSON for downstream comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "label": label,
        "query_count": len(scored),
        "records": [
            {
                "query_id": s.record.query_id,
                "category": s.record.category,
                "skills_enabled": s.record.skills_enabled,
                "trace_path": str(s.record.path),
                "pass_count": s.pass_count,
                "fail_count": s.fail_count,
                "na_count": s.na_count,
                "observed_count": s.observed_count,
                "total_input_tokens": s.record.total_input_tokens,
                "total_output_tokens": s.record.total_output_tokens,
                "final_text_truncated": s.record.final_text_truncated,
                "checks": [
                    {
                        "name": c.name,
                        "outcome": c.outcome.value,
                        "detail": c.detail,
                        "value": c.value,
                    }
                    for c in s.checks
                ],
            }
            for s in scored
        ],
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _fan_out_to_harnesses(scored: list[ScoredRecord], label: str) -> None:
    """Post scored run to any configured harness adapter.

    Each adapter is checked independently — projects can use one, both, or
    neither. ImportError is caught with a friendly note (SDK not installed).
    """
    for adapter_name in ("langfuse", "braintrust"):
        try:
            module = importlib.import_module(f"agent_evals.adapters.{adapter_name}")
        except ImportError:
            continue
        if not module.is_enabled():
            continue
        try:
            module.post_run(scored, run_label=label)
        except ImportError as e:
            print(f"note: {adapter_name} fan-out skipped — {e}", file=sys.stderr)
        except Exception as e:
            print(f"warning: {adapter_name} fan-out failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
