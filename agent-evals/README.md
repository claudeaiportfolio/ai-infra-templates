# agent-evals

A trace-driven eval framework for agentic AI workloads, designed for
post-hoc scoring of agent runs against deterministic checks. Hosted in
`claudeaiportfolio/ai-infra-templates` and consumed across portfolio
projects.

## What it is

The framework reads JSONL trace files (one per agent run) and produces a
list of `CheckResult`s per record, plus a Markdown report comparing runs.
It ships with:

- A canonical `TraceRecord` model parsed from the agent loop's JSONL output
- Universal Layer 1 deterministic checks (tool selection, parameter validity,
  stop reasons, boundary refusals)
- A `CheckSuite` that projects extend with domain-specific checks
- A `make_param_observer` factory for tracking values of specific parameters
  on specific tools
- Markdown rendering for single-run reports and side-by-side comparisons
- Optional harness adapters for Braintrust and Langfuse — no-op when env
  vars are unset, so the same code works offline and in CI

## What it is not

- An agent runtime (the framework scores traces that already exist)
- An LLM-judge framework (Layer 2 is on the roadmap, not in v0.1)
- An eval platform (the adapters post to platforms you bring)
- Coupled to any particular agent framework — works with any agent that
  emits the documented trace schema

## Install

The package lives in a subdirectory of the `ai-infra-templates` monorepo.
Consumers pin by tag:

```bash
pip install "git+https://github.com/claudeaiportfolio/ai-infra-templates.git@agent-evals-v0.1.0#subdirectory=agent-evals"
```

Optional harness extras:

```bash
pip install "git+...#subdirectory=agent-evals[braintrust]"
pip install "git+...#subdirectory=agent-evals[langfuse]"
pip install "git+...#subdirectory=agent-evals[all]"
```

For local development:

```bash
pip install -e .[dev]
pytest
```

## Quickstart

```python
from pathlib import Path
from agent_evals import (
    CheckSuite,
    load_directory,
    make_param_observer,
    render_single_run,
    score_all,
)

# 1. Build a CheckSuite — defaults plus your project's domain checks
suite = CheckSuite.with_defaults(
    extra_observations=[
        make_param_observer("get_audit_log", "limit"),
    ],
    query_specific={
        "param-test-01": my_param_check,
    },
)

# 2. Load and score
records = load_directory(Path("traces/"))
scored = score_all(records, suite)

# 3. Render
print(render_single_run(scored, label="with-skill"))
```

For full end-to-end including JSON output and Langfuse/Braintrust fan-out,
use the CLI entry points (`agent-evals-score`, `agent-evals-compare`) or
import them programmatically.

## Trace schema contract

The framework expects JSONL traces with these event `kind`s:

| Event | Required | Notes |
|---|---|---|
| `query_meta` | Yes (1x) | `run_id`, `query_id`, `category`, `expected_tools`, `skills_enabled` |
| `loop_start` | Yes (1x) | `question`, `model` |
| `claude_response` | Per turn | `turn`, `stop_reason`, `input_tokens`, `output_tokens` |
| `tool_use` | Per call | `turn`, `tool`, `input` (dict) |
| `tool_result` or `tool_error` | Per call | `turn`, `tool`, `preview`/`preview_len` or `error` |
| `loop_end` | Yes (1x) | `turns`, `tool_calls`, `stop_reason`, `final_text_full` |

`tool_use` and the matching `tool_result`/`tool_error` are paired
positionally — the parser doesn't require `tool_use_id` on results
(though it tolerates it).

Categories beyond `boundary` are project-defined; the framework
special-cases `boundary` for two universal checks (`check_boundary_no_tools`,
`check_boundary_single_turn`). Other category names affect grouping in the
report but don't trigger any special checks.

## CheckSuite extension model

The package ships with `DEFAULT_UNIVERSAL` (7 pass/fail checks) and
`DEFAULT_OBSERVATIONS` (2 observation functions). Projects extend by
passing extras to `CheckSuite.with_defaults()`:

```python
suite = CheckSuite.with_defaults(
    extra_universal=[my_custom_universal_check],
    extra_observations=[my_observer],
    query_specific={"q-01": my_param_check, "q-02": ...},
)
```

Universal checks return `PASS`/`FAIL`/`NA`. Observations return `OBSERVED`
with a value. Query-specific checks fire only when `record.query_id`
matches the dict key — useful for parameter-extraction tests that depend
on knowing the original natural-language input.

## Harness adapters

Both adapters share the same interface:

```python
from agent_evals.adapters.braintrust import is_enabled, post_run
```

- `is_enabled()` returns True iff required env vars are set
- `post_run(scored, run_label)` posts to the platform; no-op if not enabled

Env vars:

| Adapter | Required | Optional |
|---|---|---|
| `braintrust` | `BRAINTRUST_API_KEY` | `BRAINTRUST_PROJECT` (default `agent-evals`), `BRAINTRUST_API_URL` (self-hosted) |
| `langfuse` | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | `LANGFUSE_HOST` (default `https://cloud.langfuse.com`) |

Both adapters raise `ImportError` if their SDK isn't installed and the
relevant env vars are set. The CLI catches this and prints a friendly note.

## Versioning

This package is versioned independently of the host repo. Tags follow the
pattern `agent-evals-vX.Y.Z`. Consumers should pin by tag for stability.

Release flow:

1. Bump `version = "X.Y.Z"` in `pyproject.toml`
2. Bump `__version__` in `src/agent_evals/__init__.py`
3. Commit, tag `agent-evals-vX.Y.Z`, push tag
4. Consumers update their pin

## Why the post-hoc model

Most evals tools assume "instrument the agent at runtime and report
results as the agent runs." This package assumes the opposite: the agent
has already run, the trace files exist on disk, and we score them after
the fact. Two reasons this is the right shape:

1. **The trace files are durable evidence.** They round-trip into the
   scorer, into Layer 2 (eventual), into a regenerated report on any
   subsequent change to the rubric. Runtime-only scoring throws away that
   replay capability.
2. **Decouples scoring from agent runtime.** The agent loop can be raw
   Anthropic SDK, LangGraph, OpenAI Agents SDK, or anything that emits
   the schema. The scorer doesn't care.

The harness adapters fan out scores to platforms (Braintrust, Langfuse)
that expect the runtime model — they accept post-hoc uploads cleanly via
their `init_logger()` + `start_span()` APIs.

## License

MIT.
