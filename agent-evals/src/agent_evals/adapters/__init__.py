"""Harness adapters for posting scored runs to evals platforms.

Each adapter is an optional submodule that requires its own SDK extra:

    pip install agent-evals[braintrust]
    pip install agent-evals[langfuse]

Both adapters share the same interface:

    is_enabled() -> bool         # True iff required env vars are set
    post_run(scored, run_label)  # Post one scored run; no-op if not enabled

This lets a project switch harnesses (or use both) by flipping env vars,
without changing scorer code.
"""
