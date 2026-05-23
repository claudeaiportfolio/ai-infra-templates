"""Generic CLI entry points for the eval pipeline.

Projects consume these in two ways:

1. As thin shim scripts that import their own CheckSuite and call run():

    # scripts/score_traces.py in a consumer project:
    from agent_evals.cli.score import run
    from my_project.evals.wiring import build_suite
    import sys
    sys.exit(run(suite=build_suite()))

2. Via the entry-point command `agent-evals-score`, with --suite pointing
   to an importable factory (module:attr):

    agent-evals-score traces/ --suite my_project.evals.wiring:build_suite

The first form is more explicit and is the recommended pattern for
portfolio projects. The second form exists for cases where a user wants
to score traces from outside their project's Python environment.
"""
