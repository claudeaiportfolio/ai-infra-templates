"""Layer 2 — LLM-judge scoring.

Layer 1 (deterministic.py) answers "did the agent do the mechanical thing":
which tools were called, with what parameters, did the loop stop cleanly. It
runs offline against committed traces with zero API calls. That's the gating
signal in CI.

Layer 2 answers "was the answer good": parameter-default appropriateness,
domain-inference quality, severity-framing consistency, next-step
suggestion quality — the rubric criteria that need a judgement call. It
calls an Anthropic model (Haiku 4.5 by default — cheap, fast, fine for a
binary or 0–3 rubric) once per (record, criterion) pair, with the rendered
trace context as input.

The split is deliberate. Layer 1 is the cheap regression net; Layer 2 is
the ground truth that confirms Layer 1 is measuring the right thing. Most
PRs only need Layer 1; Layer 2 is run when comparing skill / prompt
changes or when re-baselining the eval bank.

Project usage:

    from agent_evals.judge import JudgeCriterion, run_judge

    PARAMETER_DEFAULTS = JudgeCriterion(
        name="parameter_defaults",
        prompt='''Judge whether the agent's chosen parameter defaults...''',
        scale="binary",
    )

    judged = await run_judge(
        records=scored_records,
        criteria=[PARAMETER_DEFAULTS, ...],
        model="claude-haiku-4-5",
    )

The judge module ships zero opinions about what to judge — that lives in the
project's domain code (e.g. loan-portfolio-investigator's evals/domain_checks
defines four criteria matching the EVAL_FINDINGS rubric).

The Anthropic SDK is an optional dependency (`agent-evals[judge]`); the
module raises ImportError with an actionable message if the extra isn't
installed.
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from agent_evals.deterministic import ScoredRecord
from agent_evals.trace import TraceRecord

Scale = Literal["binary", "rubric_0_3"]


@dataclass(frozen=True)
class JudgeCriterion:
    """One thing the judge is being asked to score.

    `name` is the criterion identifier — appears in reports and dashboards.
    `prompt` is the rubric the judge sees alongside the trace context. The
        rubric should be specific enough that two human raters would agree
        on the score 80%+ of the time; vague rubrics give noisy results.
    `scale` is either "binary" (pass/fail) or "rubric_0_3" (0=fail,
        1=below-bar, 2=meets-bar, 3=exceeds).
    `applies_to_query_id` filters which records the criterion runs on.
        If empty, applies to all. Useful for criteria that only make
        sense for certain query types (e.g. severity framing only matters
        on synthesis queries, not boundary refusals).
    `applies_to_category` is the same filter on `record.category`.
    """

    name: str
    prompt: str
    scale: Scale = "binary"
    applies_to_query_id: tuple[str, ...] = ()
    applies_to_category: tuple[str, ...] = ()

    def applies_to(self, record: TraceRecord) -> bool:
        if self.applies_to_query_id and record.query_id not in self.applies_to_query_id:
            return False
        if self.applies_to_category and record.category not in self.applies_to_category:
            return False
        return True


@dataclass(frozen=True)
class JudgeResult:
    """One criterion's score for one record."""

    record_query_id: str
    criterion_name: str
    score: int  # 0/1 for binary, 0–3 for rubric_0_3
    max_score: int  # 1 for binary, 3 for rubric_0_3
    reasoning: str
    raw_response: str = ""  # full judge response for audit

    @property
    def normalised(self) -> float:
        """Score on a 0–1 scale for cross-criterion aggregation."""
        if self.max_score == 0:
            return 0.0
        return self.score / self.max_score


@dataclass(frozen=True)
class JudgedRun:
    """All judge results for one scored run.

    Pairs a list of JudgeResult with the run label so the comparison
    renderer can identify which arm (with-skill / baseline / cached /
    uncached) each result came from.
    """

    label: str
    results: tuple[JudgeResult, ...]
    model: str
    criteria: tuple[JudgeCriterion, ...] = ()

    def for_query(self, query_id: str) -> tuple[JudgeResult, ...]:
        return tuple(r for r in self.results if r.record_query_id == query_id)

    def for_criterion(self, criterion_name: str) -> tuple[JudgeResult, ...]:
        return tuple(r for r in self.results if r.criterion_name == criterion_name)

    def mean_by_criterion(self) -> dict[str, float]:
        """Average normalised score per criterion across all records."""
        buckets: dict[str, list[float]] = {}
        for r in self.results:
            buckets.setdefault(r.criterion_name, []).append(r.normalised)
        return {k: sum(v) / len(v) for k, v in buckets.items() if v}


# ----------------------------------------------------------------------------
# Trace rendering for the judge
# ----------------------------------------------------------------------------

def render_trace_for_judge(record: TraceRecord) -> str:
    """Format one TraceRecord as input for the judge model.

    Includes the question, the tool calls with their inputs and previewed
    results, and the final answer. Truncated previews are flagged so the
    judge knows it's seeing a sample, not the whole result.

    The render is plain text — Markdown headers for orientation, no exotic
    formatting. The judge is a model not a renderer; we keep the prompt
    surface small.
    """
    parts: list[str] = []
    parts.append(f"## Question (query_id={record.query_id}, category={record.category})")
    parts.append(record.question)
    parts.append("")

    if record.tool_calls:
        parts.append("## Tool calls (in order)")
        for i, tc in enumerate(record.tool_calls, 1):
            parts.append(f"### Call {i}: {tc.tool}")
            parts.append(f"input: {tc.input}")
            if tc.error:
                parts.append(f"error: {tc.error}")
            elif tc.result_preview is not None:
                preview_len = tc.result_preview_len or 0
                preview = tc.result_preview
                if preview_len > len(preview):
                    parts.append(
                        f"result_preview ({len(preview)} of {preview_len} chars):"
                    )
                else:
                    parts.append(f"result ({preview_len} chars):")
                parts.append(preview)
            parts.append("")
    else:
        parts.append("## Tool calls")
        parts.append("(none — boundary refusal or no-tools query)")
        parts.append("")

    parts.append("## Final answer")
    if record.final_text_truncated:
        parts.append("(NOTE: only a 300-char preview of the final answer is available)")
    parts.append(record.final_text)
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Judge prompt and parsing
# ----------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for an AI agent. The user will give you a single \
rubric criterion and a record of one agent run (the question, the tool calls it made, and the final answer). \
Your job is to score how well the agent's run satisfies the criterion.

You MUST respond in exactly this format (no preamble, no code fences, no extra text):

SCORE: <integer>
REASONING: <one or two sentences explaining the score>

For binary criteria, the score is 0 (fails) or 1 (passes).
For rubric_0_3 criteria, the score is 0 (fails), 1 (below bar), 2 (meets bar), or 3 (exceeds bar).

Be strict — if the criterion isn't obviously satisfied, score lower. The point of an eval judge is \
to surface differences, not to grade kindly."""


def _build_user_prompt(criterion: JudgeCriterion, rendered_trace: str) -> str:
    return f"""# Criterion: {criterion.name} ({criterion.scale})

{criterion.prompt}

# Agent run

{rendered_trace}

# Your task

Score this run against the criterion above. Respond in the SCORE/REASONING format."""


_SCORE_RE = re.compile(r"^SCORE:\s*(-?\d+)\s*$", re.MULTILINE)
_REASONING_RE = re.compile(r"^REASONING:\s*(.+?)(?:\Z|\n[A-Z]+:)", re.MULTILINE | re.DOTALL)


def parse_judge_response(text: str, scale: Scale) -> tuple[int, str]:
    """Pull (score, reasoning) out of a judge response.

    Tolerant of leading/trailing whitespace and surrounding markdown. Falls
    back to (0, "<unparseable>") if no SCORE line is found — the run
    continues, the failure is visible in the report.
    """
    max_score = 1 if scale == "binary" else 3

    score_match = _SCORE_RE.search(text)
    if not score_match:
        return 0, f"<unparseable: no SCORE line in {text[:200]!r}>"
    try:
        score = int(score_match.group(1))
    except ValueError:
        return 0, f"<unparseable: non-integer score in {score_match.group(0)!r}>"

    # Clamp to valid range. A judge returning -1 or 5 is a misbehaving model;
    # we cap rather than fail so the run produces results.
    score = max(0, min(score, max_score))

    reasoning_match = _REASONING_RE.search(text)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    return score, reasoning


# ----------------------------------------------------------------------------
# Anthropic client (lazy import + injection point for tests)
# ----------------------------------------------------------------------------

JudgeCallable = Callable[[str, str, str], Awaitable[str]]
"""(system_prompt, user_prompt, model) -> raw response text."""


def _default_anthropic_caller() -> JudgeCallable:
    """Build the default judge caller using the Anthropic SDK.

    Imported lazily so the SDK is only required when the judge runs. If
    the SDK isn't installed, raises ImportError with an actionable hint.
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError as e:
        raise ImportError(
            "agent_evals.judge requires the `anthropic` package. "
            "Install with `pip install agent-evals[judge]` or "
            "`pip install anthropic`."
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set — the judge needs API access. "
            "Set the env var or pass a custom `caller` to run_judge for testing."
        )

    client = AsyncAnthropic(api_key=api_key)

    async def _call(system_prompt: str, user_prompt: str, model: str) -> str:
        response = await client.messages.create(
            model=model,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        chunks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return "".join(chunks)

    return _call


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"


async def run_judge(
    *,
    records: list[ScoredRecord] | list[TraceRecord],
    criteria: list[JudgeCriterion],
    model: str = DEFAULT_JUDGE_MODEL,
    label: str = "",
    caller: JudgeCallable | None = None,
    concurrency: int = 4,
) -> JudgedRun:
    """Run a set of criteria across a set of records.

    `records` accepts either ScoredRecord (from Layer 1) or raw TraceRecord;
    the judge only needs the TraceRecord. Passing the scored form makes it
    easy to chain Layer 1 → Layer 2 without re-loading.

    `caller` defaults to a real Anthropic call. Tests pass a stub.

    `concurrency` bounds the number of in-flight judge calls. The default
    (4) keeps load light on the API; bump for large eval banks.

    Returns a JudgedRun. Failed individual calls are recorded as score=0
    with reasoning="<error: ...>" so one bad call doesn't kill the run.
    """
    traces = [_to_trace(r) for r in records]
    judge_caller = caller or _default_anthropic_caller()

    semaphore = asyncio.Semaphore(concurrency)

    async def _judge_one(record: TraceRecord, criterion: JudgeCriterion) -> JudgeResult:
        rendered = render_trace_for_judge(record)
        user_prompt = _build_user_prompt(criterion, rendered)
        max_score = 1 if criterion.scale == "binary" else 3

        async with semaphore:
            try:
                raw = await judge_caller(_JUDGE_SYSTEM_PROMPT, user_prompt, model)
            except Exception as e:  # noqa: BLE001
                return JudgeResult(
                    record_query_id=record.query_id,
                    criterion_name=criterion.name,
                    score=0,
                    max_score=max_score,
                    reasoning=f"<error: {type(e).__name__}: {e}>",
                    raw_response="",
                )

        score, reasoning = parse_judge_response(raw, criterion.scale)
        return JudgeResult(
            record_query_id=record.query_id,
            criterion_name=criterion.name,
            score=score,
            max_score=max_score,
            reasoning=reasoning,
            raw_response=raw,
        )

    tasks: list[Awaitable[JudgeResult]] = []
    for record in traces:
        for criterion in criteria:
            if criterion.applies_to(record):
                tasks.append(_judge_one(record, criterion))

    results = await asyncio.gather(*tasks)
    return JudgedRun(
        label=label,
        results=tuple(results),
        model=model,
        criteria=tuple(criteria),
    )


def _to_trace(record: Any) -> TraceRecord:
    """Accept ScoredRecord or TraceRecord, return TraceRecord."""
    if isinstance(record, TraceRecord):
        return record
    inner = getattr(record, "record", None)
    if isinstance(inner, TraceRecord):
        return inner
    raise TypeError(
        f"expected TraceRecord or ScoredRecord, got {type(record).__name__}"
    )


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def render_judge_run(judged: JudgedRun) -> str:
    """One run's judge results as Markdown.

    Two tables: per-criterion mean across all records, and per-query
    breakdown. Reasoning text is included as collapsible detail (one row
    per result) at the bottom for audit.
    """
    lines: list[str] = []
    lines.append(f"# Layer 2 judge — {judged.label or 'run'}")
    lines.append("")
    lines.append(f"**Judge model:** `{judged.model}`")
    lines.append(f"**Results:** {len(judged.results)}")
    lines.append("")

    means = judged.mean_by_criterion()
    if means:
        lines.append("## Per-criterion mean score (normalised 0–1)")
        lines.append("")
        lines.append("| criterion | mean | n |")
        lines.append("| --- | --- | --- |")
        for name in sorted(means):
            n = len(judged.for_criterion(name))
            lines.append(f"| {name} | {means[name]:.2f} | {n} |")
        lines.append("")

    by_query: dict[str, list[JudgeResult]] = {}
    for r in judged.results:
        by_query.setdefault(r.record_query_id, []).append(r)

    if by_query:
        criterion_order = sorted({r.criterion_name for r in judged.results})
        header = ["query"] + criterion_order
        lines.append("## Per-query scores (score/max)")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for query_id in sorted(by_query):
            row: list[str] = [query_id]
            results_by_name = {r.criterion_name: r for r in by_query[query_id]}
            for crit_name in criterion_order:
                result = results_by_name.get(crit_name)
                row.append(f"{result.score}/{result.max_score}" if result else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if judged.results:
        lines.append("## Reasoning (audit)")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Per-result reasoning</summary>")
        lines.append("")
        for r in sorted(judged.results, key=lambda x: (x.record_query_id, x.criterion_name)):
            lines.append(
                f"- **{r.record_query_id} / {r.criterion_name}** "
                f"({r.score}/{r.max_score}): {r.reasoning}"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_judge_comparison(with_skill: JudgedRun, baseline: JudgedRun) -> str:
    """Side-by-side judge comparison of two runs.

    Renders per-criterion mean for each side and the delta. Per-query
    breakdown with deltas follows.
    """
    lines: list[str] = []
    lines.append("# Layer 2 judge — with-skill vs baseline")
    lines.append("")
    lines.append(f"**Judge model:** `{with_skill.model}`")
    lines.append("")

    ws_means = with_skill.mean_by_criterion()
    bl_means = baseline.mean_by_criterion()
    all_criteria = sorted(set(ws_means) | set(bl_means))

    if all_criteria:
        lines.append("## Per-criterion mean score (normalised 0–1)")
        lines.append("")
        lines.append("| criterion | with skill | baseline | delta |")
        lines.append("| --- | --- | --- | --- |")
        for name in all_criteria:
            ws = ws_means.get(name, 0.0)
            bl = bl_means.get(name, 0.0)
            delta = ws - bl
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {name} | {ws:.2f} | {bl:.2f} | {sign}{delta:.2f} |")
        lines.append("")

    # Per-query, per-criterion breakdown
    all_queries = sorted(
        {r.record_query_id for r in with_skill.results}
        | {r.record_query_id for r in baseline.results}
    )
    if all_queries and all_criteria:
        lines.append("## Per-query, per-criterion (with / baseline)")
        lines.append("")
        header = ["query"] + all_criteria
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        ws_idx = _index_by_query_criterion(with_skill)
        bl_idx = _index_by_query_criterion(baseline)
        for query_id in all_queries:
            row = [query_id]
            for crit_name in all_criteria:
                ws_r = ws_idx.get((query_id, crit_name))
                bl_r = bl_idx.get((query_id, crit_name))
                ws_s = f"{ws_r.score}/{ws_r.max_score}" if ws_r else "—"
                bl_s = f"{bl_r.score}/{bl_r.max_score}" if bl_r else "—"
                row.append(f"{ws_s} / {bl_s}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _index_by_query_criterion(
    judged: JudgedRun,
) -> dict[tuple[str, str], JudgeResult]:
    return {(r.record_query_id, r.criterion_name): r for r in judged.results}
