"""Markdown renderer for eval scoring results.

Two report shapes:

- `render_single_run(scored, run_label)` — one scored run's results as a
  Markdown table, plus per-category pass-rate summary. Useful for local
  inspection of a single run.

- `render_comparison(with_skill, without_skill)` — side-by-side comparison
  of two scored runs (same queries, scored separately). The output is the
  PR-comment shape: per-query delta on pass-rate, plus observed-value
  deltas (audit limit, token cost). This is what `compare_runs.py` calls.

Both reports are designed to be readable in GitHub's PR view (kept narrow,
Markdown tables with one row per query) AND in a terminal (the same output
works as plain text).

The renderer does no I/O — the caller writes the string wherever it needs
to go (stdout, a file, the GitHub Actions PR comment API).
"""
from __future__ import annotations

from collections import defaultdict

from agent_evals.deterministic import CheckResult, Outcome, ScoredRecord


def render_single_run(scored: list[ScoredRecord], run_label: str = "run") -> str:
    """Render one scored run as Markdown.

    Sections:
      1. Header with run label, query count, overall pass-rate
      2. Per-query summary table (one row per query, columns for each check)
      3. Per-category pass-rate breakdown
      4. Observed values (token cost, audit limit, forecast horizon, empties)
      5. Caveats (truncated final_text traces, missing baseline notes)
    """
    if not scored:
        return f"# Eval results — {run_label}\n\n_No traces to score._\n"

    lines: list[str] = []
    lines.append(f"# Eval results — {run_label}")
    lines.append("")
    lines.extend(_render_overall_summary(scored, run_label))
    lines.append("")
    lines.extend(_render_per_query_table(scored))
    lines.append("")
    lines.extend(_render_category_breakdown(scored))
    lines.append("")
    lines.extend(_render_observations(scored))
    lines.append("")
    lines.extend(_render_caveats(scored))
    return "\n".join(lines).rstrip() + "\n"


def render_comparison(
    with_skill: list[ScoredRecord],
    without_skill: list[ScoredRecord],
) -> str:
    """Render a side-by-side comparison of two scored runs.

    The two runs are expected to have the same query_ids (the same query
    bank, run with skills_enabled toggled). Queries present in one but not
    the other are listed under a 'mismatched queries' note.

    The output structure mirrors `render_single_run` but with delta columns:
    pass-rate delta per query, observed-value delta where comparable, token
    cost delta.
    """
    lines: list[str] = []
    lines.append("# Eval results — with-skill vs baseline")
    lines.append("")
    lines.extend(_render_comparison_header(with_skill, without_skill))
    lines.append("")
    lines.extend(_render_comparison_table(with_skill, without_skill))
    lines.append("")
    lines.extend(_render_comparison_observations(with_skill, without_skill))
    lines.append("")
    lines.extend(_render_comparison_caveats(with_skill, without_skill))
    return "\n".join(lines).rstrip() + "\n"


# ----------------------------------------------------------------------------
# Single-run sections
# ----------------------------------------------------------------------------

def _render_overall_summary(scored: list[ScoredRecord], label: str) -> list[str]:
    total_pass = sum(s.pass_count for s in scored)
    total_fail = sum(s.fail_count for s in scored)
    denom = total_pass + total_fail
    rate = (total_pass / denom * 100) if denom else 0.0
    return [
        f"**Run:** `{label}`  ",
        f"**Queries:** {len(scored)}  ",
        f"**Pass rate:** {total_pass}/{denom} ({rate:.1f}%) — NA and OBSERVED excluded",
    ]


def _render_per_query_table(scored: list[ScoredRecord]) -> list[str]:
    """One row per query, columns for each pass/fail check name."""
    # Collect every pass/fail check name in order of first appearance.
    pf_names: list[str] = []
    seen: set[str] = set()
    for s in scored:
        for c in s.checks:
            if c.outcome in (Outcome.PASS, Outcome.FAIL, Outcome.NA) and c.name not in seen:
                pf_names.append(c.name)
                seen.add(c.name)

    header = ["query", "category", "pass/total"] + pf_names
    sep = ["---"] * len(header)
    rows: list[list[str]] = []
    for s in scored:
        check_by_name = {c.name: c for c in s.checks}
        cells = [s.record.query_id, s.record.category, f"{s.pass_count}/{s.pass_count + s.fail_count}"]
        for name in pf_names:
            c = check_by_name.get(name)
            cells.append(_outcome_glyph(c) if c else "")
        rows.append(cells)

    out = ["## Per-query results", ""]
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(sep) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("✓ pass · ✗ fail · — not applicable")
    return out


def _render_category_breakdown(scored: list[ScoredRecord]) -> list[str]:
    by_cat: dict[str, list[ScoredRecord]] = defaultdict(list)
    for s in scored:
        by_cat[s.record.category].append(s)

    out = ["## By category", "", "| category | queries | pass/total | rate |", "| --- | --- | --- | --- |"]
    for cat in sorted(by_cat):
        members = by_cat[cat]
        p = sum(m.pass_count for m in members)
        f = sum(m.fail_count for m in members)
        denom = p + f
        rate = (p / denom * 100) if denom else 0.0
        out.append(f"| {cat} | {len(members)} | {p}/{denom} | {rate:.1f}% |")
    return out


def _render_observations(scored: list[ScoredRecord]) -> list[str]:
    """Group all OBSERVED check results into one section."""
    out = ["## Observations", ""]
    obs_by_name: dict[str, list[tuple[str, CheckResult]]] = defaultdict(list)
    for s in scored:
        for c in s.checks:
            if c.outcome == Outcome.OBSERVED:
                obs_by_name[c.name].append((s.record.query_id, c))

    if not obs_by_name:
        out.append("_No observations recorded._")
        return out

    for obs_name in sorted(obs_by_name):
        out.append(f"### {obs_name}")
        out.append("")
        out.append("| query | value |")
        out.append("| --- | --- |")
        for qid, c in obs_by_name[obs_name]:
            out.append(f"| {qid} | {c.detail} |")
        out.append("")
    return out


def _render_caveats(scored: list[ScoredRecord]) -> list[str]:
    """Surface trace-level limitations that affect scoring."""
    out: list[str] = []
    truncated = [s.record.query_id for s in scored if s.record.final_text_truncated]
    if truncated:
        out.append("## Caveats")
        out.append("")
        out.append(
            f"- {len(truncated)} trace(s) have only the 300-char `final_text_preview` "
            f"and not `final_text_full`. Layer 1 checks are unaffected; Layer 2 "
            f"synthesis-quality scoring will be incomplete until traces are regenerated."
        )
        out.append(f"  Affected: {', '.join(sorted(truncated))}")
    return out


# ----------------------------------------------------------------------------
# Comparison sections
# ----------------------------------------------------------------------------

def _render_comparison_header(
    with_skill: list[ScoredRecord],
    without_skill: list[ScoredRecord],
) -> list[str]:
    w_pass = sum(s.pass_count for s in with_skill)
    w_fail = sum(s.fail_count for s in with_skill)
    b_pass = sum(s.pass_count for s in without_skill)
    b_fail = sum(s.fail_count for s in without_skill)
    w_rate = (w_pass / (w_pass + w_fail) * 100) if (w_pass + w_fail) else 0.0
    b_rate = (b_pass / (b_pass + b_fail) * 100) if (b_pass + b_fail) else 0.0
    return [
        f"**With skill:** {w_pass}/{w_pass + w_fail} ({w_rate:.1f}%)  ",
        f"**Baseline:** {b_pass}/{b_pass + b_fail} ({b_rate:.1f}%)  ",
        f"**Delta:** {w_rate - b_rate:+.1f} percentage points",
    ]


def _render_comparison_table(
    with_skill: list[ScoredRecord],
    without_skill: list[ScoredRecord],
) -> list[str]:
    """One row per query, side-by-side pass/total + delta marker."""
    w_by_id = {s.record.query_id: s for s in with_skill}
    b_by_id = {s.record.query_id: s for s in without_skill}
    all_ids = sorted(set(w_by_id) | set(b_by_id))

    out = [
        "## Per-query comparison",
        "",
        "| query | category | with skill | baseline | delta |",
        "| --- | --- | --- | --- | --- |",
    ]
    for qid in all_ids:
        w = w_by_id.get(qid)
        b = b_by_id.get(qid)
        cat = (w or b).record.category if (w or b) else "?"
        w_cell = f"{w.pass_count}/{w.pass_count + w.fail_count}" if w else "—"
        b_cell = f"{b.pass_count}/{b.pass_count + b.fail_count}" if b else "—"
        if w and b:
            delta_n = w.pass_count - b.pass_count
            if delta_n > 0:
                delta = f"+{delta_n} ✓"
            elif delta_n < 0:
                delta = f"{delta_n} ✗"
            else:
                delta = "—"
        else:
            delta = "_(unpaired)_"
        out.append(f"| {qid} | {cat} | {w_cell} | {b_cell} | {delta} |")
    return out


def _render_comparison_observations(
    with_skill: list[ScoredRecord],
    without_skill: list[ScoredRecord],
) -> list[str]:
    """Side-by-side observed values for queries that exist in both runs."""
    w_by_id = {s.record.query_id: s for s in with_skill}
    b_by_id = {s.record.query_id: s for s in without_skill}
    paired_ids = sorted(set(w_by_id) & set(b_by_id))

    out = ["## Observation deltas", ""]

    # Collect every observation name across both runs in stable order.
    obs_names: list[str] = []
    seen: set[str] = set()
    for s in with_skill + without_skill:
        for c in s.checks:
            if c.outcome == Outcome.OBSERVED and c.name not in seen:
                obs_names.append(c.name)
                seen.add(c.name)

    if not paired_ids or not obs_names:
        out.append("_No paired observations to compare._")
        return out

    for obs_name in obs_names:
        out.append(f"### {obs_name}")
        out.append("")
        out.append("| query | with skill | baseline |")
        out.append("| --- | --- | --- |")
        any_row = False
        for qid in paired_ids:
            w_check = _find_check(w_by_id[qid].checks, obs_name)
            b_check = _find_check(b_by_id[qid].checks, obs_name)
            # Skip queries where neither run produced an observation.
            if w_check is None and b_check is None:
                continue
            if (w_check is None or w_check.outcome == Outcome.NA) and (
                b_check is None or b_check.outcome == Outcome.NA
            ):
                continue
            w_val = w_check.detail if w_check and w_check.outcome == Outcome.OBSERVED else "—"
            b_val = b_check.detail if b_check and b_check.outcome == Outcome.OBSERVED else "—"
            out.append(f"| {qid} | {w_val} | {b_val} |")
            any_row = True
        if not any_row:
            out.append("| _(no observations)_ | | |")
        out.append("")

    # Token-cost-specific summary (the cost-as-first-class-output commitment
    # from EVAL_FINDINGS.md). Compute totals across all paired queries.
    out.extend(_render_token_cost_summary(w_by_id, b_by_id, paired_ids))
    return out


def _render_token_cost_summary(
    w_by_id: dict[str, ScoredRecord],
    b_by_id: dict[str, ScoredRecord],
    paired_ids: list[str],
) -> list[str]:
    """Token cost totals across the paired query set.

    Per EVAL_FINDINGS.md, the SKILL pushes input cost ~10x via the system
    prompt. This summary makes the cost side legible at the run level, not
    just per-query, so the trade-off vs the quality delta is obvious.
    """
    w_input = sum(w_by_id[q].record.total_input_tokens for q in paired_ids)
    w_output = sum(w_by_id[q].record.total_output_tokens for q in paired_ids)
    b_input = sum(b_by_id[q].record.total_input_tokens for q in paired_ids)
    b_output = sum(b_by_id[q].record.total_output_tokens for q in paired_ids)

    def pct(num: int, denom: int) -> str:
        if denom == 0:
            return "n/a"
        return f"{(num / denom - 1) * 100:+.1f}%"

    return [
        "### Token cost (paired queries only)",
        "",
        "| | with skill | baseline | delta |",
        "| --- | --- | --- | --- |",
        f"| total input tokens | {w_input:,} | {b_input:,} | {pct(w_input, b_input)} |",
        f"| total output tokens | {w_output:,} | {b_output:,} | {pct(w_output, b_output)} |",
        f"| total tokens | {w_input + w_output:,} | {b_input + b_output:,} | {pct(w_input + w_output, b_input + b_output)} |",
    ]


def _render_comparison_caveats(
    with_skill: list[ScoredRecord],
    without_skill: list[ScoredRecord],
) -> list[str]:
    out: list[str] = []
    notes: list[str] = []

    w_ids = {s.record.query_id for s in with_skill}
    b_ids = {s.record.query_id for s in without_skill}
    unpaired = (w_ids ^ b_ids)
    if unpaired:
        notes.append(
            f"Unpaired queries (present in only one run): {sorted(unpaired)}. "
            "Their rows show '—' for the missing side."
        )

    w_skill_states = {s.record.skills_enabled for s in with_skill}
    if w_skill_states and w_skill_states != {True}:
        notes.append(
            f"`with_skill` run has mixed skills_enabled states: {sorted(w_skill_states)}. "
            "Check the run was emitted with the SKILL loaded."
        )
    b_skill_states = {s.record.skills_enabled for s in without_skill}
    if b_skill_states and b_skill_states != {False}:
        notes.append(
            f"`without_skill` run has mixed skills_enabled states: {sorted(b_skill_states)}. "
            "Check the run was emitted with no SKILL."
        )

    truncated_w = sum(1 for s in with_skill if s.record.final_text_truncated)
    truncated_b = sum(1 for s in without_skill if s.record.final_text_truncated)
    if truncated_w or truncated_b:
        notes.append(
            f"final_text_full missing on {truncated_w} with-skill and {truncated_b} baseline "
            "trace(s). Layer 1 is unaffected; Layer 2 synthesis-quality scoring will be "
            "incomplete on these until traces are regenerated."
        )

    if notes:
        out.append("## Caveats")
        out.append("")
        for n in notes:
            out.append(f"- {n}")
    return out


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _outcome_glyph(c: CheckResult) -> str:
    """Single-char glyph for the per-query table."""
    return {
        Outcome.PASS: "✓",
        Outcome.FAIL: "✗",
        Outcome.NA: "—",
        Outcome.OBSERVED: "◯",
    }[c.outcome]


def _find_check(checks: list[CheckResult], name: str) -> CheckResult | None:
    for c in checks:
        if c.name == name:
            return c
    return None
