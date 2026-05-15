# Security Review Action — Design Decisions

## What this is

A reusable GitHub Actions composite action that runs a Claude-powered security
review on every PR. Scoped specifically to credential handling and secrets
hygiene — not a general code reviewer.

Designed to be referenced from any `claudeaiportfolio` repo without
modification.

---

## Design decisions

### Soft fail (warn, don't block)

**Decision:** `CRITICAL` severity posts a PR comment and sets a failed check.
`WARNING` posts a comment only, does not block merge.

**Reasoning:** Portfolio repos iterate fast. A hard block on every `WARNING`
would create friction that discourages use of the action. The value is in
surfacing issues early, not in being a gatekeeper. In a production bank
environment you'd invert this — `WARNING` blocks too — but for a solo
portfolio the signal matters more than the enforcement.

**How to harden for production:** change `fail-on-warning: false` input to
`true` in the calling workflow.

---

### Single API call, full diff

**Decision:** The entire PR diff is sent in one Anthropic API call rather than
per-file calls.

**Reasoning:** For the file sizes in this portfolio (workflow YAML, Python
scripts) a full diff fits comfortably within a single context window. Per-file
calls would give more precise line-level attribution but at meaningfully higher
cost and complexity. The tradeoff: if a diff exceeds ~50KB the prompt should
be chunked — not implemented here, noted as a known limitation.

**Known limitation:** Very large PRs (many files, generated code) will produce
less precise feedback. Add a diff size check and fallback to per-file if
needed.

---

### Structured JSON output from Claude

**Decision:** Claude is prompted to return a JSON array of findings, not
free-form text.

**Reasoning:** Structured output lets the action parse severity levels
programmatically and route accordingly (comment vs fail). Free-form text
would require fragile regex or NLP to extract severity. JSON parsing is
deterministic and cheap.

**Retry logic:** The action retries once on JSON parse failure before giving
up. Claude occasionally produces malformed JSON on complex diffs.

---

### Scoped review prompt

**Decision:** The review prompt is narrowly scoped to credential and secrets
hygiene, not general code quality.

**Reasoning:** A general code reviewer is noisy and expensive. An interviewer
reading this action wants to see that you understand the specific problem
domain, not that you can call an LLM and dump the output. Narrow scope also
means the signal-to-noise ratio is high enough to act on.

**Scope of review:**
- Credentials or secrets in `$GITHUB_ENV` without `::add-mask::`
- Hardcoded secrets, passwords, or API keys in any file
- Auth patterns that should be rejected (service principals with client
  secrets, passwords instead of key-pair or workload identity)
- Key Vault bypass (secrets passed as env vars instead of fetched at runtime)
- Python: broad exception catches that could hide credential errors
- Python: missing input validation on MCP tool functions

---

### Composite action, not a reusable workflow

**Decision:** Implemented as a composite action (`action.yml`) not a reusable
workflow (`workflow_call`).

**Reasoning:** Composite actions are referenced with `uses:` in any job step,
making them more flexible. Reusable workflows require a dedicated job and
can't be composed with other steps as naturally. For a security check that
needs to run alongside other steps, composite is the right primitive.

---

### ANTHROPIC_API_KEY as a required input

**Decision:** The API key is passed as an input from the calling workflow's
secrets, not hardcoded or fetched from Key Vault inside the action.

**Reasoning:** The action is generic and doesn't know which Key Vault the
calling repo uses. Passing the key as an input keeps the action portable.
The calling workflow is responsible for sourcing it appropriately — from
GitHub org secrets in this portfolio, from Key Vault in a production
environment.

---

## Known limitations

- Diffs > ~50KB may produce imprecise feedback (no chunking implemented)
- JSON parse failure falls back to a warning comment, not a hard fail
- Does not check binary files or generated code
- Line numbers in findings are approximate (Claude infers from diff context)