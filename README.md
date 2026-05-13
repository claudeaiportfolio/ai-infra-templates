# claude-security-review

A reusable GitHub Actions composite action that reviews PR diffs for credential
handling issues, secrets hygiene, and authentication anti-patterns using Claude.

Part of the `claudeaiportfolio` infrastructure — referenced from all portfolio
repos as a shared security gate.

## What it checks

- Secrets written to `$GITHUB_ENV` without `::add-mask::`
- Hardcoded credentials, API keys, or private keys
- Auth anti-patterns (service principal secrets, passwords over managed identity)
- Key Vault bypass — secrets passed as env vars instead of fetched at runtime
- Python: bare exception catches that could hide credential errors
- Python: missing input validation on externally-facing functions

## What it doesn't do

General code review, style checking, or test coverage analysis. Narrow scope
is intentional — see [design decisions](docs/design.md).

## Usage

Add to any repo's `.github/workflows/` directory:

```yaml
name: Security Review

on:
  pull_request:
    paths:
      - '**.py'
      - '**.yml'
      - '**.yaml'
      - '**.toml'
      - '**.sh'

permissions:
  contents: read
  pull-requests: write

jobs:
  security-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: claudeaiportfolio/claude-security-review/action@main
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          github-token: ${{ secrets.GITHUB_TOKEN }}
          fail-on-warning: "false"
```

Add `ANTHROPIC_API_KEY` to your org secrets and the action is available to
all repos without further configuration.

## Severity levels

| Severity | Example | Behaviour |
|---|---|---|
| CRITICAL | Private key in `$GITHUB_ENV` | Posts comment + fails check |
| WARNING | Bare except clause | Posts comment only |

Set `fail-on-warning: "true"` to block on warnings too (recommended for
production environments).

## Design decisions

See [docs/design.md](docs/design.md) for reasoning behind each design choice —
including the soft-fail default, single API call approach, and structured JSON
output format.

## Cost

Approximately £0.001–0.003 per PR review at current Sonnet 4 pricing.
Diffs are capped at 40KB to control cost and stay within context limits.