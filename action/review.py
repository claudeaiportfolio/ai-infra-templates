"""Claude security review of a PR diff.

Reads the diff from /tmp/pr_diff.txt, asks Claude for a JSON list of findings,
and writes `review-output=<json>` to $GITHUB_OUTPUT. Invoked by action.yml as a
standalone script so the (large) prompt text doesn't have to live inside a YAML
block scalar.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "claude-sonnet-4-6"
API_URL = "https://api.anthropic.com/v1/messages"


def _write_output(value: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"review-output={value}\n")


def main() -> int:
    diff = Path("/tmp/pr_diff.txt").read_text() if Path("/tmp/pr_diff.txt").exists() else ""
    if not diff.strip():
        print("No relevant files changed — skipping security review")
        _write_output("[]")
        return 0

    prompt = f"""You are a security reviewer specialising in credential handling,
secrets hygiene, and authentication patterns in CI/CD pipelines and Python applications.

Review the following PR diff and identify security issues. Focus exclusively on:

1. CREDENTIAL EXPOSURE
   - Secrets or credentials written to $GITHUB_ENV without ::add-mask::
   - Hardcoded passwords, API keys, private keys, or tokens in any file
   - Secrets passed as plain environment variables instead of being fetched
     at runtime from a secrets manager

2. AUTH ANTI-PATTERNS
   - Service principal client secrets (should use workload identity federation)
   - Password-based authentication where key-pair or managed identity is available
   - Credentials fetched in shell/workflow steps instead of in application code
     using DefaultAzureCredential or equivalent

3. KEY VAULT / SECRETS MANAGER BYPASS
   - Secrets sourced from GitHub secrets directly instead of Key Vault at runtime
   - Secrets stored in .env files committed to the repo
   - Any pattern that moves a secret out of the secrets manager unnecessarily

4. PYTHON SECURITY HYGIENE
   - Bare except clauses that could silently swallow credential errors
   - Missing input validation on functions that accept external input
   - Logging statements that could expose sensitive values

For each issue found, respond with a JSON array. Each item must have:
- "severity": "CRITICAL" or "WARNING"
  - CRITICAL: secret exposure, credential in plaintext, auth bypass
  - WARNING: anti-pattern, missing validation, style issue with security implications
- "title": short title (max 60 chars)
- "description": clear explanation of the issue and why it matters
- "recommendation": specific fix
- "file": filename if identifiable from the diff, otherwise null
- "line_hint": approximate line number if identifiable, otherwise null

If no issues are found, return an empty array: []

Respond with ONLY the JSON array. No preamble, no markdown, no explanation outside the JSON.

PR DIFF:
{diff}"""

    payload = json.dumps(
        {
            "model": MODEL,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    findings: list = []
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read())
            raw = data["content"][0]["text"].strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = "\n".join(raw.split("\n")[:-1])
            findings = json.loads(raw.strip())
            break
        except (json.JSONDecodeError, KeyError) as e:
            if attempt == 1:
                print(f"Failed to parse Claude response after 2 attempts: {e}")
                findings = []
                break
            print(f"Parse attempt {attempt + 1} failed, retrying...")
        except urllib.error.HTTPError as e:
            print(f"API error: {e.code} {e.reason}")
            findings = []
            break

    _write_output(json.dumps(findings))
    print(f"Found {len(findings)} issue(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
