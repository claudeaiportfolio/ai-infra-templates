"""Post the Claude security-review findings as a PR comment.

Reads the findings JSON from $FINDINGS and posts a formatted comment to the PR.
Invoked by action.yml as a standalone script.
"""

import json
import os
import urllib.request


def _finding_block(f: dict) -> list[str]:
    loc = f"**{f['file']}**" if f.get("file") else "Unknown file"
    if f.get("line_hint"):
        loc += f" (line ~{f['line_hint']})"
    return [
        f"#### {f['title']}",
        f"**Location:** {loc}",
        f"**Issue:** {f['description']}",
        f"**Fix:** {f['recommendation']}",
        "",
    ]


def main() -> int:
    findings = json.loads(os.environ["FINDINGS"])
    diff_size = int(os.environ.get("DIFF_SIZE", 0))

    critical = [f for f in findings if f["severity"] == "CRITICAL"]
    warnings = [f for f in findings if f["severity"] == "WARNING"]

    lines = ["## 🔍 Claude Security Review", ""]
    if diff_size >= 39000:
        lines += [
            "> ⚠️ **Note:** Diff was truncated at 40KB. Large PRs may have incomplete coverage.",
            "",
        ]
    if critical:
        lines += ["### 🔴 Critical findings — must fix before merge", ""]
        for f in critical:
            lines += _finding_block(f)
    if warnings:
        lines += ["### 🟡 Warnings — advisory", ""]
        for f in warnings:
            lines += _finding_block(f)
    lines += [
        "---",
        f"*Reviewed by Claude · {len(findings)} finding(s) · "
        "[Design decisions](../../blob/main/docs/design.md)*",
    ]

    payload = json.dumps({"body": "\n".join(lines)}).encode()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}"
        f"/issues/{os.environ['PR_NUMBER']}/comments",
        data=payload,
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        print(f"Comment posted: {resp.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
