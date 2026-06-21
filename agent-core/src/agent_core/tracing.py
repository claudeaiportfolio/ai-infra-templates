"""Structured event tracing for the agent loop.

Every event is written as a single JSON line to a per-run trace file
under traces/. Format is deliberately simple — jq and grep are the
analysis tools. The eval suite (next sprint) consumes these files
directly, so they're structured from day one.
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Tracer:
    """Append-only JSON-lines tracer with stderr mirroring.

    Each run gets its own file: traces/run-<timestamp>-<short_id>.jsonl.
    run_id is stamped onto every event for downstream filtering.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    trace_dir: Path = field(default_factory=lambda: Path("traces"))
    mirror_to_stderr: bool = True
    _start: float = field(default_factory=time.monotonic)
    _path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self._path = self.trace_dir / f"run-{stamp}-{self.run_id}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def event(self, kind: str, **fields: Any) -> None:
        """Record one event. kind is a short snake_case string."""
        payload = {
            "ts": round(time.monotonic() - self._start, 4),
            "run_id": self.run_id,
            "kind": kind,
            **fields,
        }
        line = json.dumps(payload, default=str, ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.mirror_to_stderr:
            self._pretty(payload)

    @staticmethod
    def _pretty(p: dict[str, Any]) -> None:
        kind = p["kind"]
        ts = p["ts"]
        if kind == "turn_start":
            print(f"\n[{ts:7.2f}s] ── turn {p.get('turn')} ────────────────────────────", file=sys.stderr)
        elif kind == "claude_response":
            text = p.get("text_preview") or ""
            print(f"[{ts:7.2f}s] claude: {text}", file=sys.stderr)
        elif kind == "tool_use":
            print(f"[{ts:7.2f}s]   → {p['tool']}({p.get('input')})", file=sys.stderr)
        elif kind == "tool_result":
            rows = p.get("rows")
            preview_len = p.get("preview_len")
            if rows is not None:
                n = rows
            elif preview_len is not None:
                n = preview_len
            else:
                n = "?"
            print(f"[{ts:7.2f}s]   ← {p['tool']}: {n} chars", file=sys.stderr)
        elif kind == "tool_error":
            print(f"[{ts:7.2f}s]   ✗ {p['tool']}: {p.get('error')}", file=sys.stderr)
        elif kind == "stop_reason":
            print(f"[{ts:7.2f}s] stop_reason={p['reason']}", file=sys.stderr)
        elif kind == "loop_end":
            print(
                f"[{ts:7.2f}s] ── done "
                f"(turns={p.get('turns')}, tools={p.get('tool_calls')}) ──",
                file=sys.stderr,
            )
        else:
            extras = {k: v for k, v in p.items() if k not in {"ts", "run_id", "kind"}}
            print(f"[{ts:7.2f}s] {kind}: {json.dumps(extras, default=str)}", file=sys.stderr)
