"""agent-core — a reusable agent runtime for portfolio MCP agents.

A small, hand-written plan/act/observe loop (not a framework): bounded turns,
bail-on-repeated-errors, MCP tool dispatch, Auth0 M2M auth, dual tracing
(JSONL for evals + OTel GenAI/MCP spans), Anthropic prompt caching, and
SKILL.md progressive disclosure. Domain specifics (system prompt, tools) are
supplied by the consuming project.
"""

from __future__ import annotations

from .auth import Auth0M2MClient, AuthError
from .loop import DEFAULT_SYSTEM_PROMPT, AgentLoop, LoopResult
from .mcp_client import MCPClient, ToolSchema
from .skills import Skill, SkillLoader
from .telemetry import get_tracer, setup_telemetry
from .tracing import Tracer

__version__ = "0.1.1"

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "AgentLoop",
    "Auth0M2MClient",
    "AuthError",
    "LoopResult",
    "MCPClient",
    "Skill",
    "SkillLoader",
    "ToolSchema",
    "Tracer",
    "get_tracer",
    "setup_telemetry",
    "__version__",
]
