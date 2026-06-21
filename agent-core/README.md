# agent-core

A reusable **agent runtime** for the `claudeaiportfolio` MCP agents — one source of
truth instead of copy-pasting the loop per project (centralise-don't-copy).
Generalised from `snowflake-forecasting`'s loan-portfolio agent; consumed as a
git-subdirectory dependency.

It's a small, hand-written **plan → act → observe** loop (not a framework):

- **`AgentLoop`** — bounded turns (`max_turns`), clean stop-reason handling, and
  **bail-on-repeated-errors** (same tool fails twice → stop, don't thrash). Anthropic
  Messages API with optional **prompt caching** (`cache_control: ephemeral`) and
  cache-token accounting.
- **`MCPClient`** — streamable-HTTP MCP session, tool discovery, per-tool timeout;
  tool failures surface as `ERROR: …` strings fed back to the model, not exceptions.
- **`Auth0M2MClient`** — OAuth2 `client_credentials` with async-safe token caching.
- **`Tracer`** — append-only JSONL events (eval-consumable) + OTel GenAI/MCP spans.
- **`SkillLoader`** — `SKILL.md` progressive disclosure (inventory always on, bodies
  loaded per selector).

Domain specifics — the system prompt, the tools (served by your MCP server), and config
defaults — are supplied by the consuming project. `agent-core` ships none.

## Install

```toml
dependencies = [
  "agent-core @ git+https://github.com/claudeaiportfolio/ai-infra-templates.git@agent-core-v0.1.0#subdirectory=agent-core",
]
```

## Use

```python
from agent_core import AgentLoop, MCPClient, Tracer

mcp = MCPClient.from_env()            # MCP_SERVER_URL + AUTH0_M2M_* env
async with mcp.session():
    loop = AgentLoop(mcp=mcp, tracer=Tracer(), system_prompt=MY_PLANNER_PROMPT)
    result = await loop.run("How do KEDA and the HPA differ?")
    print(result.final_text, result.turns, result.tool_calls)
```

The JSONL trace at `result.trace_path` feeds `agent-evals` (Layer 1/2 scoring).

## Versioning

Tagged `agent-core-vX.Y.Z`; consumers pin to a tag.
