"""MCP client — connects to an MCP server over streamable HTTP.

Wraps the MCP SDK's streamable HTTP client with Auth0 M2M auth. The
agent runs as its own machine identity, not on behalf of a user.

Responsibilities:
  1. List tools the server exposes in Anthropic tool-schema shape.
  2. Call a tool by name with a dict of args and return the result as
     a plain string for a `tool_result` content block.

The MCP streamable-HTTP client takes headers at connection time; the
M2M token is fetched once when the session opens.

Every call_tool opens an OTel client span following the MCP semantic
convention ("tools/call <name>" span name, mcp.method.name="tools/call",
gen_ai.tool.name=<name>, gen_ai.operation.name="execute_tool"). This
sits inside the chat span the loop opens, and is the agent-side mate
to the server-side span emitted by the MCP server — together they form
the full distributed trace for one tool call.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from opentelemetry import trace as otel_trace

from agent_core.telemetry import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_TOOL_NAME,
    MCP_METHOD_NAME,
    get_tracer,
)

from .auth import Auth0M2MClient

_tracer = get_tracer(__name__)

# Default per-tool-call timeout. Long enough for a slow backend tool call
# but short enough that a hung connection fails fast rather than holding up
# the whole loop.
DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0


@dataclass
class ToolSchema:
    """A tool in the shape Anthropic's Messages API expects."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class MCPClient:
    """Thin wrapper over an MCP streamable-HTTP session, authed via Auth0 M2M."""

    def __init__(self, url: str, auth: Auth0M2MClient) -> None:
        self.url = url
        self.auth = auth
        self._session: ClientSession | None = None
        self._tools: list[ToolSchema] = []

    @classmethod
    def from_env(cls) -> MCPClient:
        """Build from environment variables.

        MCP_SERVER_URL is required. Auth0 vars consumed by Auth0M2MClient.from_env().
        """
        url = os.environ.get("MCP_SERVER_URL")
        if not url:
            raise ValueError("MCP_SERVER_URL is required")
        auth = Auth0M2MClient.from_env()
        return cls(url=url, auth=auth)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[MCPClient]:
        """Open the streamable-HTTP session and discover tools once."""
        headers = await self.auth.auth_header()
        async with streamablehttp_client(self.url, headers=headers) as (
            read,
            write,
            _get_session_id,
        ), ClientSession(read, write) as session:
            await session.initialize()
            self._session = session
            self._tools = await self._discover_tools()
            try:
                yield self
            finally:
                self._session = None

    async def _discover_tools(self) -> list[ToolSchema]:
        assert self._session is not None
        result = await self._session.list_tools()
        return [
            ToolSchema(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema or {"type": "object", "properties": {}},
            )
            for t in result.tools
        ]

    @property
    def tools(self) -> list[ToolSchema]:
        return self._tools

    def tools_for_anthropic(self) -> list[dict[str, Any]]:
        return [t.to_anthropic() for t in self._tools]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    ) -> str:
        """Call a tool and return its content as a single string.

        Surfaces three distinct failure modes as "ERROR: ..." strings so
        the agent loop can pass them back to Claude as tool_result blocks
        with is_error=True, rather than letting them bubble up and kill
        the session:

          - MCP-level errors (result.isError) — e.g. invalid arguments.
          - Timeouts — tool didn't respond within timeout_seconds.
          - Connection errors — HTTP connection failed mid-call (server
            restart, network blip). Caught here so one bad call doesn't
            take out the rest of the loop.

        Emits an OTel client span ("tools/call <name>") so the agent-side
        invocation shows up in the trace alongside the server-side span
        from the MCP server. The span is in error state on timeout / MCP
        error / exception, regardless of whether the string returned to
        the loop is well-formed.
        """
        if self._session is None:
            raise RuntimeError("MCPClient not connected. Use `async with client.session():`")
        with _tracer.start_as_current_span(f"tools/call {name}") as span:
            span.set_attribute(MCP_METHOD_NAME, "tools/call")
            span.set_attribute(GEN_AI_TOOL_NAME, name)
            span.set_attribute(GEN_AI_OPERATION_NAME, "execute_tool")
            try:
                result = await asyncio.wait_for(
                    self._session.call_tool(name, arguments),
                    timeout=timeout_seconds,
                )
            except TimeoutError as exc:
                span.record_exception(exc)
                span.set_status(
                    otel_trace.StatusCode.ERROR,
                    f"timeout after {timeout_seconds}s",
                )
                return f"ERROR: tool call timed out after {timeout_seconds}s"
            except Exception as exc:  # noqa: BLE001 — surface anything else to the model as a tool error
                span.record_exception(exc)
                span.set_status(otel_trace.StatusCode.ERROR, str(exc))
                return f"ERROR: {type(exc).__name__}: {exc}"

            if result.isError:
                err = self._content_to_text(result.content)
                span.set_status(otel_trace.StatusCode.ERROR, err[:200])
                return f"ERROR: {err}"
            return self._content_to_text(result.content)

    @staticmethod
    def _content_to_text(content: list[Any]) -> str:
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(str(block))
        return "\n".join(parts)
