"""Auth0 M2M (machine-to-machine) authentication.

The agent runs as its own client identity, not on behalf of a user.
Exchanges client_id + client_secret for a JWT via OAuth 2.0 client
credentials, then presents it to the MCP server. The MCP server
validates it via JWKS exactly as it does for user-delegated tokens.

Secrets model:
  - Locally: values in .env (gitignored), loaded by the caller.
  - Kubernetes: Workload Identity → Key Vault → CSI driver → env vars.
  - GitHub Actions: workload identity federation → Key Vault at job start.
  Code here is environment-agnostic — reads env vars only.

Token caching: Auth0 M2M tokens are valid for ~24 hours. We cache per
instance and refresh 60 seconds before expiry to avoid boundary races.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


class AuthError(RuntimeError):
    """Raised for any failure in the M2M token exchange.

    Surfaced as a distinct category so the agent loop can tell auth
    failures apart from MCP errors or Claude API errors in traces.
    """


@dataclass
class _CachedToken:
    access_token: str
    expires_at: float  # monotonic time
    token_type: str = "Bearer"

    def is_fresh(self, buffer_seconds: float = 60.0) -> bool:
        return time.monotonic() < (self.expires_at - buffer_seconds)


@dataclass
class Auth0M2MClient:
    """OAuth 2.0 client credentials flow against Auth0.

    Usage:
        auth = Auth0M2MClient.from_env()
        token = await auth.get_token()  # JWT string, auto-refreshed
    """

    domain: str
    client_id: str
    client_secret: str
    audience: str
    scope: str | None = None
    timeout_seconds: float = 10.0

    _cache: _CachedToken | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @classmethod
    def from_env(cls) -> Auth0M2MClient:
        """Build from environment variables.

        Required: AUTH0_DOMAIN, AUTH0_M2M_CLIENT_ID,
                  AUTH0_M2M_CLIENT_SECRET, AUTH0_M2M_AUDIENCE
        Optional: AUTH0_M2M_SCOPE (space-separated)
        """
        missing = [
            name
            for name in (
                "AUTH0_DOMAIN",
                "AUTH0_M2M_CLIENT_ID",
                "AUTH0_M2M_CLIENT_SECRET",
                "AUTH0_M2M_AUDIENCE",
            )
            if not os.environ.get(name)
        ]
        if missing:
            raise AuthError(
                "Missing required Auth0 M2M env vars: "
                + ", ".join(missing)
                + ". Locally these come from .env (gitignored); in K8s "
                "from Workload Identity → Key Vault → CSI driver; in CI "
                "from workload identity federation."
            )
        return cls(
            domain=os.environ["AUTH0_DOMAIN"],
            client_id=os.environ["AUTH0_M2M_CLIENT_ID"],
            client_secret=os.environ["AUTH0_M2M_CLIENT_SECRET"],
            audience=os.environ["AUTH0_M2M_AUDIENCE"],
            scope=os.environ.get("AUTH0_M2M_SCOPE") or None,
        )

    @property
    def _token_url(self) -> str:
        return f"https://{self.domain}/oauth/token"

    async def get_token(self) -> str:
        """Return a fresh JWT, fetching from Auth0 if necessary.

        Concurrent calls share a single in-flight fetch via the lock.
        """
        if self._cache is not None and self._cache.is_fresh():
            return self._cache.access_token

        async with self._lock:
            if self._cache is not None and self._cache.is_fresh():
                return self._cache.access_token
            self._cache = await self._fetch_token()
            return self._cache.access_token

    async def _fetch_token(self) -> _CachedToken:
        payload: dict = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "audience": self.audience,
        }
        if self.scope:
            payload["scope"] = self.scope

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as http:
                response = await http.post(self._token_url, json=payload)
        except httpx.HTTPError as e:
            raise AuthError(f"Auth0 token request failed: {type(e).__name__}: {e}") from e

        if response.status_code != 200:
            try:
                body = response.json()
            except ValueError:
                body = {"raw": response.text[:500]}
            raise AuthError(
                f"Auth0 token request returned {response.status_code}: {body}"
            )

        data = response.json()
        access_token = data.get("access_token")
        expires_in = data.get("expires_in")
        token_type = data.get("token_type", "Bearer")
        if not access_token or not expires_in:
            raise AuthError(f"Auth0 response missing access_token or expires_in: {data}")

        logger.info("event=m2m_token_refreshed expires_in=%s", expires_in)
        return _CachedToken(
            access_token=access_token,
            expires_at=time.monotonic() + float(expires_in),
            token_type=token_type,
        )

    async def auth_header(self) -> dict[str, str]:
        token = await self.get_token()
        return {"Authorization": f"{self._cache.token_type} {token}"}
