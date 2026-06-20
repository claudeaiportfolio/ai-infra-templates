"""RQ-on-Redis queue plumbing for portfolio background-worker services.

One TLS-capable Redis connection factory plus RQ ``Queue``/``Worker`` builders,
so every service (RAG ingestion today, future pipelines) shares the same
hardened queue setup instead of re-implementing connection / TLS / ACL handling.

Secrets (``password``) come from the environment — a Key Vault-synced Kubernetes
Secret — never hard-coded. ``rediss://`` URLs (or ``use_tls=True``) enable
in-cluster TLS; point ``ssl_ca_certs`` at the mounted CA bundle.

Install via the ``platform-core[queue]`` extra.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from urllib.parse import urlparse

import redis
from rq import Queue, Retry, Worker

__all__ = [
    "RedisSettings",
    "Retry",
    "build_worker",
    "get_queue",
    "redis_connection",
]


@dataclass(frozen=True)
class RedisSettings:
    """Connection settings for a Redis instance.

    ``username``/``password`` drive Redis ACL auth (scope each consumer to the
    least privilege it needs). ``use_tls`` turns on in-cluster TLS verified
    against ``ssl_ca_certs``.
    """

    host: str
    port: int = 6379
    db: int = 0
    username: str | None = None
    password: str | None = None
    use_tls: bool = False
    ssl_ca_certs: str | None = None
    socket_timeout: float = 30.0
    socket_connect_timeout: float = 10.0

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        password: str | None = None,
        ssl_ca_certs: str | None = None,
    ) -> RedisSettings:
        """Parse a ``redis://`` or ``rediss://`` URL; ``rediss`` implies TLS.

        ``password`` / ``ssl_ca_certs`` are injected out-of-band (from a Key
        Vault-synced secret / mounted CA bundle) rather than embedded in the URL;
        an explicit ``password`` wins over any in the URL.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("redis", "rediss"):
            raise ValueError(f"unsupported redis scheme: {parsed.scheme!r}")
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            db=int(parsed.path.lstrip("/") or 0),
            username=parsed.username,
            password=password if password is not None else parsed.password,
            use_tls=parsed.scheme == "rediss",
            ssl_ca_certs=ssl_ca_certs,
        )


def redis_connection(settings: RedisSettings) -> redis.Redis:
    """Build a Redis client from ``settings`` (TLS requires a valid CA)."""
    kwargs: dict[str, object] = {
        "host": settings.host,
        "port": settings.port,
        "db": settings.db,
        "username": settings.username,
        "password": settings.password,
        "socket_timeout": settings.socket_timeout,
        "socket_connect_timeout": settings.socket_connect_timeout,
        "health_check_interval": 30,
    }
    if settings.use_tls:
        kwargs.update(
            ssl=True,
            ssl_cert_reqs=ssl.CERT_REQUIRED,
            ssl_ca_certs=settings.ssl_ca_certs,
        )
    return redis.Redis(**kwargs)  # type: ignore[arg-type]


def get_queue(name: str, connection: redis.Redis) -> Queue:
    """Return the RQ ``Queue`` named ``name`` on ``connection``."""
    return Queue(name, connection=connection)


def build_worker(queue_names: list[str], connection: redis.Redis) -> Worker:
    """Build an RQ ``Worker`` listening on ``queue_names``."""
    queues = [Queue(name, connection=connection) for name in queue_names]
    return Worker(queues, connection=connection)
