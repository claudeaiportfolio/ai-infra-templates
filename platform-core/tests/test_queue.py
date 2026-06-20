"""Pure-logic tests for the queue connection settings (no live Redis)."""

import ssl
from unittest.mock import patch

import pytest

from platform_core.queue import RedisSettings, redis_connection


def test_from_url_plain():
    s = RedisSettings.from_url("redis://cache:6380/2")
    assert (s.host, s.port, s.db, s.use_tls) == ("cache", 6380, 2, False)


def test_from_url_tls_with_credentials():
    s = RedisSettings.from_url("rediss://worker:secret@redis.svc:6379/0")
    assert s.use_tls is True
    assert s.username == "worker"
    assert s.password == "secret"


def test_from_url_defaults():
    s = RedisSettings.from_url("redis://localhost")
    assert (s.port, s.db) == (6379, 0)


def test_from_url_overrides_win():
    # Password/CA injected out-of-band (from a KV-synced secret), not in the URL.
    s = RedisSettings.from_url("rediss://redis.svc:6379", password="kv-secret", ssl_ca_certs="/tls/ca.crt")
    assert s.password == "kv-secret"
    assert s.ssl_ca_certs == "/tls/ca.crt"
    assert s.use_tls is True


def test_from_url_rejects_bad_scheme():
    with pytest.raises(ValueError, match="unsupported redis scheme"):
        RedisSettings.from_url("http://nope")


def test_connection_passes_tls_params():
    captured = {}

    class _Fake:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    s = RedisSettings(host="r", use_tls=True, ssl_ca_certs="/tls/ca.crt", password="p", username="u")
    with patch("platform_core.queue.redis.Redis", _Fake):
        redis_connection(s)
    assert captured["ssl"] is True
    assert captured["ssl_cert_reqs"] == ssl.CERT_REQUIRED
    assert captured["ssl_ca_certs"] == "/tls/ca.crt"
    assert captured["username"] == "u"
    assert captured["password"] == "p"


def test_connection_omits_tls_when_disabled():
    captured = {}

    class _Fake:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("platform_core.queue.redis.Redis", _Fake):
        redis_connection(RedisSettings(host="r"))
    assert "ssl" not in captured
