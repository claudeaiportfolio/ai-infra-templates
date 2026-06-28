"""Tests for the reranker client. The HTTP client is injected as a fake — no
network calls. (httpx is a dev/test dependency, used only for its exception
types in the transient-retry path.)"""
from __future__ import annotations

import httpx
import pytest

from retrieval_core import make_reranker, rerank


class FakeResponse:
    def __init__(self, payload: list[dict]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[dict]:
        return self._payload


class FakeClient:
    """Records calls and replays a scripted sequence of responses/exceptions."""

    def __init__(self, *results: object) -> None:
        self._results = list(results)
        self.calls: list[dict] = []

    async def post(self, url: str, json: dict) -> FakeResponse:
        self.calls.append({"url": url, "json": json})
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, FakeResponse)
        return result


class TestPassThrough:
    async def test_no_url_preserves_order(self) -> None:
        out = await rerank("q", ["a", "b", "c"], reranker_url="")
        assert out == [(0, 0.0), (1, 0.0), (2, 0.0)]

    async def test_no_url_respects_top_n(self) -> None:
        out = await rerank("q", ["a", "b", "c"], reranker_url="", top_n=2)
        assert out == [(0, 0.0), (1, 0.0)]

    async def test_empty_texts(self) -> None:
        out = await rerank("q", [], reranker_url="http://reranker")
        assert out == []


class TestScoring:
    async def test_sorts_by_score_desc(self) -> None:
        client = FakeClient(
            FakeResponse([{"index": 0, "score": 0.1}, {"index": 1, "score": 0.9}, {"index": 2, "score": 0.5}])
        )
        out = await rerank("q", ["a", "b", "c"], reranker_url="http://reranker", client=client)
        assert out == [(1, 0.9), (2, 0.5), (0, 0.1)]

    async def test_top_n_truncates(self) -> None:
        client = FakeClient(
            FakeResponse([{"index": 0, "score": 0.1}, {"index": 1, "score": 0.9}, {"index": 2, "score": 0.5}])
        )
        out = await rerank("q", ["a", "b", "c"], reranker_url="http://reranker", top_n=1, client=client)
        assert out == [(1, 0.9)]

    async def test_posts_to_rerank_path(self) -> None:
        client = FakeClient(FakeResponse([{"index": 0, "score": 1.0}]))
        await rerank("hello", ["a"], reranker_url="http://reranker/", client=client)
        assert client.calls[0]["url"] == "http://reranker/rerank"
        assert client.calls[0]["json"] == {"query": "hello", "texts": ["a"], "truncate": True}


class TestRetry:
    async def test_retries_transient_then_succeeds(self) -> None:
        client = FakeClient(
            httpx.ReadTimeout("slow"),
            FakeResponse([{"index": 0, "score": 1.0}]),
        )
        out = await rerank("q", ["a"], reranker_url="http://reranker", attempts=3, client=client)
        assert out == [(0, 1.0)]
        assert len(client.calls) == 2

    async def test_gives_up_after_attempts(self) -> None:
        client = FakeClient(
            httpx.ReadTimeout("1"), httpx.ReadTimeout("2"), httpx.ReadTimeout("3")
        )
        with pytest.raises(httpx.ReadTimeout):
            await rerank("q", ["a"], reranker_url="http://reranker", attempts=3, client=client)
        assert len(client.calls) == 3

    async def test_non_transient_status_error_not_retried(self) -> None:
        request = httpx.Request("POST", "http://reranker/rerank")
        response = httpx.Response(500, request=request)
        client = FakeClient(httpx.HTTPStatusError("boom", request=request, response=response))
        with pytest.raises(httpx.HTTPStatusError):
            await rerank("q", ["a"], reranker_url="http://reranker", attempts=3, client=client)
        assert len(client.calls) == 1  # not retried


class TestMakeReranker:
    async def test_binds_url(self) -> None:
        client = FakeClient(FakeResponse([{"index": 0, "score": 0.7}]))
        reranker = make_reranker("http://reranker", client=client)
        out = await reranker("q", ["a"], 5)
        assert out == [(0, 0.7)]
