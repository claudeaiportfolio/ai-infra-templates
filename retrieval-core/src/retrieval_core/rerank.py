"""Cross-encoder reranker client (TEI-compatible ``/rerank``).

A cross-encoder scores (query, passage) *jointly*, so it catches relevance a
bi-encoder (the embedding retriever) can't — at a real latency cost, which is
exactly the tradeoff a rerank ON/OFF eval measures. It's typically served as a
warm pod, not the LLM.

``rerank`` is a pass-through (preserves input order, zero scores) when no
``reranker_url`` is given or there are no texts, so a stack runs without a
reranker in dev/CI. ``httpx`` is imported lazily and is an optional dependency
(``pip install retrieval-core[rerank]``) — importing this module costs nothing
if you only use the pure pipeline pieces.
"""

from __future__ import annotations

import asyncio
from typing import Any

from retrieval_core.types import Reranker


async def rerank(
    query: str,
    texts: list[str],
    *,
    reranker_url: str,
    top_n: int | None = None,
    timeout: float = 90.0,
    attempts: int = 3,
    client: Any | None = None,
) -> list[tuple[int, float]]:
    """Return ``[(original_index, score), ...]`` sorted by score descending.

    Pass-through (input order, zero scores) when ``reranker_url`` is empty or
    ``texts`` is empty. ``client`` lets callers inject a pre-built
    ``httpx.AsyncClient`` (e.g. for connection pooling or tests); when omitted a
    short-lived client is created per call.
    """
    n = len(texts)
    if not reranker_url or n == 0:
        keep = n if top_n is None else min(top_n, n)
        return [(i, 0.0) for i in range(keep)]

    results = await _call_reranker(
        query, texts, reranker_url=reranker_url, timeout=timeout, attempts=attempts, client=client
    )
    ranked = sorted(
        ((int(r["index"]), float(r["score"])) for r in results),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked if top_n is None else ranked[:top_n]


def make_reranker(
    reranker_url: str,
    *,
    timeout: float = 90.0,
    attempts: int = 3,
    client: Any | None = None,
) -> Reranker:
    """Bind a ``reranker_url`` into a :data:`~retrieval_core.types.Reranker`
    callable suitable for passing to :func:`~retrieval_core.retrieval.retrieve`.
    """

    async def _reranker(q: str, texts: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
        return await rerank(
            q,
            texts,
            reranker_url=reranker_url,
            top_n=top_n,
            timeout=timeout,
            attempts=attempts,
            client=client,
        )

    return _reranker


async def _call_reranker(
    query: str,
    texts: list[str],
    *,
    reranker_url: str,
    timeout: float,
    attempts: int,
    client: Any | None,
) -> list[dict[str, Any]]:
    """POST to the TEI-compatible ``/rerank`` endpoint, retrying transient
    transport faults with exponential backoff. CPU cross-encoders are slow
    (queue + inference), so the default timeout is generous; the pipeline
    degrades to fused order if this still fails.
    """
    import httpx  # lazy: optional dependency

    transient = (
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
    )
    url = f"{reranker_url.rstrip('/')}/rerank"
    payload = {"query": query, "texts": texts, "truncate": True}

    async def _post(c: Any) -> list[dict[str, Any]]:
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = await c.post(url, json=payload)
                resp.raise_for_status()
                data: list[dict[str, Any]] = resp.json()
                return data
            except transient as exc:  # 4xx/5xx are NOT transient — they re-raise
                last_exc = exc
                if attempt + 1 < attempts:
                    await asyncio.sleep(min(2**attempt, 8))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("unreachable: retry loop exited without a result or error")

    if client is not None:
        return await _post(client)
    async with httpx.AsyncClient(timeout=timeout) as owned:
        return await _post(owned)
