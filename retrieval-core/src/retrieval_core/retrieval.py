"""Hybrid retrieval orchestration: vector + (optional) keyword, fused with RRF,
with a freshness tiebreak and an optional cross-encoder rerank step.

The pipeline is fully determinate: retrieve candidates from each arm -> fuse ->
(optional) rerank -> return top-k. No step's behaviour depends on what an
earlier step *found*; only on the injected configuration. The datastore and the
reranker are injected as callables (see ``retrieval_core.types``), so this module
contains the reusable ordering logic and nothing store-specific.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from retrieval_core.config import RetrievalConfig
from retrieval_core.fusion import freshness_boost, rrf_fuse
from retrieval_core.types import (
    Candidate,
    FetchCandidates,
    KeywordSearch,
    Reranker,
    VectorSearch,
)

logger = logging.getLogger(__name__)


async def retrieve(
    conn: object,
    *,
    qvec: object,
    query: str,
    config: RetrievalConfig,
    vector_search: VectorSearch,
    fetch_candidates: FetchCandidates,
    keyword_search: KeywordSearch | None = None,
    reranker: Reranker | None = None,
    now: datetime | None = None,
) -> list[Candidate]:
    """Run the hybrid pipeline and return the final top-k candidates.

    Args:
        conn: Opaque datastore handle, passed through to the injected callables.
        qvec: Opaque encoded query vector, passed through to ``vector_search``.
        query: Raw query text, used for keyword search and reranking.
        config: Frozen pipeline configuration.
        vector_search: ``(conn, qvec, k) -> row ids`` by vector similarity.
        fetch_candidates: ``(conn, ids) -> Candidates`` hydrating the fused ids.
        keyword_search: ``(conn, query, k) -> row ids``. Required when
            ``config.hybrid_enabled`` is True; ignored otherwise.
        reranker: ``(query, texts, top_n) -> [(idx, score), ...]``. Used only
            when ``config.rerank_enabled`` is True and this is provided. A
            failure here degrades to the fused order (logged), never raises.
        now: Reference time for the freshness boost (defaults to ``utcnow``;
            inject for deterministic tests).

    Returns:
        Up to ``config.top_k`` candidates ordered best-first.
    """
    if config.hybrid_enabled and keyword_search is None:
        raise ValueError("config.hybrid_enabled is True but no keyword_search was provided")

    rankings: list[list[object]] = [list(await vector_search(conn, qvec, config.candidate_k))]
    if config.hybrid_enabled and keyword_search is not None:
        rankings.append(list(await keyword_search(conn, query, config.candidate_k)))

    fused = rrf_fuse(rankings, config.rrf_k)
    if not fused:
        return []

    candidate_ids = sorted(fused, key=lambda i: fused[i], reverse=True)[: config.candidate_k]
    hydrated = {c.id: c for c in await fetch_candidates(conn, candidate_ids)}
    ref_now = now or datetime.now(UTC)

    candidates: list[Candidate] = []
    for cid in candidate_ids:
        cand = hydrated.get(cid)
        if cand is None:
            continue
        cand.score = fused[cid] + freshness_boost(
            cand.created_at, config.freshness_half_life_days, ref_now, rrf_k=config.rrf_k
        )
        candidates.append(cand)
    candidates.sort(key=lambda c: c.score, reverse=True)

    if config.rerank_enabled and reranker is not None and candidates:
        try:
            order = await reranker(query, [c.text for c in candidates], config.top_k)
        except Exception:
            # A reranker blip must not fail the query — degrade to the fused
            # (still hybrid-ranked) order and log so the dip is observable.
            # Scoped to the rerank call only: a bug in the reorder below must
            # NOT be masked as "reranker unavailable".
            logger.warning("event=rerank_unavailable action=degrade_to_fused")
        else:
            reranked: list[Candidate] = []
            for orig_idx, rscore in order:
                cand = candidates[orig_idx]
                cand.score = rscore
                reranked.append(cand)
            return reranked[: config.top_k]

    return candidates[: config.top_k]
