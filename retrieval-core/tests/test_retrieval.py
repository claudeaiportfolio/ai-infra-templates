"""Tests for the retrieve() orchestration, with the datastore and reranker
injected as in-memory fakes (no live services)."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from retrieval_core import Candidate, RetrievalConfig, retrieve

NOW = datetime(2026, 6, 28, tzinfo=UTC)

# A tiny in-memory corpus keyed by id.
CORPUS = {
    1: "alpha beta gamma",
    2: "beta delta epsilon",
    3: "gamma zeta eta",
    4: "theta iota kappa",
}


def make_fakes(vector_ids: list[int], keyword_ids: list[int] | None = None):
    """Build injected vector/keyword/fetch callables over CORPUS."""

    async def vector_search(conn: object, qvec: object, k: int) -> list[int]:
        return vector_ids[:k]

    async def keyword_search(conn: object, query: str, k: int) -> list[int]:
        return (keyword_ids or [])[:k]

    async def fetch_candidates(conn: object, ids: list[int]) -> list[Candidate]:
        return [Candidate(id=i, text=CORPUS[i]) for i in ids if i in CORPUS]

    return vector_search, keyword_search, fetch_candidates


@pytest.fixture
def base_config() -> RetrievalConfig:
    return RetrievalConfig(candidate_k=10, top_k=3, hybrid_enabled=False)


class TestVectorOnly:
    async def test_returns_top_k_in_fused_order(self, base_config: RetrievalConfig) -> None:
        vector, _, fetch = make_fakes([3, 1, 2, 4])
        out = await retrieve(
            None,
            qvec="q",
            query="anything",
            config=base_config,
            vector_search=vector,
            fetch_candidates=fetch,
            now=NOW,
        )
        assert [c.id for c in out] == [3, 1, 2]
        assert out[0].score > out[1].score > out[2].score

    async def test_empty_when_nothing_retrieved(self, base_config: RetrievalConfig) -> None:
        vector, _, fetch = make_fakes([])
        out = await retrieve(
            None, qvec="q", query="x", config=base_config,
            vector_search=vector, fetch_candidates=fetch, now=NOW,
        )
        assert out == []

    async def test_missing_hydration_row_is_skipped(self, base_config: RetrievalConfig) -> None:
        vector, _, fetch = make_fakes([1, 999, 2])  # 999 not in CORPUS
        out = await retrieve(
            None, qvec="q", query="x", config=base_config,
            vector_search=vector, fetch_candidates=fetch, now=NOW,
        )
        assert [c.id for c in out] == [1, 2]


class TestHybrid:
    async def test_consensus_doc_ranks_first(self) -> None:
        cfg = RetrievalConfig(candidate_k=10, top_k=3, hybrid_enabled=True)
        # doc 2 appears in both arms -> highest fused score
        vector, keyword, fetch = make_fakes([1, 2, 3], keyword_ids=[2, 4])
        out = await retrieve(
            None, qvec="q", query="beta", config=cfg,
            vector_search=vector, keyword_search=keyword, fetch_candidates=fetch, now=NOW,
        )
        assert out[0].id == 2

    async def test_hybrid_requires_keyword_search(self) -> None:
        cfg = RetrievalConfig(candidate_k=10, top_k=3, hybrid_enabled=True)
        vector, _, fetch = make_fakes([1, 2])
        with pytest.raises(ValueError, match="keyword_search"):
            await retrieve(
                None, qvec="q", query="x", config=cfg,
                vector_search=vector, fetch_candidates=fetch, now=NOW,
            )


class TestRerank:
    async def test_reranker_reorders_and_sets_scores(self) -> None:
        cfg = RetrievalConfig(candidate_k=10, top_k=2, hybrid_enabled=False, rerank_enabled=True)
        vector, _, fetch = make_fakes([1, 2, 3])

        async def reranker(query: str, texts: list[str], top_n: int | None) -> list[tuple[int, float]]:
            # reverse the fused order; assign descending scores
            order = list(range(len(texts)))[::-1]
            return [(idx, float(rank)) for rank, idx in enumerate(order)]

        out = await retrieve(
            None, qvec="q", query="x", config=cfg,
            vector_search=vector, fetch_candidates=fetch, reranker=reranker, now=NOW,
        )
        # fused order was [1,2,3]; reversed -> [3,2,1]; top_n=2 -> [3,2]
        assert [c.id for c in out] == [3, 2]
        assert out[0].score == 0.0  # first reranked item, rank 0

    async def test_reranker_failure_degrades_to_fused(self) -> None:
        cfg = RetrievalConfig(candidate_k=10, top_k=3, hybrid_enabled=False, rerank_enabled=True)
        vector, _, fetch = make_fakes([3, 1, 2])

        async def boom(query: str, texts: list[str], top_n: int | None) -> list[tuple[int, float]]:
            raise RuntimeError("reranker pod unavailable")

        out = await retrieve(
            None, qvec="q", query="x", config=cfg,
            vector_search=vector, fetch_candidates=fetch, reranker=boom, now=NOW,
        )
        assert [c.id for c in out] == [3, 1, 2]  # fused order preserved

    async def test_rerank_skipped_when_no_reranker(self) -> None:
        cfg = RetrievalConfig(candidate_k=10, top_k=3, hybrid_enabled=False, rerank_enabled=True)
        vector, _, fetch = make_fakes([1, 2, 3])
        out = await retrieve(
            None, qvec="q", query="x", config=cfg,
            vector_search=vector, fetch_candidates=fetch, now=NOW,
        )
        assert [c.id for c in out] == [1, 2, 3]


class TestFreshness:
    async def test_freshness_breaks_ties_toward_newer(self) -> None:
        cfg = RetrievalConfig(
            candidate_k=10, top_k=2, hybrid_enabled=False, freshness_half_life_days=30.0
        )

        async def vector_search(conn: object, qvec: object, k: int) -> list[int]:
            return [1, 2]

        async def fetch_candidates(conn: object, ids: list[int]) -> list[Candidate]:
            # same fused rank position is impossible (distinct ranks), but the
            # freshness boost is large enough here to flip adjacent near-ties.
            return [
                Candidate(id=1, text="old", created_at=datetime(2020, 1, 1, tzinfo=UTC)),
                Candidate(id=2, text="new", created_at=NOW),
            ]

        out = await retrieve(
            None, qvec="q", query="x", config=cfg,
            vector_search=vector_search, fetch_candidates=fetch_candidates, now=NOW,
        )
        # id=1 has higher base RRF (rank 0), but the gap (1/60 - 1/61) is tiny and
        # the fresh boost on id=2 (0.5/60) overcomes it.
        assert out[0].id == 2


class TestConfigValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"candidate_k": 0, "top_k": 3}, "candidate_k"),
            ({"candidate_k": 3, "top_k": 0}, "top_k"),
            ({"candidate_k": 3, "top_k": 3, "rrf_k": 0}, "rrf_k"),
            ({"candidate_k": 3, "top_k": 3, "freshness_half_life_days": -1.0}, "freshness"),
        ],
    )
    def test_rejects_bad_values(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            RetrievalConfig(**kwargs)
