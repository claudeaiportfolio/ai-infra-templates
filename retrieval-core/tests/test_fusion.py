"""Tests for the pure fusion + freshness functions."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from retrieval_core import freshness_boost, rrf_fuse


class TestRrfFuse:
    def test_single_ranking_matches_reciprocal(self) -> None:
        scores = rrf_fuse([["a", "b", "c"]], rrf_k=60)
        assert scores["a"] == pytest.approx(1 / 60)
        assert scores["b"] == pytest.approx(1 / 61)
        assert scores["c"] == pytest.approx(1 / 62)

    def test_documents_in_both_rankings_sum(self) -> None:
        scores = rrf_fuse([["a", "b"], ["b", "a"]], rrf_k=60)
        # both at rank 0 and rank 1 in the two lists -> identical fused score
        assert scores["a"] == pytest.approx(1 / 60 + 1 / 61)
        assert scores["b"] == pytest.approx(1 / 60 + 1 / 61)
        assert scores["a"] == pytest.approx(scores["b"])

    def test_consensus_beats_single_top(self) -> None:
        # 'b' is mid-rank in both arms; 'a' is top in one only.
        scores = rrf_fuse([["a", "b", "c"], ["c", "b", "x"]], rrf_k=60)
        assert scores["b"] > scores["a"]

    def test_empty_rankings(self) -> None:
        assert rrf_fuse([], rrf_k=60) == {}
        assert rrf_fuse([[]], rrf_k=60) == {}

    def test_invalid_rrf_k_raises(self) -> None:
        with pytest.raises(ValueError, match="rrf_k"):
            rrf_fuse([["a"]], rrf_k=0)


class TestFreshnessBoost:
    def test_disabled_when_half_life_zero(self) -> None:
        now = datetime.now(UTC)
        assert freshness_boost(now, half_life_days=0.0, now=now, rrf_k=60) == 0.0

    def test_none_created_at_is_zero(self) -> None:
        now = datetime.now(UTC)
        assert freshness_boost(None, half_life_days=30.0, now=now, rrf_k=60) == 0.0

    def test_decays_with_age(self) -> None:
        now = datetime.now(UTC)
        fresh = freshness_boost(now, 30.0, now, rrf_k=60)
        old = freshness_boost(now - timedelta(days=30), 30.0, now, rrf_k=60)
        assert fresh > old
        # one half-life => exactly half the boost
        assert old == pytest.approx(fresh / 2)

    def test_capped_below_one_rrf_step(self) -> None:
        now = datetime.now(UTC)
        boost = freshness_boost(now, 30.0, now, rrf_k=60)
        # max boost (age 0) is 0.5/rrf_k, strictly less than a rank-0 RRF step (1/rrf_k)
        assert boost == pytest.approx(0.5 / 60)
        assert boost < 1 / 60
