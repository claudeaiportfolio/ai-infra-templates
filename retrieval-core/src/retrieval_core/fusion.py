"""Pure fusion + freshness functions — no I/O, no globals, fully deterministic.

**Why RRF over weighted-sum.** Vector and keyword retrievers score on
incomparable scales (cosine similarity vs ``ts_rank_cd``), so summing them needs
per-corpus normalisation and weight tuning that doesn't transfer between
corpora. RRF fuses on *rank*, not score — no normalisation, no tuning, a robust
default you can defend without a grid search.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from typing import Any


def rrf_fuse(rankings: Sequence[Sequence[Any]], rrf_k: int) -> dict[Any, float]:
    """Reciprocal Rank Fusion.

    ``score(d) = Σ 1/(rrf_k + rank_d)`` over the rankings that contain ``d``
    (rank 0-based). Pure and deterministic.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")
    scores: dict[Any, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def freshness_boost(
    created_at: datetime | None,
    half_life_days: float,
    now: datetime,
    *,
    rrf_k: int,
) -> float:
    """Tiny additive boost decaying by age, capped below one RRF rank-step so it
    only breaks near-ties (never reorders a clearly-better chunk). Returns 0
    when freshness is off or the row has no timestamp."""
    if half_life_days <= 0 or created_at is None:
        return 0.0
    age_days = (now - created_at).total_seconds() / 86400.0
    return (0.5 / rrf_k) * math.exp(-age_days * math.log(2) / half_life_days)
