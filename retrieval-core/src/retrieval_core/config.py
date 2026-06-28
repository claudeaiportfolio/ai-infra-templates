"""Explicit, frozen configuration objects.

These replace the module-level ``settings`` singleton the code was extracted
from: every knob is passed in, nothing is read from global state, and there are
no solution-identifying defaults. ``candidate_k`` and ``top_k`` are required
(no sensible universal default); ``rrf_k = 60`` is the canonical RRF constant
from the original paper, not a corpus-specific tuning, so it keeps a default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Context-assembly policy (see ``retrieval_core.assembly``).
AssemblyPolicy = Literal["top_k_by_fused", "rerank_then_top_k", "rerank_then_compress"]


@dataclass(frozen=True)
class RetrievalConfig:
    """Knobs for the fuse + freshness + rerank pipeline.

    candidate_k:
        Candidates pulled from *each* retriever before fusion/rerank.
    top_k:
        Final number of candidates returned.
    rrf_k:
        Reciprocal Rank Fusion constant; ``score = Σ 1/(rrf_k + rank)``. Larger
        values flatten the contribution of any single retriever's top rank. 60
        is the canonical default.
    hybrid_enabled:
        When True, fuse vector + keyword rankings (requires a ``keyword_search``
        callable). When False, run vector-only.
    rerank_enabled:
        When True and a reranker is supplied, reorder the fused candidates with
        the cross-encoder; on reranker failure the pipeline degrades to the
        fused order rather than erroring.
    freshness_half_life_days:
        When > 0, newer candidates get a small additive RRF boost decaying over
        this half-life (in days). 0 disables it (pure RRF/rerank ordering). The
        boost is capped below one RRF rank-step, so it only breaks near-ties.
    """

    candidate_k: int
    top_k: int
    rrf_k: int = 60
    hybrid_enabled: bool = True
    rerank_enabled: bool = False
    freshness_half_life_days: float = 0.0

    def __post_init__(self) -> None:
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be > 0")
        if self.freshness_half_life_days < 0:
            raise ValueError("freshness_half_life_days must be >= 0")


@dataclass(frozen=True)
class AssemblyConfig:
    """Knobs for budgeted context assembly.

    policy:
        One of ``top_k_by_fused`` | ``rerank_then_top_k`` |
        ``rerank_then_compress``. The first two pack whole chunks greedily in
        the order ``retrieve`` returned; the third extractively compresses each
        chunk so more distinct evidence fits.
    token_budget:
        The knapsack capacity for the assembled context, in tokens (counted by
        the ``token_counter`` passed to ``assemble``).
    compress_per_chunk_tokens:
        Per-chunk cap applied only by the ``rerank_then_compress`` policy.
    """

    policy: AssemblyPolicy
    token_budget: int
    compress_per_chunk_tokens: int = 200

    def __post_init__(self) -> None:
        if self.token_budget <= 0:
            raise ValueError("token_budget must be > 0")
        if self.compress_per_chunk_tokens <= 0:
            raise ValueError("compress_per_chunk_tokens must be > 0")
