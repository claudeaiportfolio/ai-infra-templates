"""Hybrid retrieval core — fuse, rerank, assemble.

A store-agnostic retrieval pipeline extracted from a production RAG service and
generalised: vector + keyword retrieval fused with Reciprocal Rank Fusion, an
optional freshness tiebreak, an optional cross-encoder rerank step that degrades
gracefully, and budgeted context assembly with swappable packing policies.

The package owns the *orchestration* logic and injects everything store- or
model-specific. Configuration is explicit (frozen dataclasses); there is no
global ``settings`` singleton and no solution-identifying default.

Public API:

    from retrieval_core import (
        # data model
        Candidate,
        # config
        RetrievalConfig, AssemblyConfig, AssemblyPolicy,
        # pipeline
        retrieve, rrf_fuse, freshness_boost,
        # rerank
        rerank, make_reranker,
        # assembly
        assemble, AssemblyResult, default_token_counter, default_citation,
        # injection seam types
        VectorSearch, KeywordSearch, FetchCandidates, Reranker, TokenCounter,
    )

A ready-made pgvector + Postgres-FTS adapter for the injection seams lives in a
submodule (no runtime dependency, connection is duck-typed):

    from retrieval_core.pg import (
        TableSchema, make_vector_search, make_keyword_search, make_fetch_candidates,
    )

``rerank`` needs ``httpx`` and the default token counter needs ``tiktoken`` —
both are optional extras (``retrieval-core[rerank]``, ``retrieval-core[tiktoken]``,
or ``retrieval-core[all]``) imported lazily, so importing this package costs
nothing if you only use the pure pieces.
"""

from __future__ import annotations

from retrieval_core.assembly import (
    AssemblyResult,
    assemble,
    default_citation,
    default_token_counter,
)
from retrieval_core.config import AssemblyConfig, AssemblyPolicy, RetrievalConfig
from retrieval_core.fusion import freshness_boost, rrf_fuse
from retrieval_core.rerank import make_reranker, rerank
from retrieval_core.retrieval import retrieve
from retrieval_core.types import (
    Candidate,
    FetchCandidates,
    KeywordSearch,
    Reranker,
    TokenCounter,
    VectorSearch,
)

__version__ = "0.1.0"

__all__ = [
    # data model
    "Candidate",
    # config
    "RetrievalConfig",
    "AssemblyConfig",
    "AssemblyPolicy",
    # pipeline
    "retrieve",
    "rrf_fuse",
    "freshness_boost",
    # rerank
    "rerank",
    "make_reranker",
    # assembly
    "assemble",
    "AssemblyResult",
    "default_token_counter",
    "default_citation",
    # injection seam types
    "VectorSearch",
    "KeywordSearch",
    "FetchCandidates",
    "Reranker",
    "TokenCounter",
    # version
    "__version__",
]
