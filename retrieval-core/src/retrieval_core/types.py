"""Core data model and the injection seams the pipeline is built around.

The package owns the *orchestration* IP — Reciprocal Rank Fusion, the freshness
tiebreak, the rerank-with-graceful-degradation step, and budgeted context
assembly. It deliberately owns no datastore: the data-access pieces (vector
search, keyword search, candidate hydration, reranking) are injected as plain
async callables, so the same pipeline runs against any store that can satisfy
the seams. ``retrieval_core.pg`` ships a ready-made pgvector + Postgres-FTS
adapter for the common case; consumers with a different store provide their own
callables of the same shape.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# A row identifier. Usually an int (a primary key), but kept opaque so callers
# are free to key candidates on UUIDs, strings, or composite tuples.
RowId = Any


@dataclass
class Candidate:
    """A retrieved chunk flowing through fuse -> (rerank) -> assemble.

    ``score`` is mutated in place as the pipeline progresses (fused RRF score,
    then the reranker's score if reranking ran). ``metadata`` carries arbitrary
    per-row fields (source document, heading path, page numbers, ...) without
    the core model needing to know any domain-specific column names — assembly's
    citation formatting reads them through an injected ``citation_fn``.
    """

    id: RowId
    text: str
    created_at: datetime | None = None
    score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


# --- Injection seams ---------------------------------------------------------
# ``conn`` and ``qvec`` are passed through opaquely: the pipeline never inspects
# the connection object or the encoded query vector, it only routes them to the
# injected callables. That keeps the embedding model and DB driver entirely the
# caller's concern.

#: ``(conn, qvec, k) -> top-k row ids by vector similarity``.
VectorSearch = Callable[[Any, Any, int], Awaitable[Sequence[RowId]]]

#: ``(conn, query_text, k) -> top-k row ids by keyword/full-text relevance``.
KeywordSearch = Callable[[Any, str, int], Awaitable[Sequence[RowId]]]

#: ``(conn, ids) -> candidates`` hydrating the given ids (order-independent).
FetchCandidates = Callable[[Any, Sequence[RowId]], Awaitable[Sequence[Candidate]]]

#: ``(query, texts, top_n) -> [(original_index, score), ...]`` sorted desc.
Reranker = Callable[[str, "list[str]", "int | None"], Awaitable[Sequence[tuple[int, float]]]]

#: ``(text) -> token count`` for the model whose budget assembly is packing.
TokenCounter = Callable[[str], int]
