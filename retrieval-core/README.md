# retrieval-core

A store-agnostic hybrid retrieval pipeline for RAG, extracted from a production
service and generalised. Hosted in `claudeaiportfolio/ai-infra-templates` and
consumed across portfolio projects.

## What it is

The package owns the reusable *orchestration* IP of a retrieval stack and
injects everything store- or model-specific:

- **Hybrid retrieval** — vector + keyword rankings fused with **Reciprocal Rank
  Fusion** (RRF). RRF fuses on *rank*, not score, so the two retrievers'
  incomparable scales (cosine similarity vs `ts_rank_cd`) need no normalisation
  or per-corpus weight tuning.
- **Freshness tiebreak** — an optional, capped additive boost that only breaks
  near-ties toward newer chunks (never reorders a clearly-better one).
- **Cross-encoder rerank** — an optional TEI-compatible `/rerank` step that
  **degrades gracefully**: a reranker blip falls back to the fused order rather
  than failing the query.
- **Budgeted context assembly** — three swappable packing policies
  (`top_k_by_fused`, `rerank_then_top_k`, `rerank_then_compress`) under a fixed
  token budget, with a deterministic model-free extractive compressor.

The datastore (`conn`), the encoded query vector (`qvec`), the reranker HTTP
call, and the tokenizer are all **injected** — as plain async callables and
config objects, not globals. There is no `settings` singleton and no
solution-identifying default anywhere.

## What it is not

- A datastore or ORM — the connection is passed through opaquely; you bring the
  driver. A ready-made pgvector + Postgres-FTS adapter is included for the
  common case (`retrieval_core.pg`), but it is optional.
- An embedding service — you encode the query and pass the vector in.
- An eval framework — measure policies with `agent-evals` (sibling package).

## Install

The package lives in a subdirectory of the `ai-infra-templates` monorepo.
Consumers pin by tag:

```bash
pip install "git+https://github.com/claudeaiportfolio/ai-infra-templates.git@retrieval-core-v0.1.0#subdirectory=retrieval-core"
```

Optional extras (both imported lazily — the core has zero runtime deps):

```bash
pip install "git+...#subdirectory=retrieval-core[rerank]"    # httpx, for the reranker client
pip install "git+...#subdirectory=retrieval-core[tiktoken]"  # default token counter
pip install "git+...#subdirectory=retrieval-core[all]"
```

For local development:

```bash
pip install -e .[dev]
ruff check . && mypy src && pytest
```

## Quickstart

```python
import asyncpg  # your driver of choice
from retrieval_core import RetrievalConfig, AssemblyConfig, retrieve, assemble, make_reranker
from retrieval_core.pg import (
    TableSchema, make_vector_search, make_keyword_search, make_fetch_candidates,
)

# 1. Describe your table once (no names are baked in — you supply them all).
schema = TableSchema(
    table="chunks",
    id_column="id",
    text_column="text",
    embedding_column="embedding",   # pgvector column
    tsvector_column="tsv",          # full-text column
    created_at_column="created_at",
    tenant_column="tenant_id",      # optional multi-tenancy
    metadata_columns=("source_doc", "heading_path", "page_start", "page_end"),
)

config = RetrievalConfig(candidate_k=40, top_k=8, hybrid_enabled=True, rerank_enabled=True)

async def search(conn, qvec: str, query: str, tenant: str):
    candidates = await retrieve(
        conn,
        qvec=qvec,                  # you encoded the query elsewhere
        query=query,
        config=config,
        vector_search=make_vector_search(schema, tenant=tenant),
        keyword_search=make_keyword_search(schema, tenant=tenant),
        fetch_candidates=make_fetch_candidates(schema, tenant=tenant),
        reranker=make_reranker("http://reranker.internal"),
    )
    result = assemble(
        candidates,
        config=AssemblyConfig(policy="rerank_then_top_k", token_budget=4000),
        query=query,
        citation_fn=lambda c: f"{c.metadata['source_doc']} :: {c.metadata['heading_path']}",
    )
    return result.context
```

### Bringing your own store

`retrieval_core.pg` is just one implementation of the injection seams. Any
async callables of the right shape work — see `retrieval_core.types`:

```python
VectorSearch    = (conn, qvec, k)          -> ids
KeywordSearch   = (conn, query_text, k)    -> ids
FetchCandidates = (conn, ids)              -> [Candidate, ...]
Reranker        = (query, texts, top_n)    -> [(original_index, score), ...]
```

Pass your own and the same fuse/rerank/assemble logic runs unchanged.

## Configuration

Two frozen dataclasses replace the original service's `settings` singleton —
everything is explicit and validated at construction:

`RetrievalConfig`: `candidate_k`, `top_k` (both required), `rrf_k` (default 60,
the canonical RRF constant), `hybrid_enabled`, `rerank_enabled`,
`freshness_half_life_days`.

`AssemblyConfig`: `policy`, `token_budget` (both required),
`compress_per_chunk_tokens`.

## Design notes

- **Why injection over a `settings` global.** The source service read every knob
  and the table/column names from a process-wide Pydantic `settings` object. As
  a shared library that is both untestable and coupling: this package takes all
  configuration as explicit arguments, so two callers with different corpora,
  budgets, or schemas coexist in one process and every test pins exact inputs.
- **Why the datastore is injected, not owned.** SQL identifiers are inherently
  schema-specific; baking a table named `chunks` into a reusable package is the
  exact drift this repo's conventions forbid. The pipeline owns the ordering
  logic; the thin SQL adapter is built from an explicit `TableSchema` (with
  identifier validation) or replaced wholesale for a non-Postgres store.
- **Determinism.** Fusion, freshness, and compression are pure and unit-tested;
  the compressor is model-free (query-term sentence overlap) so assembly adds no
  model call and no nondeterminism.

## License

MIT.
