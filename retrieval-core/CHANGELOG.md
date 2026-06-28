# Changelog

All notable changes to `retrieval-core` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this package adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Releases are cut
as git tags of the form `retrieval-core-vX.Y.Z` on the `ai-infra-templates`
monorepo.

## [0.1.0] - 2026-06-28

Initial extraction. The hybrid retrieval pipeline was lifted out of a production
RAG service (`rag-retrieval-service`'s `common/retrieval.py`, `common/rerank.py`,
`common/assembly.py`) and generalised into a standalone, store-agnostic package.
No existing consumer depends on it yet, so there is no backward-compatibility
surface to preserve.

### Added

- **Fusion** (`rrf_fuse`, `freshness_boost`) — pure, deterministic Reciprocal
  Rank Fusion and the capped freshness tiebreak.
- **Retrieval** (`retrieve`) — the hybrid orchestration: vector + optional
  keyword retrieval, RRF fusion, freshness boost, and an optional cross-encoder
  rerank step that degrades to the fused order on reranker failure.
- **Rerank** (`rerank`, `make_reranker`) — a TEI-compatible `/rerank` client
  with transient-fault retry and a pass-through mode; `httpx` is an optional,
  lazily-imported extra.
- **Assembly** (`assemble`, `AssemblyResult`, `default_token_counter`,
  `default_citation`) — budgeted context packing with three swappable policies
  and a model-free extractive compressor; the tokenizer and citation formatter
  are injected (`tiktoken` is an optional, lazily-imported extra).
- **Config** (`RetrievalConfig`, `AssemblyConfig`) — frozen, validated
  dataclasses that replace the source service's module-level `settings`
  singleton. Configuration is now passed explicitly; there are no
  solution-identifying defaults.
- **Injection seams** (`Candidate`, `VectorSearch`, `KeywordSearch`,
  `FetchCandidates`, `Reranker`, `TokenCounter`) — the typed contracts the
  pipeline is built around. The datastore connection and query vector remain
  injected parameters (no DB layer is invented).
- **pgvector adapter** (`retrieval_core.pg`: `TableSchema`,
  `make_vector_search`, `make_keyword_search`, `make_fetch_candidates`) — a
  ready-made pgvector + Postgres-FTS implementation of the seams, built from an
  explicit `TableSchema` with SQL-identifier validation and optional
  multi-tenancy. No driver is imported; the connection is duck-typed.

### Changed (vs. the in-service originals)

- Removed the module-level `from common.config import settings` coupling
  throughout; every knob is now a function/dataclass argument.
- Replaced the hard-coded `chunks` table and column names with the injected
  `TableSchema` (no table/column/model name is baked in as a default).
- Replaced the domain-specific `Candidate` fields (`source_doc`, `heading_path`,
  page numbers, ...) with a generic `metadata` mapping; citation formatting is
  injected via `citation_fn`.
- Made the OpenTelemetry tracer and the `tenacity`/`httpx` retry stack internal
  and dependency-light (lazy `httpx`, a small built-in retry), so the core has
  zero runtime dependencies.
