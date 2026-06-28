"""A ready-made pgvector + Postgres full-text-search adapter.

This is the lift-and-drop implementation of the injection seams in
``retrieval_core.types`` for the common stack: chunks in a single Postgres
table, a ``vector`` column indexed with pgvector (HNSW/IVFFlat), and a
``tsvector`` column for BM25-ish keyword search via ``ts_rank_cd``.

Everything store-specific is supplied through an explicit :class:`TableSchema`
— there are no baked-in table or column names. Identifiers are validated against
a strict pattern before being interpolated into SQL (they come from developer
config, never end-user input, but the validation closes the door regardless);
all *values* (query vector, tenant, limit) are bound as ``$n`` parameters, so
there is no value-injection surface.

The connection is duck-typed: any object exposing
``await conn.fetch(sql, *args) -> Sequence[Mapping]`` works (asyncpg's
``Connection`` is the reference shape). No driver is imported, so this module
has no runtime dependency.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from retrieval_core.types import Candidate, FetchCandidates, KeywordSearch, VectorSearch

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name: str) -> str:
    """Validate a SQL identifier (unqualified or schema-qualified) before
    interpolation. Raises on anything that isn't a plain identifier."""
    parts = name.split(".")
    if not parts or any(not _IDENTIFIER.match(p) for p in parts):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return name


@dataclass(frozen=True)
class TableSchema:
    """Maps the pipeline's generic needs onto a concrete Postgres table.

    All names are required except where a neutral, non-identifying default is
    universal. ``table`` and the column names have **no defaults** so nothing
    solution-specific is ever assumed.
    """

    table: str
    id_column: str
    text_column: str
    embedding_column: str
    tsvector_column: str
    created_at_column: str | None = None
    tenant_column: str | None = None
    metadata_columns: tuple[str, ...] = ()
    fts_language: str = "english"

    def __post_init__(self) -> None:
        for name in (self.table, self.id_column, self.text_column, self.embedding_column, self.tsvector_column):
            _ident(name)
        for opt in (self.created_at_column, self.tenant_column):
            if opt is not None:
                _ident(opt)
        for col in self.metadata_columns:
            _ident(col)
        if not _IDENTIFIER.match(self.fts_language):
            raise ValueError(f"invalid fts_language: {self.fts_language!r}")


def make_vector_search(schema: TableSchema, *, tenant: Any | None = None) -> VectorSearch:
    """Build a vector-similarity :data:`~retrieval_core.types.VectorSearch`.

    Orders by ``embedding <=> $vec`` (pgvector cosine distance). When ``tenant``
    is given (and the schema has a ``tenant_column``) results are scoped to it.
    """
    scoped = tenant is not None and schema.tenant_column is not None

    async def _search(conn: Any, qvec: Any, k: int) -> list[Any]:
        if scoped:
            sql = (
                f"SELECT {schema.id_column} FROM {schema.table} "  # noqa: S608 - identifiers validated
                f"WHERE {schema.tenant_column} = $2 "
                f"ORDER BY {schema.embedding_column} <=> $1::vector LIMIT $3"
            )
            rows = await conn.fetch(sql, qvec, tenant, k)
        else:
            sql = (
                f"SELECT {schema.id_column} FROM {schema.table} "  # noqa: S608 - identifiers validated
                f"ORDER BY {schema.embedding_column} <=> $1::vector LIMIT $2"
            )
            rows = await conn.fetch(sql, qvec, k)
        return [r[schema.id_column] for r in rows]

    return _search


def make_keyword_search(schema: TableSchema, *, tenant: Any | None = None) -> KeywordSearch:
    """Build a full-text :data:`~retrieval_core.types.KeywordSearch`.

    Uses ``websearch_to_tsquery`` (accepts user-style queries — quotes, OR —
    with no tsquery-injection surface) and ranks by ``ts_rank_cd`` (term density
    + proximity, the BM25-ish signal Postgres FTS gives natively).
    """
    scoped = tenant is not None and schema.tenant_column is not None
    lang = schema.fts_language

    async def _search(conn: Any, query_text: str, k: int) -> list[Any]:
        if scoped:
            sql = (
                f"SELECT {schema.id_column} FROM {schema.table} "  # noqa: S608 - identifiers validated
                f"WHERE {schema.tenant_column} = $2 "
                f"AND {schema.tsvector_column} @@ websearch_to_tsquery('{lang}', $1) "
                f"ORDER BY ts_rank_cd({schema.tsvector_column}, websearch_to_tsquery('{lang}', $1)) DESC "
                f"LIMIT $3"
            )
            rows = await conn.fetch(sql, query_text, tenant, k)
        else:
            sql = (
                f"SELECT {schema.id_column} FROM {schema.table} "  # noqa: S608 - identifiers validated
                f"WHERE {schema.tsvector_column} @@ websearch_to_tsquery('{lang}', $1) "
                f"ORDER BY ts_rank_cd({schema.tsvector_column}, websearch_to_tsquery('{lang}', $1)) DESC "
                f"LIMIT $2"
            )
            rows = await conn.fetch(sql, query_text, k)
        return [r[schema.id_column] for r in rows]

    return _search


def make_fetch_candidates(schema: TableSchema, *, tenant: Any | None = None) -> FetchCandidates:
    """Build a :data:`~retrieval_core.types.FetchCandidates` that hydrates ids
    into :class:`~retrieval_core.types.Candidate` objects, populating
    ``metadata`` from ``schema.metadata_columns``.
    """
    scoped = tenant is not None and schema.tenant_column is not None
    select_cols = [schema.id_column, schema.text_column]
    if schema.created_at_column is not None:
        select_cols.append(schema.created_at_column)
    select_cols.extend(schema.metadata_columns)
    # De-dup while preserving order (id/text could overlap metadata if misconfigured).
    seen: dict[str, None] = {}
    for c in select_cols:
        seen.setdefault(c, None)
    projection = ", ".join(seen)

    async def _fetch(conn: Any, ids: Sequence[Any]) -> list[Candidate]:
        id_list = list(ids)
        if not id_list:
            return []
        if scoped:
            sql = (
                f"SELECT {projection} FROM {schema.table} "  # noqa: S608 - identifiers validated
                f"WHERE {schema.id_column} = ANY($1) AND {schema.tenant_column} = $2"
            )
            rows = await conn.fetch(sql, id_list, tenant)
        else:
            sql = (
                f"SELECT {projection} FROM {schema.table} "  # noqa: S608 - identifiers validated
                f"WHERE {schema.id_column} = ANY($1)"
            )
            rows = await conn.fetch(sql, id_list)
        return [_row_to_candidate(r, schema) for r in rows]

    return _fetch


def _row_to_candidate(row: Mapping[str, Any], schema: TableSchema) -> Candidate:
    created: datetime | None = None
    if schema.created_at_column is not None:
        created = row[schema.created_at_column]
    metadata = {col: row[col] for col in schema.metadata_columns}
    return Candidate(
        id=row[schema.id_column],
        text=row[schema.text_column],
        created_at=created,
        metadata=metadata,
    )


# Re-exported for type clarity in user code; the dataclass above is the surface.
__all__ = [
    "TableSchema",
    "make_vector_search",
    "make_keyword_search",
    "make_fetch_candidates",
]
