"""Tests for the pgvector + Postgres-FTS adapter. The connection is a fake that
records SQL/params and replays rows — no database."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from retrieval_core.pg import (
    TableSchema,
    make_fetch_candidates,
    make_keyword_search,
    make_vector_search,
)


class FakeConn:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self._rows = rows or []
        self.calls: list[tuple[str, tuple]] = []

    async def fetch(self, sql: str, *args: object) -> list[dict]:
        self.calls.append((sql, args))
        return self._rows


def schema(**overrides: object) -> TableSchema:
    base: dict = {
        "table": "documents",
        "id_column": "id",
        "text_column": "body",
        "embedding_column": "embedding",
        "tsvector_column": "tsv",
        "created_at_column": "created_at",
        "tenant_column": "tenant_id",
        "metadata_columns": ("source", "page"),
    }
    base.update(overrides)
    return TableSchema(**base)  # type: ignore[arg-type]


class TestIdentifierValidation:
    def test_rejects_injection_in_table(self) -> None:
        with pytest.raises(ValueError, match="identifier"):
            TableSchema(
                table="documents; DROP TABLE x",
                id_column="id", text_column="body",
                embedding_column="embedding", tsvector_column="tsv",
            )

    def test_rejects_bad_column(self) -> None:
        with pytest.raises(ValueError, match="identifier"):
            TableSchema(
                table="t", id_column="id", text_column="body",
                embedding_column="embedding", tsvector_column="tsv",
                metadata_columns=("ok", "no-dashes"),
            )

    def test_rejects_bad_language(self) -> None:
        with pytest.raises(ValueError, match="fts_language"):
            TableSchema(
                table="t", id_column="id", text_column="body",
                embedding_column="embedding", tsvector_column="tsv",
                fts_language="english'); DROP",
            )

    def test_accepts_schema_qualified_table(self) -> None:
        s = TableSchema(
            table="rag.documents", id_column="id", text_column="body",
            embedding_column="embedding", tsvector_column="tsv",
        )
        assert s.table == "rag.documents"


class TestVectorSearch:
    async def test_no_tenant_sql_and_params(self) -> None:
        conn = FakeConn([{"id": 7}, {"id": 9}])
        search = make_vector_search(schema(tenant_column=None))
        ids = await search(conn, "qvec", 5)
        assert ids == [7, 9]
        sql, args = conn.calls[0]
        assert "<=> $1::vector" in sql
        assert "WHERE" not in sql
        assert args == ("qvec", 5)

    async def test_tenant_scoped_sql_and_params(self) -> None:
        conn = FakeConn([{"id": 1}])
        search = make_vector_search(schema(), tenant="acme")
        await search(conn, "qvec", 3)
        sql, args = conn.calls[0]
        assert "WHERE tenant_id = $2" in sql
        assert args == ("qvec", "acme", 3)


class TestKeywordSearch:
    async def test_uses_websearch_tsquery_and_rank(self) -> None:
        conn = FakeConn([{"id": 2}])
        search = make_keyword_search(schema(tenant_column=None))
        await search(conn, "hello world", 4)
        sql, args = conn.calls[0]
        assert "websearch_to_tsquery('english', $1)" in sql
        assert "ts_rank_cd(tsv, websearch_to_tsquery('english', $1)) DESC" in sql
        assert args == ("hello world", 4)

    async def test_tenant_scoped(self) -> None:
        conn = FakeConn([])
        search = make_keyword_search(schema(), tenant="acme")
        await search(conn, "q", 4)
        sql, args = conn.calls[0]
        assert "tenant_id = $2" in sql
        assert args == ("q", "acme", 4)


class TestFetchCandidates:
    async def test_hydrates_with_metadata(self) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        conn = FakeConn(
            [
                {"id": 1, "body": "text one", "created_at": ts, "source": "a.pdf", "page": 3},
                {"id": 2, "body": "text two", "created_at": ts, "source": "b.pdf", "page": 9},
            ]
        )
        fetch = make_fetch_candidates(schema(tenant_column=None))
        out = await fetch(conn, [1, 2])
        assert [c.id for c in out] == [1, 2]
        assert out[0].text == "text one"
        assert out[0].created_at == ts
        assert out[0].metadata == {"source": "a.pdf", "page": 3}

    async def test_empty_ids_skips_query(self) -> None:
        conn = FakeConn([])
        fetch = make_fetch_candidates(schema())
        out = await fetch(conn, [])
        assert out == []
        assert conn.calls == []  # no SQL issued for an empty id list

    async def test_projection_includes_only_configured_columns(self) -> None:
        conn = FakeConn([{"id": 1, "body": "t"}])
        fetch = make_fetch_candidates(
            schema(tenant_column=None, created_at_column=None, metadata_columns=())
        )
        await fetch(conn, [1])
        sql, _ = conn.calls[0]
        assert "SELECT id, body FROM documents" in sql

    async def test_tenant_scoped_fetch(self) -> None:
        conn = FakeConn([])
        fetch = make_fetch_candidates(schema(metadata_columns=()), tenant="acme")
        await fetch(conn, [1, 2])
        sql, args = conn.calls[0]
        assert "= ANY($1) AND tenant_id = $2" in sql
        assert args == ([1, 2], "acme")
