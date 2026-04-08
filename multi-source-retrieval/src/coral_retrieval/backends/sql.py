"""Generic SQL backend for structured retrieval.

Works with any DB-API 2.0 connection or SQLAlchemy engine. Useful for
OLTP stores, data warehouses, or Athena-style analytics backends.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol, runtime_checkable

from ..types import SearchHit

logger = logging.getLogger(__name__)


@runtime_checkable
class DBConnection(Protocol):
    """Minimal DB-API 2.0 connection."""

    def execute(self, sql: str, params: Any = ...) -> Any: ...
    def fetchall(self) -> list[tuple[Any, ...]]: ...


class SQLBackend:
    """Execute a parameterized SQL query and return rows as SearchHits.

    This is intentionally simple: you provide the query template and
    column mappings. The backend executes it and wraps results.

    Args:
        connection: Any DB-API 2.0 connection (DuckDB, psycopg2, etc.)
                    or a SQLAlchemy engine.
        query_template: SQL with a ``{query}`` placeholder for the search
                        term and ``{top_k}`` for the limit. Use ``?`` for
                        parameter binding. Example::

                            SELECT id, title, body, relevance
                            FROM articles
                            WHERE title ILIKE ?
                            ORDER BY relevance DESC
                            LIMIT ?

        id_column: Column index (0-based) for the hit id.
        text_column: Column index for the hit text.
        score_column: Column index for the score (or None for rank-based).
        source_name: Name reported in :attr:`SearchHit.source`.
    """

    def __init__(
        self,
        connection: Any,
        query_template: str,
        id_column: int = 0,
        text_column: int = 1,
        score_column: int | None = None,
        source_name: str = "sql",
    ) -> None:
        self._conn = connection
        self._template = query_template
        self._id_col = id_column
        self._text_col = text_column
        self._score_col = score_column
        self._source_name = source_name

    @property
    def name(self) -> str:
        return self._source_name

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        pattern = f"%{query}%"
        cursor = self._conn.execute(self._template, [pattern, top_k])
        rows = cursor.fetchall()

        hits: list[SearchHit] = []
        for rank, row in enumerate(rows):
            doc_id = str(row[self._id_col])
            text = str(row[self._text_col])

            if self._score_col is not None and self._score_col < len(row):
                score = float(row[self._score_col])
            else:
                score = 1.0 / (1.0 + rank)

            extra = {f"col_{i}": v for i, v in enumerate(row) if i not in (self._id_col, self._text_col)}
            hits.append(
                SearchHit(id=doc_id, text=text, score=score, source=self.name, metadata=extra)
            )
        return hits


class DuckDBSQLBackend:
    """Convenience wrapper for DuckDB with SQL retrieval.

    Accepts a path to a DuckDB database and a query template.
    Manages its own connection.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        query_template: str = "",
        id_column: int = 0,
        text_column: int = 1,
        score_column: int | None = None,
        source_name: str = "duckdb_sql",
    ) -> None:
        import duckdb

        self._conn = duckdb.connect(db_path, read_only=True)
        self._inner = SQLBackend(
            connection=self._conn,
            query_template=query_template,
            id_column=id_column,
            text_column=text_column,
            score_column=score_column,
            source_name=source_name,
        )

    @property
    def name(self) -> str:
        return self._inner.name

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        return self._inner.search(query, top_k=top_k)
