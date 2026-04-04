"""DuckDB property-graph backend.

Models a lightweight property graph as DuckDB tables (vertices + edges)
and exposes neighbor-expansion as retrieval. Requires
``pip install coral-retrieval[duckdb]``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..types import SearchHit

logger = logging.getLogger(__name__)


class DuckDBGraphBackend:
    """Retrieve related entities via multi-hop traversal on a DuckDB graph.

    The graph is stored as one or more edge tables with ``src`` / ``dst``
    columns, plus a vertex table with at least ``name`` and ``text`` columns.

    Args:
        db_path: Path to the DuckDB database (or ``:memory:``).
        vertex_table: Table containing vertex data.
        edge_tables: Edge tables to traverse (each must have ``src``, ``dst``).
        text_column: Column in *vertex_table* that holds displayable text.
        name_column: Column used as the vertex identifier.
        max_hops: Maximum traversal depth from seed nodes.
        source_name: Name reported in :attr:`SearchHit.source`.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        vertex_table: str = "vertex",
        edge_tables: list[str] | None = None,
        text_column: str = "text",
        name_column: str = "name",
        max_hops: int = 2,
        source_name: str = "graph",
    ) -> None:
        import duckdb

        self._conn = duckdb.connect(db_path, read_only=True)
        self._vertex_table = vertex_table
        self._edge_tables = edge_tables or ["edge"]
        self._text_col = text_column
        self._name_col = name_column
        self._max_hops = max_hops
        self._source_name = source_name

    @property
    def name(self) -> str:
        return self._source_name

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        """Seed with vertices matching *query*, then expand via edges.

        Seeding uses a case-insensitive ``LIKE`` filter. For production
        use, replace this with an embedding lookup or full-text index.
        """
        seeds = self._seed_vertices(query)
        if not seeds:
            return []
        neighbors = self._expand(seeds)
        return self._to_hits(neighbors, top_k)

    def _seed_vertices(self, query: str) -> list[str]:
        sql = f"""
            SELECT {self._name_col}
            FROM {self._vertex_table}
            WHERE {self._text_col} ILIKE ?
            LIMIT 50
        """
        pattern = f"%{query}%"
        rows = self._conn.execute(sql, [pattern]).fetchall()
        return [r[0] for r in rows]

    def _expand(self, seeds: list[str]) -> dict[str, dict[str, Any]]:
        """BFS expansion over edge tables up to ``max_hops``."""
        visited: dict[str, dict[str, Any]] = {}
        frontier = set(seeds)

        for hop in range(self._max_hops):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for edge_table in self._edge_tables:
                placeholders = ", ".join(["?"] * len(frontier))
                sql = f"""
                    SELECT dst FROM {edge_table}
                    WHERE src IN ({placeholders})
                    UNION
                    SELECT src FROM {edge_table}
                    WHERE dst IN ({placeholders})
                """
                params = list(frontier) + list(frontier)
                rows = self._conn.execute(sql, params).fetchall()
                for (node,) in rows:
                    if node not in visited and node not in frontier:
                        next_frontier.add(node)

            for node in frontier:
                if node not in visited:
                    visited[node] = {"hop": hop}
            frontier = next_frontier

        for node in frontier:
            if node not in visited:
                visited[node] = {"hop": self._max_hops}

        return visited

    def _to_hits(
        self, neighbors: dict[str, dict[str, Any]], top_k: int
    ) -> list[SearchHit]:
        if not neighbors:
            return []
        placeholders = ", ".join(["?"] * len(neighbors))
        sql = f"""
            SELECT {self._name_col}, {self._text_col}
            FROM {self._vertex_table}
            WHERE {self._name_col} IN ({placeholders})
        """
        rows = self._conn.execute(sql, list(neighbors.keys())).fetchall()

        hits: list[SearchHit] = []
        for name, text in rows:
            hop = neighbors.get(name, {}).get("hop", self._max_hops)
            score = 1.0 / (1.0 + hop)
            hits.append(
                SearchHit(
                    id=name,
                    text=text or name,
                    score=score,
                    source=self.name,
                    metadata={"hop": hop},
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
