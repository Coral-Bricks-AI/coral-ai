"""Local multi-source retrieval — DIY with open source backends.

This example wires up three retrieval sources:
  1. In-memory vector search (cosine similarity)
  2. DuckDB property graph (entity expansion)
  3. DuckDB SQL (structured filters)

Results are fused with Reciprocal Rank Fusion.

Run:
    pip install coral-retrieval[duckdb]
    python examples/local_orchestration.py
"""

import duckdb

from coral_retrieval import MultiSourceRetriever
from coral_retrieval.backends.in_memory_vector import InMemoryVectorBackend
from coral_retrieval.backends.graph import DuckDBGraphBackend
from coral_retrieval.backends.sql import SQLBackend


def fake_embed(text: str) -> list[float]:
    """Toy embedding: bag-of-chars normalized to unit length.

    Replace with a real model (e.g. sentence-transformers, CoralBricks
    GPU inference) in production.
    """
    vec = [0.0] * 128
    for ch in text.lower():
        vec[ord(ch) % 128] += 1.0
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


def build_vector_backend() -> InMemoryVectorBackend:
    backend = InMemoryVectorBackend(embed_fn=fake_embed, source_name="articles")

    docs = [
        "Acme Corp acquires WidgetCo for $500M in all-stock deal",
        "WidgetCo Q3 revenue grew 40% year-over-year driven by enterprise sales",
        "Fintech startup NovaPay raises Series B led by Sequoia Capital",
        "Acme Corp reports record earnings, beating analyst estimates by 15%",
        "Regulatory filing shows Acme Corp insider sold 100k shares last month",
        "NovaPay partners with major banks to expand payment infrastructure",
        "WidgetCo CEO resigns amid board restructuring",
        "Global semiconductor shortage impacts Acme Corp supply chain",
    ]
    for doc in docs:
        backend.add(doc)

    return backend


def build_graph_backend() -> DuckDBGraphBackend:
    conn = duckdb.connect(":memory:")

    conn.execute("""
        CREATE TABLE vertex (name VARCHAR PRIMARY KEY, text VARCHAR)
    """)
    conn.execute("""
        CREATE TABLE edge (src VARCHAR, dst VARCHAR, rel VARCHAR)
    """)

    vertices = [
        ("Acme Corp", "Acme Corp — public industrial conglomerate"),
        ("WidgetCo", "WidgetCo — enterprise SaaS provider"),
        ("NovaPay", "NovaPay — fintech payments startup"),
        ("Sequoia Capital", "Sequoia Capital — venture capital firm"),
        ("John Smith", "John Smith — CEO of WidgetCo"),
    ]
    for name, text in vertices:
        conn.execute("INSERT INTO vertex VALUES (?, ?)", [name, text])

    edges = [
        ("Acme Corp", "WidgetCo", "acquired"),
        ("Sequoia Capital", "NovaPay", "invested_in"),
        ("John Smith", "WidgetCo", "leads"),
        ("NovaPay", "Acme Corp", "partner"),
    ]
    for src, dst, rel in edges:
        conn.execute("INSERT INTO edge VALUES (?, ?, ?)", [src, dst, rel])

    conn.close()

    return DuckDBGraphBackend(
        db_path=":memory:",
        vertex_table="vertex",
        edge_tables=["edge"],
        max_hops=2,
        source_name="graph",
    )


def build_sql_backend() -> SQLBackend:
    conn = duckdb.connect(":memory:")

    conn.execute("""
        CREATE TABLE filings (
            id VARCHAR,
            title VARCHAR,
            body VARCHAR,
            filing_date DATE,
            form_type VARCHAR
        )
    """)
    filings = [
        ("f-001", "Acme Corp 8-K", "Material acquisition of WidgetCo announced", "2025-11-15", "8-K"),
        ("f-002", "Acme Corp 10-Q", "Quarterly report showing 12% revenue growth", "2025-10-01", "10-Q"),
        ("f-003", "NovaPay Form D", "Notice of exempt offering of securities", "2025-09-20", "D"),
        ("f-004", "Acme Corp Form 4", "Insider transaction: 100,000 shares sold", "2025-12-01", "4"),
    ]
    for f in filings:
        conn.execute("INSERT INTO filings VALUES (?, ?, ?, ?, ?)", list(f))

    query_template = """
        SELECT id, title || ': ' || body, filing_date, form_type
        FROM filings
        WHERE title ILIKE ? OR body ILIKE ?
        LIMIT ?
    """

    class _DuckDBSQLAdapter:
        """Wrap DuckDB connection to match SQLBackend's expected param style."""

        def __init__(self, conn: duckdb.DuckDBPyConnection, template: str):
            self._conn = conn
            self._template = template
            self._source_name = "filings"

        @property
        def name(self) -> str:
            return self._source_name

        def search(self, query: str, *, top_k: int = 10) -> list:
            from coral_retrieval.types import SearchHit

            pattern = f"%{query}%"
            rows = self._conn.execute(
                self._template, [pattern, pattern, top_k]
            ).fetchall()
            hits = []
            for rank, row in enumerate(rows):
                hits.append(
                    SearchHit(
                        id=row[0],
                        text=row[1],
                        score=1.0 / (1.0 + rank),
                        source=self._source_name,
                        metadata={"filing_date": str(row[2]), "form_type": row[3]},
                    )
                )
            return hits

    return _DuckDBSQLAdapter(conn, query_template)


def main() -> None:
    vector = build_vector_backend()
    sql = build_sql_backend()

    retriever = MultiSourceRetriever(fusion="rrf")
    retriever.add(vector)
    retriever.add(sql)

    query = "Acme Corp acquisition"
    print(f"Query: {query!r}\n")

    result = retriever.search(query, top_k=5)
    print(f"Sources queried: {result.sources_queried}")
    print(f"Fusion: {result.fusion_strategy}")
    print(f"Hits: {len(result.hits)}\n")

    for i, hit in enumerate(result.hits, 1):
        print(f"  {i}. [{hit.source}] score={hit.score:.4f}")
        print(f"     {hit.text[:100]}")
        if hit.metadata:
            print(f"     meta={hit.metadata}")
        print()

    print("--- Filtering to 'filings' source only ---\n")
    filtered = retriever.search(query, top_k=5, sources=["filings"])
    for i, hit in enumerate(filtered.hits, 1):
        print(f"  {i}. [{hit.source}] {hit.text[:100]}")


if __name__ == "__main__":
    main()
