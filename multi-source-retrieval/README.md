# coral-retrieval

Multi-source retrieval orchestration — vector search, graph traversal, and SQL behind one API.

Most RAG systems retrieve from a single vector store. Real-world knowledge is spread across dense embeddings, entity graphs, and structured databases. `coral-retrieval` provides the orchestration layer: fan out queries to heterogeneous backends, fuse ranked results, and return a single answer set.

```
                        ┌──────────────────┐
                        │   Your App /     │
                        │   Agent / Chain   │
                        └────────┬─────────┘
                                 │
                   ┌─────────────▼──────────────┐
                   │   MultiSourceRetriever      │
                   │   (orchestrate + fuse)       │
                   └──┬──────────┬───────────┬───┘
                      │          │           │
             ┌────────▼──┐ ┌────▼─────┐ ┌───▼──────┐
             │  Vector    │ │  Graph   │ │   SQL    │
             │ (kNN+BM25) │ │ (DuckDB) │ │ (any DB) │
             └────────────┘ └──────────┘ └──────────┘
```

## Install

```bash
pip install coral-retrieval            # core only, no backend deps
pip install coral-retrieval[opensearch] # + OpenSearch kNN / BM25
pip install coral-retrieval[duckdb]     # + DuckDB graph and SQL
pip install coral-retrieval[all]        # everything
```

## Quick Start

```python
from coral_retrieval import MultiSourceRetriever
from coral_retrieval.backends.in_memory_vector import InMemoryVectorBackend

# 1. Create a backend and add some documents
vector = InMemoryVectorBackend(embed_fn=your_embed_function)
vector.add("Acme Corp acquires WidgetCo for $500M in all-stock deal")
vector.add("NovaPay raises Series B led by Sequoia Capital")
vector.add("WidgetCo Q3 revenue grew 40% year-over-year")

# 2. Build a retriever — add as many backends as you need
retriever = MultiSourceRetriever(fusion="rrf")
retriever.add(vector)

# 3. Search
result = retriever.search("recent acquisitions", top_k=5)
for hit in result.hits:
    print(f"[{hit.source}] {hit.score:.4f}  {hit.text}")
```

### Multi-backend example

```python
from coral_retrieval import MultiSourceRetriever
from coral_retrieval.backends.opensearch_vector import OpenSearchVectorBackend
from coral_retrieval.backends.opensearch_keyword import OpenSearchKeywordBackend
from coral_retrieval.backends.graph import DuckDBGraphBackend

retriever = MultiSourceRetriever(fusion="rrf")
retriever.add(OpenSearchVectorBackend(client=os_client, index="docs", embed_fn=embed))
retriever.add(OpenSearchKeywordBackend(client=os_client, index="docs"))
retriever.add(DuckDBGraphBackend(db_path="knowledge.duckdb"))

result = retriever.search("Apple acquisitions in 2025", top_k=10)
```

Each backend is queried concurrently. Results are fused with [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — documents appearing across multiple backends and at higher ranks score higher.

## Backends

Every backend implements the `RetrievalBackend` protocol — a single `search(query, top_k)` method. Plug in any data source:

| Backend | Module | Description |
|---------|--------|-------------|
| **OpenSearch Vector** | `backends.opensearch_vector` | kNN semantic search via OpenSearch |
| **OpenSearch Keyword** | `backends.opensearch_keyword` | BM25 lexical search via OpenSearch |
| **In-Memory Vector** | `backends.in_memory_vector` | Brute-force cosine similarity for demos and testing |
| **DuckDB Graph** | `backends.graph` | Multi-hop entity expansion on a property graph |
| **SQL** | `backends.sql` | Any DB-API 2.0 connection (Postgres, DuckDB, SQLite, etc.) |
| **DuckDB SQL** | `backends.sql` | Convenience wrapper with managed DuckDB connection |
| **Custom** | Implement `RetrievalBackend` | Any source — Elasticsearch, Redis, external APIs, etc. |

### Writing a custom backend

```python
from coral_retrieval.types import SearchHit

class MyBackend:
    @property
    def name(self) -> str:
        return "my-api"

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        response = my_api.search(query, limit=top_k)
        return [
            SearchHit(id=r["id"], text=r["text"], score=r["score"], source=self.name)
            for r in response
        ]

retriever.add(MyBackend())
```

## Fusion Strategies

| Strategy | Function | When to use |
|----------|----------|-------------|
| **RRF** | `fusion.rrf.reciprocal_rank_fusion` | Default. Robust, no tuning needed. |
| **WSF** | `fusion.wsf.weighted_score_fusion` | When you want to weight backends differently. |
| **Custom** | Any callable | `(list[list[SearchHit]]) -> list[SearchHit]` |

```python
retriever = MultiSourceRetriever(fusion="rrf")   # default
retriever = MultiSourceRetriever(fusion="wsf")   # weighted score fusion

def my_fusion(ranked_lists):
    ...
retriever = MultiSourceRetriever(fusion=my_fusion)
```

## Orchestrator Options

```python
retriever = MultiSourceRetriever(
    fusion="rrf",       # fusion strategy
    parallel=True,      # query backends concurrently (default)
    timeout=30.0,       # per-backend timeout in seconds
)

# filter to specific backends at query time
result = retriever.search("query", top_k=10, sources=["vector", "graph"])
```

## Examples

See [`examples/local_orchestration.py`](examples/local_orchestration.py) for a runnable demo wiring up vector search + SQL with RRF fusion.

## Development

```bash
pip install -e ".[all,dev]"
pytest tests/ -v
```

## License

Apache 2.0 — see [LICENSE](../LICENSE).
