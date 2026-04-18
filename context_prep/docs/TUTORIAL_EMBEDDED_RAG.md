# Tutorial: End-to-end embedded RAG with DuckDB

> Chunks → vectors → entity graph → **one DuckDB session** that does
> vector top-K *and* variable-length graph traversal. No servers,
> no API keys, no Cypher.

This tutorial closes the loop on `coralbricks-context-prep`. The
library stops at parquet files on purpose — it's the *transforms*
layer, not a database. This walk-through shows what to do *with*
those parquet files in the smallest possible runtime: a single local
DuckDB connection, with two extensions (`vss` for HNSW, `duckpgq` for
SQL/PGQ graph queries).

The full runnable script is
[`examples/embedded_rag_duckdb.py`](../examples/embedded_rag_duckdb.py).

---

## What you'll build

```
                      coralbricks-context-prep
                ┌─────────────────────────────────────┐
records ─┐      │  chunk(...)         enrich(...)     │
         ├──────►                ┌────►                │
         └──────►  embed(...)    │     hydrate(...)   │
                │       │        │            │       │
                └───────┼────────┼────────────┼───────┘
                        ▼        │            ▼
                  vectors.parquet│   nodes.parquet + edges.parquet
                        └────────┴────────┬────────┘
                                          ▼
              ┌────────────────────────────────────────────────┐
              │  one DuckDB session                            │
              │   • vss extension       (HNSW vector index)    │
              │   • duckpgq extension   (SQL/PGQ graph queries)│
              │                                                │
              │   1) vector top-K                              │
              │   2) variable-length graph expansion           │
              │   3) hybrid: hits → seed entities → expansion  │
              └────────────────────────────────────────────────┘
```

## Install

```bash
pip install 'coralbricks-context-prep[graph,chunkers]' \
            duckdb sentence-transformers
```

`vss` is built into DuckDB (auto-installs from SQL). `duckpgq` is a
DuckDB **community extension** — also auto-installable from SQL on the
first run (`INSTALL duckpgq FROM community`).

> **Note:** `duckpgq` is currently flagged as "research / under active
> development" in the DuckDB community extensions registry. Path-finding
> works reliably; some built-in graph algorithms (PageRank etc.) may
> still have issues. The example script treats `duckpgq` as a hard
> requirement and fails fast with a clear error if it can't load — no
> silent fallback engine, no extra dependencies dragged in.

## Step 1 — Prep

Use the verbs you already know. Two parallel branches: chunks for
vector search, and entities for the graph.

```python
from coralbricks.context_prep import chunk, embed, enrich, hydrate

NEWS = [
    {"id": "n1", "text": "$AAPL and $MSFT both rallied on the Fed announcement; "
                          "$NVDA followed late in the session."},
    {"id": "n2", "text": "$AAPL beat earnings; $GOOGL also up after strong cloud results."},
    # ... 6 more docs
]

chunks   = chunk(NEWS, strategy="sliding_token", target_tokens=48, overlap=8)
vectors  = embed(chunks, embedder=my_st_embedder, output_dir="./out")
enriched = enrich(NEWS, extractors=["tickers"])
g        = hydrate(enriched, graph="news", output_dir="./out")
```

That writes:

```
out/
├── vectors.parquet     # doc_id, chunk_index, text, vector (FLOAT[384])
├── nodes.parquet       # id, label, value
└── edges.parquet       # src, dst, relation, weight
```

The library's job ends here. Everything below is downstream wiring.

> **Why no `mentions.parquet`?** `hydrate()` already writes
> `Document:<doc_id> --[mentions]--> <Label>:<value>` rows into
> `edges.parquet`. The vector→graph join is just `WHERE relation =
> 'mentions'` — see Step 4. No additional schema needed.

## Step 2 — Open DuckDB and load the parquet

```python
import duckdb

con = duckdb.connect()  # in-memory; pass a path to persist

dim = vectors.metadata["dimension"]   # e.g. 384 for bge-small
con.execute(f"""
    CREATE TABLE vectors AS
    SELECT
        doc_id, chunk_index, text,
        CAST(vector AS FLOAT[{dim}]) AS vector
    FROM read_parquet('out/vectors.parquet')
""")
con.execute("CREATE TABLE nodes AS SELECT * FROM read_parquet('out/nodes.parquet')")
con.execute("CREATE TABLE edges AS SELECT * FROM read_parquet('out/edges.parquet')")
```

The `CAST(vector AS FLOAT[{dim}])` is the only spell you need: it
turns the parquet fixed-size list into DuckDB's native `ARRAY` type,
which `vss` indexes directly.

## Step 3 — Two extensions, one session

```python
con.execute("INSTALL vss; LOAD vss;")
con.execute("INSTALL duckpgq FROM community; LOAD duckpgq;")

# HNSW vector index (cosine distance — bge-small embeddings are normalized)
con.execute("""
    CREATE INDEX vec_hnsw ON vectors USING HNSW (vector)
    WITH (metric = 'cosine')
""")

# SQL/PGQ property graph defined over the existing tables
con.execute("""
    CREATE PROPERTY GRAPH news_graph
      VERTEX TABLES (nodes PROPERTIES (id, label, value))
      EDGE TABLES (
        edges
          SOURCE KEY (src) REFERENCES nodes (id)
          DESTINATION KEY (dst) REFERENCES nodes (id)
          LABEL related
      )
""")
```

Now the same DuckDB connection has a vector index *and* a property
graph layered over your parquet — no data movement, no duplicate
storage.

## Step 4 — Three queries

### 4a. Vector top-K

```python
qvec, _ = my_st_embedder.embed_texts([
    "How did AAPL react to the Fed and which AI names moved with it?"
])

con.execute(f"""
    SELECT doc_id, text,
           array_cosine_distance(vector, ?::FLOAT[{dim}]) AS d
    FROM vectors ORDER BY d LIMIT 4
""", [qvec[0]]).fetchall()
```

Result on the sample corpus:

```
d=0.3331  doc=n1  '$AAPL and $MSFT both rallied on the Fed announcement; $NVDA foll…'
d=0.3419  doc=n5  '$GOOGL and $META lead the megacap rally; $AAPL lagged.'
d=0.3447  doc=n2  '$AAPL beat earnings; $GOOGL also up after strong cloud results.'
d=0.3767  doc=n3  '$NVDA hit a new high; $MSFT followed on AI demand.'
```

### 4b. Variable-length graph traversal — without Cypher

```sql
FROM GRAPH_TABLE (news_graph
  MATCH p = ANY SHORTEST
    (a:nodes)-[r:related]->{1,3}(b:nodes)
  WHERE a.id = 'Tickers:AAPL'
  COLUMNS (a.id AS seed, b.id AS reached, path_length(p) AS hops)
)
ORDER BY hops, reached;
```

Result:

```
Tickers:AAPL ─1→ Tickers:GOOGL
Tickers:AAPL ─1→ Tickers:MSFT
Tickers:AAPL ─1→ Tickers:NVDA
Tickers:AAPL ─2→ Tickers:META
```

The `{1,3}` quantifier is the variable-length path operator from
SQL/PGQ (SQL:2023). It's the standard syntax for
`(a)-[*1..3]->(b)` in Cypher, but it's regular SQL — no separate
query language to learn.

> **Node id convention:** node ids are `<Label>:<value>`. The label
> comes from the entity bucket in your enricher; for `extractors=
> ["tickers"]`, the bucket is `tickers` and the
> `CooccurrenceExtractor` titles it to `Tickers`. See
> `coralbricks/context_prep/graph/triples.py`.

### 4c. The hybrid query — vector + graph in one flow

This is the punchline. Vector search finds the documents, the
existing `mentions` edges turn those into seed entities, then the
graph traversal expands them.

```python
# Step 1: vector hits → list of doc ids
hit_doc_ids = [row[0] for row in con.execute(f"""
    SELECT doc_id
    FROM vectors
    ORDER BY array_cosine_distance(vector, ?::FLOAT[{dim}])
    LIMIT 4
""", [qvec[0]]).fetchall()]

# Step 2: doc ids → seed entities, via the mentions edges hydrate() wrote
placeholders = ", ".join(["?"] * len(hit_doc_ids))
seed_ids = [r[0] for r in con.execute(f"""
    SELECT DISTINCT e.dst
    FROM edges e
    WHERE e.relation = 'mentions'
      AND REPLACE(e.src, 'Document:', '') IN ({placeholders})
""", hit_doc_ids).fetchall()]

# Step 3: variable-length expansion from those seeds
seed_sql = ", ".join(f"'{s}'" for s in seed_ids)
con.execute(f"""
    FROM GRAPH_TABLE (news_graph
      MATCH p = ANY SHORTEST
        (a:nodes)-[r:related]->{{1,2}}(b:nodes)
      WHERE a.id IN ({seed_sql})
      COLUMNS (a.id AS seed, b.id AS reached, path_length(p) AS hops)
    )
    ORDER BY seed, hops, reached
""").fetchall()
```

Result — note how the entities reached include `META`, which never
appeared in the top-4 vector hits but is two hops away from the
documents that did:

```
Tickers:AAPL  ─1→ Tickers:GOOGL
Tickers:AAPL  ─1→ Tickers:MSFT
Tickers:AAPL  ─1→ Tickers:NVDA
Tickers:AAPL  ─2→ Tickers:META
Tickers:GOOGL ─1→ Tickers:META
Tickers:NVDA  ─1→ Tickers:MSFT
...
```

That's **vector retrieval and graph expansion against parquet, in one
DuckDB session, without leaving SQL** (the only Python is moving
intermediate result lists between the three queries — and you can
inline that with a temp table if you prefer).

## When to reach past DuckDB

This setup is the right tool for:

- Local prototypes, demos, single-node batch jobs.
- Evaluation harnesses where the corpus is < ~10M chunks.
- Anything where "just give me a parquet file and a query engine" is
  more important than horizontal scaling.

You'll outgrow it when you need:

- **Multi-tenant concurrent writes / ACID over many writers.**
  DuckDB is fundamentally single-writer; reach for pgvector +
  Postgres or a hosted vector DB.
- **Billions of vectors with strict p99 latency SLAs.** Move vectors
  to Qdrant / LanceDB / Vespa; the same `vectors.parquet` file is the
  ingest format for all of them.
- **Production knowledge graphs with active editing.** A real graph
  store (Neo4j, MemGraph, or Kùzu if you don't mind Cypher) gives you
  better write paths and richer algorithms.

In all of those, your `coralbricks-context-prep` pipeline is
unchanged — only the loader on the right side of the diagram swaps.

## Recap

- **The library stops at parquet, on purpose.** `vectors.parquet`,
  `nodes.parquet`, `edges.parquet` are the contract.
- **DuckDB + `vss` + `duckpgq`** turns those parquet files into a
  full embedded RAG stack with vector search *and* variable-length
  graph traversal in one SQL session — and zero Cypher.
- **The hybrid join is free** because `hydrate()` already writes
  `Document → Entity` mentions into `edges.parquet`.
- **Swap-out is cheap.** When you outgrow DuckDB, ingest the same
  parquet into Qdrant / LanceDB / pgvector / Neo4j; the prep
  pipeline doesn't change.
