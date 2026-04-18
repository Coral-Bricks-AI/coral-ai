# coralbricks-context-prep

> Build-time context preparation for agentic AI.
> Clean → chunk → embed → enrich → hydrate. Plain Python, no orchestrator, no parser layer.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![PyPI](https://img.shields.io/badge/pypi-coralbricks--context--prep-orange.svg)](https://pypi.org/project/coralbricks-context-prep/)

`coralbricks-context-prep` is the open-source context-prep layer of the
[Coral Bricks](https://coralbricks.ai) platform. It ships a small,
opinionated set of **primitives over records**: turn the data you
already have (in pandas / duckdb / parquet / a queue) into agent-ready
memory — cleaned text, chunks, embeddings, and an optional knowledge
graph.

## Why another RAG library?

Most RAG / context libraries try to do three things at once:

1. Load files (PDF, HTML, parquet, S3, ...).
2. Transform them (clean, chunk, embed, NER, summarise, ...).
3. Orchestrate the pipeline (DAG, retries, scheduling, parallelism).

`coralbricks-context-prep` only does #2.

- **No file loaders.** Use `pandas`, `duckdb`, `pyarrow`, or your
  warehouse client. They are better at IO than we will ever be.
- **No orchestrator.** Use Airflow, Prefect, Dagster, Ray, or Spark.
  They are better at scheduling than we will ever be.
- **Just the transformations.** Pure Python functions over
  `list[dict]` records that work the same in a Jupyter cell, a
  Prefect task, or a Spark UDF.

This makes the library tiny, testable, easy to drop into existing
data pipelines, and easy to scale: every primitive is per-record.

## Install

```bash
pip install coralbricks-context-prep                  # core (zero hard deps)
pip install 'coralbricks-context-prep[chunkers]'      # + tiktoken
pip install 'coralbricks-context-prep[cleaners]'      # + trafilatura
pip install 'coralbricks-context-prep[embed-api]'     # + openai, requests
pip install 'coralbricks-context-prep[embed-st]'      # + sentence-transformers, torch
pip install 'coralbricks-context-prep[embed-bedrock]' # + boto3
pip install 'coralbricks-context-prep[graph]'         # + pyarrow
pip install 'coralbricks-context-prep[enrichers-spacy]' # + spaCy
pip install 'coralbricks-context-prep[all]'           # everything above
```

## 60-second quickstart

```python
from coralbricks.context_prep import clean, chunk, embed, enrich, hydrate

# Records you already loaded with whatever tool you like.
records = [
    {"id": "doc-1", "text": "<html><body><p>$AAPL is up.</p></body></html>"},
    {"id": "doc-2", "text": "<html><body><p>$MSFT and $AAPL rallied today.</p></body></html>"},
]

cleaned  = clean(records)                        # trafilatura main-content extraction
chunks   = chunk(cleaned, strategy="sliding_token", target_tokens=512)
vectors  = embed(chunks,  model="coral_embed")   # or openai, bedrock, st:bge-m3, ...
enriched = enrich(cleaned, extractors=["tickers", "dates", "urls"])
graph    = hydrate(enriched, graph="news")       # entity co-occurrence graph

# Push to your vector store / graph DB / object store of choice.
# (Or, for a fully embedded demo: see examples/embedded_rag_duckdb.py
# — vectors + graph in one local DuckDB session, no servers.)
```

## Verbs

| Verb | What it does |
| --- | --- |
| `clean(records)` | Trafilatura main-content extraction. No custom rules. |
| `chunk(records, strategy=...)` | `fixed_token`, `sliding_token`, `recursive_character`, or `sentence`. |
| `embed(chunks, model=...)` | Coral, OpenAI, Bedrock, sentence-transformers, DeepInfra. Parquet output optional. |
| `enrich(records, extractors=...)` | Regex extractors (tickers/urls/dates/money/...) + spaCy NER. |
| `hydrate(records, graph=...)` | Triple extraction → deduplicated nodes/edges. Parquet output optional. |
| `join(left, right, on=...)` | Hash join over dict records (small data; for big, use DuckDB). |

Every verb returns an `Artifact` describing what it produced
(record_count, lineage, materialised payload), and accepts
`dry_run=True` for composing recipes without executing them.

## Records, not classes

The universal record shape across pandas, duckdb, parquet, JSON, and
message queues is `dict`. So that is the only shape `prep` accepts:

```python
{"id": "doc-1", "text": "...", "source": None, "metadata": {}}
```

`source` and `metadata` are optional. We do not ship a custom
`Document` class; `normalize_records()` does the minimal coercion in
one place so verbs stay simple.

## At scale

For million-row jobs, drop the `Recipe` runner and call the primitives
directly inside your orchestrator's tasks:

```python
# in a Ray / Prefect / Spark / Airflow task:
from coralbricks.context_prep.chunkers import chunk_text
from coralbricks.context_prep.embedders import create_embedder

embedder = create_embedder("st:BAAI/bge-m3", dimension=1024)

def transform_partition(rows):
    out = []
    for row in rows:
        chunks = chunk_text(row["text"], strategy="sliding_token", target_tokens=512)
        texts = [c.text for c in chunks]
        vectors, _ = embedder.embed_texts(texts)
        out.extend(
            {"doc_id": row["id"], **c.to_dict(), "vector": v}
            for c, v in zip(chunks, vectors)
        )
    return out
```

For embedding jobs that should land on disk instead of in RAM, write
each shard to its own `vectors.parquet` (fixed-size float32 list
column — Qdrant, pgvector, LanceDB, and DuckDB ingest it directly):

```python
from coralbricks.context_prep.embedders import write_vectors_parquet

vectors, _ = embedder.embed_texts(texts)
write_vectors_parquet(
    chunks, vectors, output_dir=f"/scratch/run/shard={i:05d}",
    model=embedder.get_model_name(),
    dimension=embedder.get_dimension(),
)
```

(Or just call `embed(chunks, output_dir="/scratch/run", embedder=embedder)`
inside a single-process pipeline.)

For knowledge graphs, build per-shard graphs in parallel and combine
them with `merge_graphs(*partials)`:

```python
from coralbricks.context_prep.graph import hydrate_graph, merge_graphs

partials = pool.map(lambda shard: hydrate_graph(shard, extractors), shards)
full = merge_graphs(*partials)   # nodes deduped, edge weights summed
```

## Comparison to other tools

| Tool                | Loaders | Transforms | Orchestrator | Vector store | License |
| ------------------- | :-----: | :--------: | :----------: | :----------: | :----:  |
| LangChain           | yes     | yes        | partial      | yes          | MIT     |
| LlamaIndex          | yes     | yes        | partial      | yes          | MIT     |
| Unstructured.io     | yes     | yes        | no           | no           | Apache  |
| **coralbricks-context-prep**| **no**  | **yes**    | **no**       | **no**       | **Apache** |

If you want batteries-included, use LangChain or LlamaIndex. If you
already have a data stack and want a small, sharp library that does
one job well, use `coralbricks-context-prep`.

## Examples

See [`examples/`](examples/):

- [`rag_quickstart.py`](examples/rag_quickstart.py) — end-to-end clean → chunk → embed.
- [`knowledge_graph.py`](examples/knowledge_graph.py) — build an entity graph from news.
- [`distributed_hydrate.py`](examples/distributed_hydrate.py) — `hydrate_graph` + `merge_graphs` reduce.
- [`embedded_rag_duckdb.py`](examples/embedded_rag_duckdb.py) — close the loop:
  load the prepared parquet into one DuckDB session and run vector
  top-K + variable-length graph traversal (via `vss` + `duckpgq`,
  no servers, no Cypher). See the
  [tutorial](docs/TUTORIAL_EMBEDDED_RAG.md) for the walk-through.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design principles & verb model.
- [`docs/EXTENDING.md`](docs/EXTENDING.md) — write your own chunker / extractor / embedder.
- [`docs/TUTORIAL_EMBEDDED_RAG.md`](docs/TUTORIAL_EMBEDDED_RAG.md) — end-to-end embedded RAG with DuckDB (vectors + graph in one session).
- [`CHANGELOG.md`](CHANGELOG.md) — release history.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs welcome on
[GitHub](https://github.com/Coral-Bricks-AI/coral-ai).

## License

[Apache 2.0](LICENSE) © Coral Bricks AI.
