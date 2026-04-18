# Architecture

`coralbricks-context-prep` is a small library of **primitives over records**.
This document captures the design principles, the verb model, and what
is intentionally _out of scope_.

## Design principles

1. **Records, not classes.** Every public function takes and returns
   `list[dict]` (or `list[str]`, normalized internally). The library
   ships no domain objects users have to learn — `dict` is the
   universal record shape across pandas, duckdb, parquet, JSON APIs,
   and message queues. `normalize_records()` is the only shape adapter.

2. **Per-record, in-process.** Each verb operates on the records it
   was handed, in the current Python process. There is no global
   state, no client, no event loop. This makes the library trivially
   composable inside any orchestrator's per-row / per-batch task.

3. **Lazy heavy imports.** Optional backends (sentence-transformers,
   torch, boto3, spaCy, pyarrow, trafilatura) are imported inside the
   function that needs them, never at module top level. `pip install
   coralbricks-context-prep` pulls in **zero hard dependencies**; users opt
   into what they need with extras.

4. **Explicit lineage.** Every verb returns an `Artifact` with
   `artifact_id`, `produced_by`, `inputs`, and an optional
   `metadata["..."]` payload carrying the materialised data. This
   keeps recipes inspectable without a full lineage tracker.

5. **One way to do each common thing.** We deliberately ship only
   trafilatura for HTML cleaning, four chunkers for text splitting,
   and a small registry of regex extractors. Users who want exotic
   variants register their own classes — they don't need to wait for
   us.

## What's out of scope

- **File loaders** (PDF/HTML/parquet IO). Use `pandas`, `duckdb`,
  `pyarrow`, or your warehouse client.
- **Orchestration** (DAGs, retries, scheduling, parallelism). Use
  Airflow, Prefect, Dagster, Ray, or Spark.
- **Vector / graph stores.** We produce vectors and node/edge tables;
  pushing them into Qdrant / pgvector / Neo4j / cuGraph is the user's
  job (and a thirty-line function).
- **Object-store sinks.** `embed(..., output_dir=...)` and
  `hydrate(..., output_dir=...)` write parquet to a **local** path.
  For S3 / GCS, write to a scratch directory then upload
  (`aws s3 cp --recursive`, `gsutil cp -r`). Native `s3://` / `gs://`
  URIs via pyarrow's filesystem registry are planned for 0.2.0.
- **Query / retrieval.** Out of scope — this library is
  build-time-only.
- **Agent runtime.** Out of scope — see the closed-source Coral Bricks
  platform for that.

## Verb model

```
            ┌──────────┐
records ──► │  clean   │ ──► cleaned records
            └──────────┘
                  │
                  ▼
            ┌──────────┐
            │  chunk   │ ──► chunks (per-doc)
            └──────────┘
                  │
                  ▼
            ┌──────────┐
            │  embed   │ ──► vectors
            └──────────┘

cleaned ──► ┌──────────┐
            │  enrich  │ ──► enriched (with metadata.extractions)
            └──────────┘
                  │
                  ▼
            ┌──────────┐
            │ hydrate  │ ──► nodes + edges
            └──────────┘
```

Each verb:

- accepts records (`list[dict]`, `list[str]`, or single `str`/`dict`)
  **or** an upstream `Artifact`,
- runs synchronously and returns an `Artifact`,
- supports `dry_run=True` to skip computation and return a
  placeholder Artifact (useful for `Recipe.plan()` and tests),
- carries the materialised payload under
  `artifact.metadata["chunks"|"vectors"|"documents"|"nodes"|"edges"]`.

## Recipe vs primitives

`Recipe` is a single-process, in-memory convenience for prototyping.
For production / scale, **call the primitives directly** in your
orchestrator's task function:

| Verb              | Primitive                                              |
| ----------------- | ------------------------------------------------------ |
| `clean(...)`      | `coralbricks.context_prep.cleaners.clean_html(html_str)`       |
| `chunk(...)`      | `coralbricks.context_prep.chunkers.chunk_text(text, ...)`      |
| `embed(...)`      | `embedder.embed_texts(batch_texts)` + `coralbricks.context_prep.embedders.write_vectors_parquet(...)` |
| `enrich(...)`     | `coralbricks.context_prep.enrichers.enrich_documents(...)`     |
| `hydrate(...)`    | `coralbricks.context_prep.graph.hydrate_graph(records, ...)`   |

`merge_graphs(*partials)` is the one stage that isn't embarrassingly
parallel — use it as your reduce step after distributed hydration.

## Compatibility & versioning

- Python 3.10+.
- SemVer. The public surface is everything re-exported from
  `coralbricks.context_prep` (`__all__`). Anything under `coralbricks.context_prep._*`
  is internal and may change without notice.
- We follow a "no deprecation surprises" rule: if we're going to
  remove something, we'll deprecate-warn for one minor release first.

## Why `coralbricks.context_prep` and not `coral_prep`?

The `coralbricks` namespace is a [PEP 420 namespace
package](https://peps.python.org/pep-0420/). The same import path is
shared by:

- `coralbricks-context-prep` (this OSS package) — `coralbricks.context_prep.*`
- the closed-source Coral Bricks platform SDK — `coralbricks.client`,
  `coralbricks.project`, etc.

This lets users start with the OSS library and graduate to the
managed platform without rewriting imports.
