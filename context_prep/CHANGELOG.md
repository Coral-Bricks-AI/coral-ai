# Changelog

All notable changes to `coralbricks-context-prep` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-18

Initial open-source release.

### Added
- Verbs: `clean`, `chunk`, `embed`, `enrich`, `hydrate`, `join`.
- `Recipe` runner for in-memory prototyping.
- `Artifact` lineage handles.
- Chunkers: `FixedTokenChunker`, `SlidingTokenChunker`,
  `RecursiveCharacterChunker`, `SentenceChunker`.
- Multi-provider embedder factory: Coral, Coral Gateway, OpenAI,
  AWS Bedrock, sentence-transformers, DeepInfra. Provider options pass
  through an explicit ``embedder_kwargs={...}`` dict — typos surface
  as ``TypeError`` instead of being silently dropped.
- Cleaner: trafilatura-only HTML main-content extraction.
- Enrichers: regex extractors (tickers, urls, emails, dates, money,
  phones, hashtags, mentions) + spaCy NER adapter.
- Joiner: pure-Python hash join over dict records.
- Graph: triple extraction → nodes/edges, parquet output, plus
  `merge_graphs(*partials)` for distributed-hydration reduce step.
- Vector parquet sink:
  `coralbricks.context_prep.embedders.write_vectors_parquet(...)` and
  `embed(..., output_dir=...)` — single self-describing
  `vectors.parquet` per call with a fixed-size float32 list column
  (Qdrant / pgvector / LanceDB / DuckDB ingest it directly), model +
  dimension stamped in parquet table metadata.
- 49 unit tests, GitHub Actions CI on Python 3.10 / 3.11 / 3.12.
