"""Coral Bricks prep — build-time context preparation for agentic AI.

`coralbricks.context_prep` is a small set of **primitives over
records** that turn the data you already have (in pandas / duckdb /
parquet / a queue) into agent-ready memory: cleaned text → chunks →
embeddings → optional knowledge graph.

Scope
-----
We do not ship:

- A parser layer (PDF/HTML/parquet IO is the user's job — they have
  pandas/duckdb/etc.).
- An orchestrator (use Airflow / Prefect / Dagster / Ray / Spark).

The verbs are pure-ish functions over ``list[dict]`` records (or
``list[str]`` when there's nothing but text) that work the same
in-process or inside any task runner.

For prototyping there's an in-memory :class:`Recipe` runner. For
million-row jobs, call the underlying primitives directly inside your
orchestrator's tasks::

    # in a Ray / Prefect / Spark task:
    from coralbricks.context_prep.chunkers import chunk_text
    from coralbricks.context_prep.embedders import create_embedder

    chunks = chunk_text(row["text"], strategy="sliding_token", target_tokens=512)
    vectors = embedder.embed_texts([c.text for c in chunks])

Public verbs (each returns an :class:`Artifact`)::

    from coralbricks.context_prep import clean, chunk, embed, enrich, hydrate, join

    cleaned  = clean(records)                                    # trafilatura
    chunks   = chunk(cleaned, strategy="sliding_token", target_tokens=512)
    vecs     = embed(chunks, model="coral_embed", batch_size=128)
    enriched = enrich(cleaned, extractors=["tickers", "dates", "urls"])
    g        = hydrate(enriched, graph="filings_graph")

All verbs run for real on concrete inputs. They also accept
``dry_run=True`` to return a placeholder Artifact for recipe
composition / tests.

Backing modules:

- :mod:`coralbricks.context_prep.cleaners` — trafilatura main-content extraction
  (no custom rules).
- :mod:`coralbricks.context_prep.chunkers` — fixed/sliding token,
  recursive-character, sentence.
- :mod:`coralbricks.context_prep.embedders` — Coral, OpenAI, Bedrock,
  sentence-transformers, DeepInfra.
- :mod:`coralbricks.context_prep.enrichers` — regex extractors + spaCy NER.
- :mod:`coralbricks.context_prep.joiners` — hash join over dict records (small
  data only — for big tables use DuckDB/Spark).
- :mod:`coralbricks.context_prep.graph` — triples → nodes/edges, plus
  :func:`merge_graphs` for combining partial graphs from distributed
  hydration workers.
"""

from __future__ import annotations

from . import chunkers, embedders, enrichers, graph
from ._records import new_record_id, normalize_records
from .artifact import Artifact, ArtifactKind
from .chunkers import (
    BaseChunker,
    Chunk,
    FixedTokenChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
    SlidingTokenChunker,
    chunk_text,
)
from .cleaners import clean_documents, clean_html
from .joiners import join_records
from .recipe import Recipe, VerbCall
from .verbs import chunk, clean, embed, enrich, hydrate, join

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core types
    "Artifact",
    "ArtifactKind",
    "Recipe",
    "VerbCall",
    "normalize_records",
    "new_record_id",
    # verbs
    "chunk",
    "embed",
    "enrich",
    "clean",
    "join",
    "hydrate",
    # chunker subpackage
    "chunkers",
    "Chunk",
    "BaseChunker",
    "FixedTokenChunker",
    "SlidingTokenChunker",
    "RecursiveCharacterChunker",
    "SentenceChunker",
    "chunk_text",
    # embedder subpackage
    "embedders",
    # enricher subpackage
    "enrichers",
    # cleaners
    "clean_documents",
    "clean_html",
    # joiners
    "join_records",
    # graph
    "graph",
]
