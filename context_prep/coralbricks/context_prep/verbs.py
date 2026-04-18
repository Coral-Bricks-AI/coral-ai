"""Prep verbs.

Each verb is a callable that takes records (or upstream artifacts) plus
config kwargs, and returns an :class:`Artifact` describing what it
produced.

Scope
-----
Coral Bricks prep is a set of **primitives over records**. The input
to every verb is either:

- A list of records (``list[dict]`` — the universal record shape, also
  accepted as ``list[str]`` when there's nothing but text yet) the
  user already has in memory (read from parquet/postgres/jsonl with
  whatever tool they like), or
- An upstream :class:`Artifact` from another verb in the same process.

Coral Bricks does **not** ship a parser layer, an orchestrator, or a
distributed runner. The verbs are pure-ish functions: at scale, the
user's orchestrator (Airflow, Prefect, Dagster, Ray, Spark, ...) is
expected to call them per-batch / per-row inside its own tasks.

In-memory ``Recipe`` execution exists for prototyping and tests. For
million-row jobs, call the underlying primitives in
:mod:`coralbricks.context_prep.chunkers`, :mod:`.embedders`, :mod:`.cleaners`,
:mod:`.enrichers`, and :mod:`.graph` directly.

Status
------
- ``chunk``, ``embed``, ``clean``, ``enrich``, ``join``, ``hydrate`` run
  for real on concrete inputs and accept ``dry_run=True`` to return
  placeholder artifacts for recipe composition / tests.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable
from typing import Any

from ._records import normalize_records
from .artifact import Artifact, ArtifactKind


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _artifact_from(
    *,
    kind: ArtifactKind,
    produced_by: str,
    inputs: Iterable[Artifact] = (),
    uri: str | None = None,
    record_count: int | None = None,
    metadata: dict[str, Any] | None = None,
    schema_hint: dict[str, Any] | None = None,
) -> Artifact:
    return Artifact(
        artifact_id=_new_id(kind.value),
        kind=kind,
        uri=uri,
        record_count=record_count,
        produced_by=produced_by,
        inputs=tuple(a.artifact_id for a in inputs),
        metadata=dict(metadata or {}),
        schema_hint=dict(schema_hint or {}),
    )


def _stub_uri(kind: str, name: str) -> str:
    """Build an opaque, in-memory URI for an artifact.

    Configurable via ``CORALBRICKS_PREP_STAGING_URI`` for users who
    want to write the URIs into their own staging/object store.
    """
    base = os.environ.get("CORALBRICKS_PREP_STAGING_URI", "memory://coral-prep")
    return f"{base.rstrip('/')}/{kind}/{name}/{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# chunk
# ---------------------------------------------------------------------------


def chunk(
    docs: Artifact | str | list,
    *,
    strategy: str = "sliding_token",
    target_tokens: int = 512,
    overlap: int = 64,
    dry_run: bool | None = None,
    **kwargs: Any,
) -> Artifact:
    """Chunk text using one of the algorithms in ``coralbricks.context_prep.chunkers``.

    Inputs:
    - ``str``: chunk a single document.
    - ``list[str]`` / ``list[dict]`` (with ``"text"`` key).
    - ``Artifact``: when produced by an upstream verb that materialised
      documents (``clean``, ``enrich``), chunks each one. Otherwise
      dry-run only.

    The default ``strategy="sliding_token"`` mirrors the most common RAG
    setup. Override with ``"fixed_token"``, ``"recursive_character"``,
    or ``"sentence"``.

    For million-row jobs, call :func:`coralbricks.context_prep.chunkers.chunk_text`
    directly inside your orchestrator's per-row task.
    """
    if dry_run is None:
        dry_run = isinstance(docs, Artifact) and "documents" not in (docs.metadata or {})

    from .chunkers import BaseChunker, make_chunker

    if dry_run:
        return _artifact_from(
            kind=ArtifactKind.CHUNKS,
            produced_by="chunk",
            inputs=[docs] if isinstance(docs, Artifact) else (),
            uri=_stub_uri("chunks", "chunk"),
            metadata={
                "strategy": strategy,
                "target_tokens": target_tokens,
                "overlap": overlap,
                **kwargs,
            },
        )

    chunker_kwargs: dict[str, Any] = {}
    if strategy in ("sliding_token", "sliding", "semantic"):
        chunker_kwargs.update({"target_tokens": target_tokens, "overlap": overlap})
    elif strategy in ("fixed_token", "fixed"):
        chunker_kwargs.update({"target_tokens": target_tokens})
    elif strategy in ("recursive_character", "recursive"):
        chunker_kwargs.update(
            {
                "chunk_size": kwargs.pop("chunk_size", target_tokens * 4),
                "chunk_overlap": kwargs.pop("chunk_overlap", overlap * 4),
            }
        )
    elif strategy == "sentence":
        chunker_kwargs.update({"target_tokens": target_tokens})
    chunker_kwargs.update(kwargs)

    chunker: BaseChunker = make_chunker(strategy, **chunker_kwargs)

    upstream_inputs: tuple[Artifact, ...] = ()
    if isinstance(docs, Artifact):
        upstream_inputs = (docs,)
        payload = (docs.metadata or {}).get("documents")
        if not payload:
            raise NotImplementedError(
                "chunk() on an Artifact without materialised documents is not "
                "supported. Pass an upstream artifact (clean/enrich), raw text, "
                "or list of records."
            )
        items: list[Any] = payload
    elif isinstance(docs, str):
        items = [docs]
    else:
        items = list(docs)

    records = normalize_records(items)
    produced_lists: list[list[dict]] = []
    for rec in records:
        produced_lists.append(
            [{**c.to_dict(), "doc_id": rec["id"]} for c in chunker.chunk(rec["text"])]
        )

    record_count = sum(len(p) for p in produced_lists)
    if isinstance(docs, str):
        chunks_payload: Any = produced_lists[0] if produced_lists else []
    else:
        chunks_payload = produced_lists

    return _artifact_from(
        kind=ArtifactKind.CHUNKS,
        produced_by="chunk",
        inputs=upstream_inputs,
        uri=_stub_uri("chunks", "chunk"),
        record_count=record_count,
        metadata={
            "strategy": strategy,
            "target_tokens": target_tokens,
            "overlap": overlap,
            "doc_count": len(records),
            "chunks": chunks_payload,
        },
    )


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


def embed(
    chunks: Artifact | str | list,
    *,
    model: str = "coral_embed",
    dimension: int = 768,
    batch_size: int = 128,
    input_type: str = "document",
    output_dir: str | None = None,
    embedder_kwargs: dict[str, Any] | None = None,
    dry_run: bool | None = None,
    embedder: Any | None = None,
) -> Artifact:
    """Embed text using the multi-provider embedder factory.

    See :mod:`coralbricks.context_prep.embedders` for the full model routing
    table (``coral_embed``, ``coral_gateway``, OpenAI, Bedrock, ``st:``,
    ``di:``).

    Inputs:
    - ``str``: embed a single text.
    - ``list[str]`` / ``list[dict]`` (with ``"text"`` key).
    - ``Artifact``: when produced by ``chunk()``, embeds each chunk in
      ``metadata["chunks"]`` (works for both single and per-doc layouts).

    Provider-specific options (e.g. ``openai_api_key``, ``aws_region``,
    ``device``, ``use_fp16``, ``coral_endpoint``) are forwarded via the
    explicit ``embedder_kwargs`` dict — the verb itself takes no
    ``**kwargs``, so typos surface as :class:`TypeError` instead of
    being silently dropped::

        embed(chunks, model="openai:text-embedding-3-large",
              embedder_kwargs={"openai_api_key": os.environ["OPENAI_API_KEY"]})

    When ``output_dir`` is set, ``vectors.parquet`` is written there
    (requires pyarrow). The file uses a fixed-size float32 list column
    for the embedding (vector-DB-friendly: Qdrant, pgvector, LanceDB,
    DuckDB ingest it directly), and stamps ``model`` + ``dimension``
    into parquet table metadata so it is self-describing. The
    in-memory ``metadata["vectors"]`` payload is preserved for chaining;
    drop the artifact if you only want the file.

    ``output_dir`` must be a **local** filesystem path. For S3 / GCS,
    write here and upload (``aws s3 cp --recursive``). Native
    ``s3://`` / ``gs://`` URIs are planned for 0.2.0.

    For million-row jobs, build an embedder once via
    :func:`coralbricks.context_prep.embedders.create_embedder` and call
    ``embedder.embed_texts(batch)`` directly inside your orchestrator's
    batch task — pair it with
    :func:`coralbricks.context_prep.embedders.write_vectors_parquet` per shard.
    """
    if dry_run is None:
        dry_run = isinstance(chunks, Artifact) and "chunks" not in (chunks.metadata or {})

    if dry_run:
        return _artifact_from(
            kind=ArtifactKind.VECTORS,
            produced_by="embed",
            inputs=[chunks] if isinstance(chunks, Artifact) else (),
            uri=output_dir or _stub_uri("vectors", model),
            metadata={
                "model": model,
                "batch_size": batch_size,
                "input_type": input_type,
                "output_dir": output_dir,
                "embedder_kwargs": dict(embedder_kwargs or {}),
            },
        )

    upstream_inputs: tuple[Artifact, ...] = ()
    if isinstance(chunks, Artifact):
        upstream_inputs = (chunks,)
        payload = (chunks.metadata or {}).get("chunks")
        if payload is None:
            raise NotImplementedError(
                "embed() on an Artifact without materialised chunks is not "
                "supported. Pass a chunk() artifact, raw text, or list of records."
            )
        if payload and isinstance(payload[0], list):
            items: list[Any] = [c for sub in payload for c in sub]
        else:
            items = list(payload)
    elif isinstance(chunks, str):
        items = [chunks]
    else:
        items = list(chunks)

    texts = [c["text"] if isinstance(c, dict) else str(c) for c in items]

    if embedder is None:
        from .embedders import create_embedder

        embedder = create_embedder(
            model_id=model,
            dimension=dimension,
            input_type=input_type,
            batch_size=batch_size,
            **(embedder_kwargs or {}),
        )

    result = embedder.embed_texts(texts)
    if isinstance(result, tuple) and len(result) == 2:
        vectors, usage = result
    else:
        vectors, usage = result, {}

    resolved_model = embedder.get_model_name()
    resolved_dim = embedder.get_dimension()

    parquet_path: str | None = None
    if output_dir is not None:
        from .embedders.parquet import write_vectors_parquet

        parquet_path = write_vectors_parquet(
            items,
            vectors,
            output_dir,
            model=resolved_model,
            dimension=resolved_dim,
            extra_metadata={
                "batch_size": batch_size,
                "input_type": input_type,
            },
        )

    return _artifact_from(
        kind=ArtifactKind.VECTORS,
        produced_by="embed",
        inputs=upstream_inputs,
        uri=parquet_path or _stub_uri("vectors", model),
        record_count=len(vectors),
        metadata={
            "model": resolved_model,
            "dimension": resolved_dim,
            "batch_size": batch_size,
            "input_type": input_type,
            "vectors": vectors,
            "usage": usage,
            "output_dir": output_dir,
            "parquet_path": parquet_path,
        },
    )


# ---------------------------------------------------------------------------
# clean (trafilatura main-content extraction)
# ---------------------------------------------------------------------------


def clean(
    docs: Artifact | str | list,
    *,
    drop_empty: bool = True,
    dry_run: bool | None = None,
    **kwargs: Any,
) -> Artifact:
    """Run trafilatura over each document's HTML body.

    Cleaning in Coral Bricks is intentionally one thing: trafilatura
    main-content extraction. No custom regex / normalisation rules ship
    with the library.

    Inputs:
    - ``str``: a single HTML document.
    - ``list[str|dict]``.
    - ``Artifact``: when produced by another verb that materialised
      documents (``enrich``), cleans them.

    For million-row jobs, call :func:`coralbricks.context_prep.cleaners.clean_html`
    directly per row inside your orchestrator's task.
    """
    if dry_run is None:
        dry_run = isinstance(docs, Artifact) and "documents" not in (docs.metadata or {})

    if dry_run:
        return _artifact_from(
            kind=ArtifactKind.CLEANED,
            produced_by="clean",
            inputs=[docs] if isinstance(docs, Artifact) else (),
            uri=_stub_uri("cleaned", "clean"),
            metadata={"cleaner": "trafilatura", "drop_empty": drop_empty, **kwargs},
        )

    from .cleaners import clean_documents

    upstream_inputs: tuple[Artifact, ...] = ()
    if isinstance(docs, Artifact):
        upstream_inputs = (docs,)
        payload = (docs.metadata or {}).get("documents")
        if payload is None:
            raise NotImplementedError(
                "clean() on an Artifact without materialised documents is not "
                "supported. Pass an upstream artifact (enrich), raw HTML, or "
                "list of records."
            )
        items: list[Any] = payload
    elif isinstance(docs, (str, dict)):
        items = [docs]
    else:
        items = list(docs)

    cleaned = clean_documents(items, drop_empty=drop_empty)

    return _artifact_from(
        kind=ArtifactKind.CLEANED,
        produced_by="clean",
        inputs=upstream_inputs,
        uri=_stub_uri("cleaned", "clean"),
        record_count=len(cleaned),
        metadata={
            "cleaner": "trafilatura",
            "drop_empty": drop_empty,
            "doc_count": len(cleaned),
            "documents": cleaned,
        },
    )


# ---------------------------------------------------------------------------
# enrich
# ---------------------------------------------------------------------------


def enrich(
    docs: Artifact | str | list,
    *,
    extractors: list | None = None,
    dry_run: bool | None = None,
    **kwargs: Any,
) -> Artifact:
    """Extract structured signals (tickers, dates, urls, NER, ...) per document.

    ``extractors`` is a list of names from
    :data:`coralbricks.context_prep.enrichers.REGISTRY` (e.g. ``"tickers"``,
    ``"dates"``, ``"urls"``) or :class:`BaseExtractor` instances. Pass a
    :class:`SpacyEntityExtractor` instance to get NER.

    Inputs follow the same pattern as ``clean()`` / ``chunk()``: raw
    text, lists, or an upstream Artifact carrying ``documents``.

    Each output record gets ``metadata["extractions"][name]`` populated.

    For million-row jobs, call
    :func:`coralbricks.context_prep.enrichers.enrich_documents` (or the
    individual extractor classes) inside your orchestrator's task.
    """
    if dry_run is None:
        dry_run = isinstance(docs, Artifact) and "documents" not in (docs.metadata or {})

    extractor_specs = extractors or ["urls", "tickers", "dates"]

    if dry_run:
        return _artifact_from(
            kind=ArtifactKind.ENRICHED,
            produced_by="enrich",
            inputs=[docs] if isinstance(docs, Artifact) else (),
            uri=_stub_uri("enriched", "enrich"),
            metadata={"extractors": [str(e) for e in extractor_specs], **kwargs},
        )

    from .enrichers import enrich_documents

    upstream_inputs: tuple[Artifact, ...] = ()
    if isinstance(docs, Artifact):
        upstream_inputs = (docs,)
        payload = (docs.metadata or {}).get("documents")
        if payload is None:
            raise NotImplementedError(
                "enrich() on an Artifact without materialised documents is not "
                "supported. Pass an upstream artifact (clean), raw text, or "
                "list of records."
            )
        items: list[Any] = payload
    elif isinstance(docs, (str, dict)):
        items = [docs]
    else:
        items = list(docs)

    enriched = enrich_documents(items, extractor_specs)

    return _artifact_from(
        kind=ArtifactKind.ENRICHED,
        produced_by="enrich",
        inputs=upstream_inputs,
        uri=_stub_uri("enriched", "enrich"),
        record_count=len(enriched),
        metadata={
            "extractors": [getattr(e, "name", str(e)) for e in extractor_specs],
            "doc_count": len(enriched),
            "documents": enriched,
        },
    )


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------


def join(
    left: Artifact | list,
    right: Artifact | list,
    *,
    on: str | list[str],
    how: str = "left",
    right_on: str | list[str] | None = None,
    suffixes: tuple[str, str] = ("_left", "_right"),
    dry_run: bool | None = None,
    **kwargs: Any,
) -> Artifact:
    """Hash-join two sets of records on a shared key.

    Backed by :func:`coralbricks.context_prep.joiners.join_records`. **Small
    data only** — for tables larger than a few hundred thousand rows,
    use DuckDB / pandas / Spark and pass the result back in.

    When passed Artifacts, looks for ``metadata["documents"]`` (from
    ``clean()``/``enrich()``) or ``metadata["records"]`` (free-form) on
    each side.
    """
    if dry_run is None:
        dry_run = (
            isinstance(left, Artifact)
            and isinstance(right, Artifact)
            and (
                "documents" not in (left.metadata or {}) and "records" not in (left.metadata or {})
            )
        )

    if dry_run:
        return _artifact_from(
            kind=ArtifactKind.JOINED,
            produced_by="join",
            inputs=[a for a in (left, right) if isinstance(a, Artifact)],
            uri=_stub_uri("joined", "join"),
            metadata={"on": on, "how": how, "right_on": right_on, **kwargs},
        )

    from .joiners import join_records

    def _materialise(side: Artifact | list) -> list[Any]:
        if isinstance(side, Artifact):
            meta = side.metadata or {}
            return list(meta.get("documents") or meta.get("records") or [])
        return list(side)

    left_records = _materialise(left)
    right_records = _materialise(right)

    joined = join_records(
        left_records,
        right_records,
        on=on,
        right_on=right_on,
        how=how,  # type: ignore[arg-type]
        suffixes=suffixes,
    )

    upstream = [a for a in (left, right) if isinstance(a, Artifact)]
    return _artifact_from(
        kind=ArtifactKind.JOINED,
        produced_by="join",
        inputs=upstream,
        uri=_stub_uri("joined", "join"),
        record_count=len(joined),
        metadata={"on": on, "how": how, "right_on": right_on, "records": joined},
    )


# ---------------------------------------------------------------------------
# hydrate (graph)
# ---------------------------------------------------------------------------


def hydrate(
    enriched: Artifact | list,
    *,
    graph: str,
    extractors: list | None = None,
    output_dir: str | None = None,
    dry_run: bool | None = None,
    **kwargs: Any,
) -> Artifact:
    """Aggregate triples extracted from documents into a nodes+edges graph.

    Defaults to a :class:`CooccurrenceExtractor` driven by entity
    buckets emitted by :func:`enrich`. Pass custom
    :class:`BaseTripleExtractor` instances via ``extractors=``.

    When ``output_dir`` is set, ``nodes.parquet`` and ``edges.parquet``
    are written there (requires pyarrow).

    For million-row jobs, run :func:`coralbricks.context_prep.graph.hydrate_graph`
    on each shard inside your orchestrator's tasks, then combine the
    results with :func:`coralbricks.context_prep.graph.merge_graphs` before
    handing them to your downstream graph store.
    """
    if dry_run is None:
        dry_run = isinstance(enriched, Artifact) and "documents" not in (enriched.metadata or {})

    if dry_run:
        return _artifact_from(
            kind=ArtifactKind.GRAPH,
            produced_by="hydrate",
            inputs=[enriched] if isinstance(enriched, Artifact) else (),
            uri=_stub_uri("graph", graph),
            metadata={
                "graph": graph,
                "extractors": [str(e) for e in (extractors or [])],
                "output_dir": output_dir,
                **kwargs,
            },
        )

    from .graph import (
        BaseTripleExtractor,
        CooccurrenceExtractor,
        hydrate_graph,
    )

    upstream_inputs: tuple[Artifact, ...] = ()
    if isinstance(enriched, Artifact):
        upstream_inputs = (enriched,)
        payload = (enriched.metadata or {}).get("documents")
        if payload is None:
            raise NotImplementedError(
                "hydrate() on an Artifact without materialised documents is not "
                "supported. Pass an enrich() artifact or list of records."
            )
        items: list[Any] = payload
    else:
        items = list(enriched)

    records = normalize_records(items)

    if not extractors:
        sample_meta = records[0]["metadata"] if records else {}
        ext_buckets = list((sample_meta or {}).get("extractions", {}).keys())
        sources = {b: b.title() for b in ext_buckets} or {"tickers": "Ticker"}
        triple_extractors: list[BaseTripleExtractor] = [CooccurrenceExtractor(sources)]
    else:
        triple_extractors = list(extractors)

    graph_payload = hydrate_graph(records, triple_extractors, output_dir=output_dir)

    return _artifact_from(
        kind=ArtifactKind.GRAPH,
        produced_by="hydrate",
        inputs=upstream_inputs,
        uri=output_dir or _stub_uri("graph", graph),
        record_count=graph_payload.get("node_count", 0) + graph_payload.get("edge_count", 0),
        metadata={
            "graph": graph,
            "node_count": graph_payload.get("node_count", 0),
            "edge_count": graph_payload.get("edge_count", 0),
            "nodes": graph_payload.get("nodes", []),
            "edges": graph_payload.get("edges", []),
            "output_dir": graph_payload.get("output_dir"),
        },
    )
