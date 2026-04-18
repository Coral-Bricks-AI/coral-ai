"""Parquet writer for vectors produced by :func:`coralbricks.context_prep.embed`.

Writes a single ``vectors.parquet`` file with one row per (chunk, vector)
pair. The vector column is a **fixed-size float32 list** (the most
vector-DB-friendly layout — Qdrant, pgvector, LanceDB, and DuckDB all
ingest it directly). Model name + dimension are stamped into parquet
table metadata so the file is self-describing.

For S3 / GCS, write to a local scratch directory and upload — native
object-store URIs are planned for 0.2.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Union


def write_vectors_parquet(
    items: Sequence[Any],
    vectors: Sequence[Sequence[float]],
    output_dir: Union[str, Path],
    *,
    model: str,
    dimension: int,
    file_name: str = "vectors.parquet",
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Persist ``(items, vectors)`` rows to ``<output_dir>/<file_name>``.

    Args:
        items: The texts / chunks the vectors correspond to. Each item
            may be a string or a chunk dict
            (``{"text": ..., "doc_id": ..., "index": ..., "start": ...,
            "end": ..., "token_count": ...}``). Items are positionally
            aligned with ``vectors``.
        vectors: ``list[list[float]]`` (or any sequence-of-sequences).
        output_dir: **Local** filesystem directory. Created if missing.
            For S3 / GCS, write here and upload (``aws s3 cp
            --recursive``). Native ``s3://`` / ``gs://`` URIs planned
            for 0.2.0.
        model: Embedder model name; written to parquet metadata.
        dimension: Embedding dimension. Used to build a fixed-size list
            column.
        file_name: Override the output file name.
        extra_metadata: Additional key/value pairs to stamp into parquet
            table metadata (e.g. ``batch_size``, ``input_type``,
            ``run_id``). Values are stringified.

    Returns:
        Absolute path to the written parquet file.

    Raises:
        RuntimeError: if ``pyarrow`` is not installed.
        ValueError: if ``len(items) != len(vectors)`` or any vector
            length differs from ``dimension``.
    """
    if len(items) != len(vectors):
        raise ValueError(
            f"items / vectors length mismatch: {len(items)} vs {len(vectors)}"
        )

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "write_vectors_parquet requires pyarrow; install with "
            "`pip install pyarrow` (or `pip install 'coralbricks-context-prep[graph]'`)."
        ) from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / file_name

    rows: list[dict[str, Any]] = []
    for it, vec in zip(items, vectors):
        if len(vec) != dimension:
            raise ValueError(
                f"vector length {len(vec)} does not match dimension {dimension}"
            )
        if isinstance(it, dict):
            rows.append(
                {
                    "doc_id": it.get("doc_id"),
                    "chunk_index": it.get("index"),
                    "text": it.get("text", ""),
                    "start": it.get("start"),
                    "end": it.get("end"),
                    "token_count": it.get("token_count"),
                    "vector": [float(x) for x in vec],
                }
            )
        else:
            rows.append(
                {
                    "doc_id": None,
                    "chunk_index": None,
                    "text": str(it),
                    "start": None,
                    "end": None,
                    "token_count": None,
                    "vector": [float(x) for x in vec],
                }
            )

    schema = pa.schema(
        [
            pa.field("doc_id", pa.string()),
            pa.field("chunk_index", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("start", pa.int64()),
            pa.field("end", pa.int64()),
            pa.field("token_count", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), dimension)),
        ],
        metadata={
            b"model": model.encode(),
            b"dimension": str(dimension).encode(),
            b"count": str(len(rows)).encode(),
            **{
                k.encode(): str(v).encode()
                for k, v in (extra_metadata or {}).items()
            },
        },
    )

    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)
    return str(path)


__all__ = ["write_vectors_parquet"]
