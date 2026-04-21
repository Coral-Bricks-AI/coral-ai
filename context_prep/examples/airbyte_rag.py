"""End-to-end Airbyte → context_prep → DuckDB RAG.

Grown version of ``embedded_rag_duckdb.py`` — same retrieval pipeline,
but the starting records come from an Airbyte destination directory
instead of a hardcoded Python list::

    Airbyte JSONL (Local JSON or S3 sync)
              │
              ▼
    coralbricks.connectors.airbyte.read_airbyte_output()  ──► list[dict]
              │
              ▼
    chunk → embed → enrich → hydrate            (coralbricks.context_prep)
              │        │
              ▼        ▼
       vectors.parquet   nodes + edges.parquet
              │        │
              └───┬────┘
                  ▼
      one DuckDB session: vss (HNSW) + duckpgq (SQL/PGQ)

Why this example exists
-----------------------
``coralbricks-context-prep`` deliberately ships no file loaders. The
companion package ``coralbricks-connectors`` is the bridge — its
``coralbricks.connectors.airbyte`` submodule reads Airbyte JSONL,
strips the envelope, and returns records in the exact shape every
``context_prep`` verb already accepts. Everything after that is
byte-for-byte identical to ``embedded_rag_duckdb.py``.

Run
---
::

    pip install -e integrations/connectors
    pip install -e 'context_prep[graph,chunkers]'
    pip install duckdb sentence-transformers
    python context_prep/examples/airbyte_rag.py

By default this reads the checked-in HackerNews-style fixtures shipped
with ``coralbricks-connectors``. Point ``AIRBYTE_PATH`` at your own
destination directory for a real run — Local JSON default root is
``/tmp/airbyte_local/<path>``; for S3, first
``aws s3 sync s3://bucket/prefix /tmp/ab-sync/``.

If ``sentence-transformers`` is not installed, a deterministic hash
embedder is used so the SQL plumbing still runs (vector hits become
meaningless, but the structure of the demo is intact). DuckPGQ is a
soft requirement — if it fails to load, the graph-traversal section
is skipped.
"""

from __future__ import annotations

import hashlib
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

from coralbricks.connectors.airbyte import read_airbyte_output
from coralbricks.context_prep import chunk, embed, enrich, hydrate
from coralbricks.context_prep.embedders import BaseEmbedder

# ---------------------------------------------------------------------------
# 0. Data source.
# ---------------------------------------------------------------------------
# Point at your Airbyte destination directory. The default is the
# checked-in HackerNews-style fixture that ships with
# ``coralbricks-connectors`` so this file runs out of the box in the monorepo.
AIRBYTE_PATH = (
    Path(__file__).resolve().parents[2]
    / "integrations"
    / "connectors"
    / "airbyte"
    / "tests"
    / "fixtures"
)

QUERY = "How did AAPL react to the Fed and which AI names moved with it?"


# ---------------------------------------------------------------------------
# 1. Embedder — real sentence-transformers if installed, hash fallback
#    otherwise so the plumbing still exercises end-to-end without a model
#    download.
# ---------------------------------------------------------------------------


class _LocalSTEmbedder(BaseEmbedder):
    """Pull a small model from HuggingFace Hub. We don't go through
    ``create_embedder("st:...")`` because that path expects Coral's
    internal S3 mirror; OSS users get models straight from the Hub."""

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_id)
        self._name = model_id
        self._dim = int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> tuple[list[list[float]], dict]:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs], {}

    def get_model_name(self) -> str:
        return self._name

    def get_dimension(self) -> int:
        return self._dim


class _HashEmbedder(BaseEmbedder):
    """Deterministic 64-dim fallback. Vector hits are meaningless, but
    the full plumbing (parquet writes, DuckDB load, HNSW index, graph
    queries) still exercises end-to-end."""

    _DIM = 64

    def embed_texts(self, texts: list[str]) -> tuple[list[list[float]], dict]:
        out: list[list[float]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            # 32-byte digest × 4 → 128 bytes → 64 big-endian shorts.
            shorts = struct.unpack(">64h", digest * 4)
            vec = [s / 32768.0 for s in shorts][: self._DIM]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out, {}

    def get_model_name(self) -> str:
        return "hash-fallback"

    def get_dimension(self) -> int:
        return self._DIM


def _build_embedder() -> BaseEmbedder:
    try:
        emb = _LocalSTEmbedder()
        print(
            f"[embedder] sentence-transformers: {emb.get_model_name()}  dim={emb.get_dimension()}"
        )
        return emb
    except Exception as exc:
        print(
            f"[embedder] sentence-transformers unavailable "
            f"({exc.__class__.__name__}); falling back to hash embedder "
            "(vector hits will be meaningless)."
        )
        return _HashEmbedder()


# ---------------------------------------------------------------------------
# 2. Read Airbyte output → records → prep pipeline.
# ---------------------------------------------------------------------------


def _prep(work_dir: Path, embedder: BaseEmbedder) -> tuple[Path, Path, Path]:
    records = read_airbyte_output(
        AIRBYTE_PATH,
        stream="stories",  # only files whose name contains "stories"
        text_field="title",  # HN story row: searchable text = title
        id_field=lambda d: f"hn-{d['id']}",
    )
    print(f"[airbyte] read {len(records)} records from {AIRBYTE_PATH}")

    chunks = chunk(records, strategy="sliding_token", target_tokens=48, overlap=8)
    vectors = embed(chunks, embedder=embedder, output_dir=str(work_dir))
    enriched = enrich(records, extractors=["tickers"])
    g = hydrate(enriched, graph="news", output_dir=str(work_dir))

    vec_path = Path(vectors.metadata["parquet_path"])
    nodes_path = work_dir / "nodes.parquet"
    edges_path = work_dir / "edges.parquet"

    print(
        f"[prep] {chunks.record_count} chunks / {vectors.record_count} vectors "
        f"@ dim {vectors.metadata['dimension']} / "
        f"{g.metadata['node_count']} nodes / {g.metadata['edge_count']} edges"
    )
    return vec_path, nodes_path, edges_path


# ---------------------------------------------------------------------------
# 3. DuckDB session — parquet → HNSW + property graph (identical wiring
#    to embedded_rag_duckdb.py).
# ---------------------------------------------------------------------------


def _open_duckdb(vec_path: Path, nodes_path: Path, edges_path: Path, dim: int) -> tuple[Any, bool]:
    import duckdb

    con = duckdb.connect()
    con.execute(
        f"""
        CREATE TABLE vectors AS
        SELECT doc_id, chunk_index, text,
               CAST(vector AS FLOAT[{dim}]) AS vector
        FROM read_parquet('{vec_path}')
        """
    )
    con.execute("CREATE TABLE nodes AS SELECT * FROM read_parquet(?)", [str(nodes_path)])
    con.execute("CREATE TABLE edges AS SELECT * FROM read_parquet(?)", [str(edges_path)])

    con.execute("INSTALL vss; LOAD vss;")
    con.execute("CREATE INDEX vec_hnsw ON vectors USING HNSW (vector) WITH (metric = 'cosine')")

    has_pgq = True
    try:
        con.execute("INSTALL duckpgq FROM community; LOAD duckpgq;")
        con.execute(
            """
            CREATE PROPERTY GRAPH news_graph
              VERTEX TABLES (nodes PROPERTIES (id, label, value))
              EDGE TABLES (
                edges
                  SOURCE KEY (src) REFERENCES nodes (id)
                  DESTINATION KEY (dst) REFERENCES nodes (id)
                  LABEL related
              )
            """
        )
    except Exception as exc:
        print(
            f"[duckpgq] unavailable ({exc.__class__.__name__}); "
            "skipping the graph-traversal section."
        )
        has_pgq = False

    return con, has_pgq


def _vector_topk(con: Any, query_vec: list[float], k: int = 4) -> list[tuple]:
    dim = len(query_vec)
    return con.execute(
        f"""
        SELECT doc_id, text,
               array_cosine_distance(vector, ?::FLOAT[{dim}]) AS d
        FROM vectors
        ORDER BY d
        LIMIT {k}
        """,
        [query_vec],
    ).fetchall()


def _graph_neighbors(con: Any, seeds: list[str], max_hops: int = 2) -> list[tuple]:
    if not seeds:
        return []
    seed_sql = ", ".join(f"'{s}'" for s in seeds)
    return con.execute(
        f"""
        FROM GRAPH_TABLE (news_graph
          MATCH p = ANY SHORTEST
            (a:nodes)-[r:related]->{{1,{max_hops}}}(b:nodes)
          WHERE a.id IN ({seed_sql})
          COLUMNS (a.id AS seed, b.id AS reached, path_length(p) AS hops)
        )
        ORDER BY seed, hops, reached
        """
    ).fetchall()


# ---------------------------------------------------------------------------
# 4. Print helpers.
# ---------------------------------------------------------------------------


def _hr(title: str) -> None:
    print()
    print("─" * 72)
    print(title)
    print("─" * 72)


def _print_vector_hits(rows: list[tuple]) -> None:
    for doc_id, text, dist in rows:
        snippet = (text[:64] + "…") if len(text) > 64 else text
        print(f"  d={dist:.4f}  doc={doc_id:<14}  {snippet!r}")


def _print_graph_rows(rows: list[tuple]) -> None:
    if not rows:
        print("  (no paths)")
        return
    for seed, reached, hops in rows:
        print(f"  {seed} ─{hops}→ {reached}")


# ---------------------------------------------------------------------------
# 5. Main.
# ---------------------------------------------------------------------------


def main() -> None:
    embedder = _build_embedder()
    work_dir = Path(tempfile.mkdtemp(prefix="cb_airbyte_rag_"))
    try:
        vec_path, nodes_path, edges_path = _prep(work_dir, embedder)
        con, has_pgq = _open_duckdb(vec_path, nodes_path, edges_path, dim=embedder.get_dimension())

        _hr(f"1. Vector top-K   query={QUERY!r}")
        qv, _ = embedder.embed_texts([QUERY])
        _print_vector_hits(_vector_topk(con, qv[0], k=4))

        if has_pgq:
            seed = "Tickers:AAPL"
            _hr(f"2. Variable-length graph traversal  seed={seed}  hops=1..2")
            _print_graph_rows(_graph_neighbors(con, [seed], max_hops=2))

        print()
        print(f"Done. Parquet artifacts kept at {work_dir}")
        print("(deleted by the cleanup at the end — remove the shutil.rmtree to keep them).")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
