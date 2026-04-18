"""End-to-end embedded RAG with DuckDB — vectors + graph in one session.

Pipeline:

    chunk → embed → enrich → hydrate
              │       │
              ▼       ▼
      vectors.parquet  nodes.parquet + edges.parquet
              │       │
              └──┬────┘
                 ▼
    one DuckDB session: VSS (HNSW) + DuckPGQ (SQL/PGQ)

(``clean()`` would prepend trafilatura main-content extraction if the
input were full HTML pages — see ``rag_quickstart.py``. The records
below are already plain text, so we skip it.)

What this demonstrates
----------------------
1. Vector search over chunks using DuckDB's ``vss`` extension (HNSW
   index, cosine distance).
2. Variable-length graph traversal over entity co-occurrence using
   DuckDB's ``duckpgq`` extension (SQL/PGQ from the SQL:2023 standard
   — *not* Cypher).
3. A hybrid query: vector top-K → seed entities (via the document
   ``mentions`` edges already in ``edges.parquet``) → 1..3 hop graph
   expansion. All in one DuckDB connection over the parquet files
   that ``embed()`` and ``hydrate()`` produced.

Why this lives in ``examples/`` and not in the library
------------------------------------------------------
``coralbricks-context-prep`` deliberately ships no DB drivers and no
loaders. This file shows the downstream wiring so you can copy the
pattern into your own pipeline. The same parquet files load just as
cleanly into LanceDB, Qdrant, pgvector, or any other store — the lib
stops at parquet on purpose.

Run
---
::

    pip install 'coralbricks-context-prep[graph,chunkers]' duckdb \\
                sentence-transformers
    python examples/embedded_rag_duckdb.py

If ``sentence-transformers`` is not installed, a deterministic hash
embedder is used so the SQL plumbing still runs (vector hits become
arbitrary, but the structure of the demo is intact). DuckPGQ is a
hard requirement — failing to install it is a clear error rather than
a silent fallback to another engine.
"""

from __future__ import annotations

import hashlib
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

from coralbricks.context_prep import chunk, embed, enrich, hydrate
from coralbricks.context_prep.embedders import BaseEmbedder

# ---------------------------------------------------------------------------
# 0. Sample corpus — small but rich enough that the graph has structure.
# ---------------------------------------------------------------------------

NEWS: list[dict[str, str]] = [
    {
        "id": "n1",
        "text": (
            "$AAPL and $MSFT both rallied on the Fed announcement; "
            "$NVDA followed late in the session."
        ),
    },
    {
        "id": "n2",
        "text": "$AAPL beat earnings; $GOOGL also up after strong cloud results.",
    },
    {
        "id": "n3",
        "text": "$NVDA hit a new high; $MSFT followed on AI demand.",
    },
    {
        "id": "n4",
        "text": "$TSLA dropped after delivery miss; $F and $GM caught a small bid.",
    },
    {
        "id": "n5",
        "text": "$GOOGL and $META lead the megacap rally; $AAPL lagged.",
    },
    {
        "id": "n6",
        "text": "$NVDA partners with $MSFT on a new inference cluster.",
    },
    {
        "id": "n7",
        "text": "$AMZN raised guidance; $GOOGL and $META also higher.",
    },
    {
        "id": "n8",
        "text": "$JPM and $GS rallied on rate-cut expectations.",
    },
]

QUERY = "How did AAPL react to the Fed and which AI names moved with it?"


# ---------------------------------------------------------------------------
# 1. Embedder — a real local sentence-transformers model if available,
#    a deterministic hash fallback otherwise. Both implement BaseEmbedder
#    so they plug straight into `embed(...)`.
# ---------------------------------------------------------------------------


class _LocalSTEmbedder(BaseEmbedder):
    """Tiny sentence-transformers wrapper that pulls from HuggingFace Hub.

    We don't go through ``create_embedder("st:...")`` because that path
    expects models from Coral's internal S3 mirror. OSS users get them
    straight from HuggingFace.
    """

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
    """Deterministic 64-dim fallback. Useless for retrieval quality;
    exists so the SQL parts of this demo run without any model download."""

    _DIM = 64

    def embed_texts(self, texts: list[str]) -> tuple[list[list[float]], dict]:
        out: list[list[float]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            # Stretch 32 bytes → 64 floats in [-1, 1] via 16-bit shorts.
            shorts = struct.unpack(">32h", digest + digest)
            vec = [s / 32768.0 for s in shorts][: self._DIM]
            # L2-normalise so cosine math behaves.
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
            f"[embedder] using sentence-transformers: {emb.get_model_name()}  "
            f"dim={emb.get_dimension()}"
        )
        return emb
    except Exception as exc:
        print(
            f"[embedder] sentence-transformers unavailable ({exc.__class__.__name__}); "
            "falling back to hash embedder (search results will be meaningless)."
        )
        return _HashEmbedder()


# ---------------------------------------------------------------------------
# 2. Run the prep pipeline → land vectors + graph as parquet files.
# ---------------------------------------------------------------------------


def _prep(work_dir: Path, embedder: BaseEmbedder) -> tuple[Path, Path, Path, Path]:
    chunks = chunk(NEWS, strategy="sliding_token", target_tokens=48, overlap=8)
    vectors = embed(chunks, embedder=embedder, output_dir=str(work_dir))
    enriched = enrich(NEWS, extractors=["tickers"])
    g = hydrate(enriched, graph="news", output_dir=str(work_dir))

    vec_path = Path(vectors.metadata["parquet_path"])
    nodes_path = work_dir / "nodes.parquet"
    edges_path = work_dir / "edges.parquet"

    print(f"[prep] {chunks.record_count} chunks across {chunks.metadata['doc_count']} docs")
    print(f"[prep] {vectors.record_count} vectors @ dim {vectors.metadata['dimension']}")
    print(f"[prep] graph: {g.metadata['node_count']} nodes / {g.metadata['edge_count']} edges")
    print(f"[prep] wrote: {vec_path.name}, {nodes_path.name}, {edges_path.name} → {work_dir}")
    return vec_path, nodes_path, edges_path, work_dir


# ---------------------------------------------------------------------------
# 3. DuckDB session — load parquet, build HNSW, build property graph.
# ---------------------------------------------------------------------------


def _open_duckdb(vec_path: Path, nodes_path: Path, edges_path: Path, dim: int) -> Any:
    import duckdb

    con = duckdb.connect()

    # Vectors: cast the fixed-size list column to FLOAT[dim] so VSS can index it.
    con.execute(f"""
        CREATE TABLE vectors AS
        SELECT
            doc_id,
            chunk_index,
            text,
            CAST(vector AS FLOAT[{dim}]) AS vector
        FROM read_parquet('{vec_path}')
    """)
    con.execute("CREATE TABLE nodes AS SELECT * FROM read_parquet(?)", [str(nodes_path)])
    con.execute("CREATE TABLE edges AS SELECT * FROM read_parquet(?)", [str(edges_path)])

    # VSS: HNSW index for cosine distance. Built in to DuckDB; auto-installs.
    con.execute("INSTALL vss; LOAD vss;")
    con.execute("CREATE INDEX vec_hnsw ON vectors USING HNSW (vector) WITH (metric = 'cosine')")

    return con


def _load_duckpgq(con: Any) -> None:
    """Install + load the DuckPGQ community extension and define the graph.

    Hard requirement: this demo is *about* doing graph queries inside
    DuckDB. If DuckPGQ won't load, fail loudly rather than silently
    degrading to another engine and dragging in extra deps.
    """
    try:
        con.execute("INSTALL duckpgq FROM community; LOAD duckpgq;")
    except Exception as exc:
        raise RuntimeError(
            "Failed to install/load the duckpgq community extension. "
            "This demo requires it for variable-length graph traversal. "
            "Check your DuckDB version's compatibility at "
            "https://duckdb.org/community_extensions/extensions/duckpgq "
            f"(underlying error: {exc.__class__.__name__}: {exc})"
        ) from exc

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


# ---------------------------------------------------------------------------
# 4. Three queries: pure vector, pure graph, hybrid.
# ---------------------------------------------------------------------------


def _vector_topk(con: Any, query_vec: list[float], k: int = 4) -> list[tuple]:
    dim = len(query_vec)
    rows = con.execute(
        f"""
        SELECT doc_id, text, array_cosine_distance(vector, ?::FLOAT[{dim}]) AS d
        FROM vectors
        ORDER BY d
        LIMIT {k}
        """,
        [query_vec],
    ).fetchall()
    return rows


def _graph_neighbors(con: Any, seed_ids: list[str], max_hops: int = 3) -> list[tuple]:
    """Variable-length traversal over DuckPGQ (SQL/PGQ)."""
    if not seed_ids:
        return []
    seed_sql = ", ".join(f"'{s}'" for s in seed_ids)
    return con.execute(f"""
        FROM GRAPH_TABLE (news_graph
          MATCH p = ANY SHORTEST
            (a:nodes)-[r:related]->{{1,{max_hops}}}(b:nodes)
          WHERE a.id IN ({seed_sql})
          COLUMNS (a.id AS seed, b.id AS reached, path_length(p) AS hops)
        )
        ORDER BY seed, hops, reached
    """).fetchall()


def _seed_entities_from_hits(con: Any, doc_ids: list[str]) -> list[str]:
    """Find entity nodes mentioned by the documents that vector search returned.

    Uses the existing 'mentions' edges (Document:<doc_id> -mentions-> Entity:<value>)
    that hydrate() already wrote into edges.parquet — no schema additions needed.
    """
    if not doc_ids:
        return []
    placeholders = ", ".join(["?"] * len(doc_ids))
    rows = con.execute(
        f"""
        SELECT DISTINCT e.dst AS entity_id
        FROM edges e
        WHERE e.relation = 'mentions'
          AND REPLACE(e.src, 'Document:', '') IN ({placeholders})
        """,
        doc_ids,
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# 5. Print helpers.
# ---------------------------------------------------------------------------


def _hr(title: str) -> None:
    print()
    print("─" * 72)
    print(title)
    print("─" * 72)


def _print_vector_hits(rows: list[tuple]) -> None:
    for doc_id, text, dist in rows:
        snippet = (text[:64] + "…") if len(text) > 64 else text
        print(f"  d={dist:.4f}  doc={doc_id:<4}  {snippet!r}")


def _print_graph_rows(rows: list[tuple]) -> None:
    if not rows:
        print("  (no paths)")
        return
    for seed, reached, hops in rows:
        print(f"  {seed} ─{hops}→ {reached}")


# ---------------------------------------------------------------------------
# 6. Main.
# ---------------------------------------------------------------------------


def main() -> None:
    embedder = _build_embedder()

    work_dir = Path(tempfile.mkdtemp(prefix="cb_embedded_rag_"))
    try:
        vec_path, nodes_path, edges_path, _ = _prep(work_dir, embedder)

        con = _open_duckdb(vec_path, nodes_path, edges_path, dim=embedder.get_dimension())
        _load_duckpgq(con)

        # --- 1. Pure vector search -----------------------------------------
        _hr(f"1. Vector top-K  query={QUERY!r}")
        query_vec, _ = embedder.embed_texts([QUERY])
        hits = _vector_topk(con, query_vec[0], k=4)
        _print_vector_hits(hits)

        # --- 2. Pure graph: 1..3-hop expansion from a known seed -----------
        # Note: node ids are "<Label>:<value>"; the label comes from the
        # extractor bucket ("tickers" → "Tickers"). See
        # CooccurrenceExtractor in coralbricks.context_prep.graph.triples.
        seed = "Tickers:AAPL"
        _hr(f"2. Variable-length graph traversal  seed={seed}  hops=1..3")
        rows = _graph_neighbors(con, [seed], max_hops=3)
        _print_graph_rows(rows)

        # --- 3. Hybrid: vector hits → seed entities → graph expansion ------
        _hr("3. Hybrid:  vector top-K → mentioned entities → 1..2-hop expansion")
        seed_doc_ids = [doc_id for doc_id, _, _ in hits]
        seeds = _seed_entities_from_hits(con, seed_doc_ids)
        print(f"  vector hits   : {seed_doc_ids}")
        print(f"  seed entities : {seeds}")
        rows = _graph_neighbors(con, seeds, max_hops=2)
        _print_graph_rows(rows)

        print()
        print("Done. Parquet artifacts kept at:", work_dir)
        print("(They will be deleted by the cleanup at the end of this script —")
        print(" remove the shutil.rmtree call below to keep them around.)")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
