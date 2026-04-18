"""End-to-end RAG prep over a few in-memory records.

Run:

    pip install 'coralbricks-context-prep[chunkers]'
    python examples/rag_quickstart.py

This example does NOT call any embedding API; it uses a fake embedder
so it stays runnable in CI. Swap in a real model with
``embed(..., model="openai:text-embedding-3-large")`` etc.

If your input is HTML, prepend ``clean(records)`` (requires
``coralbricks-context-prep[cleaners]``).
"""

from __future__ import annotations

from context_prep import chunk, embed
from context_prep.embedders import BaseEmbedder


class _FakeEmbedder(BaseEmbedder):
    """Deterministic 8-dim "embedding" for offline demos / CI."""

    def embed_texts(self, texts):
        return [[float(i % 7) / 7.0] * 8 for i, _ in enumerate(texts)], {}

    def get_model_name(self) -> str:
        return "fake-embedder"

    def get_dimension(self) -> int:
        return 8


SAMPLE_RECORDS = [
    {
        "id": "post-1",
        "text": (
            "Coral Bricks is the memory layer for agentic AI. "
            "It includes a context preparation library, a sandbox runtime, "
            "and a GPU-native engine. The OSS prep library is intentionally "
            "small: clean, chunk, embed, enrich, hydrate. Records are dicts."
        ),
    },
    {
        "id": "post-2",
        "text": (
            "The universal record shape is dict. We do not ship a Document "
            "class because every caller already has dicts in pandas, "
            "duckdb, or their queue. normalize_records() does the minimal "
            "shape coercion in one place so verbs stay simple and easy "
            "to test."
        ),
    },
]


def main() -> None:
    chunks = chunk(
        SAMPLE_RECORDS,
        strategy="sliding_token",
        target_tokens=24,
        overlap=4,
    )
    print(f"chunks:  {chunks.record_count} chunks across {chunks.metadata['doc_count']} docs")

    vectors = embed(chunks, embedder=_FakeEmbedder())
    print(f"vectors: {vectors.record_count} vectors of dim {vectors.metadata['dimension']}")

    # In a real app, push (chunk, vector) pairs into your vector store:
    print("\nFirst chunk (truncated):")
    first_chunk = chunks.metadata["chunks"][0][0]
    print(f"  doc_id={first_chunk['doc_id']}  text={first_chunk['text'][:80]!r}")
    print(f"  vector[:4]={vectors.metadata['vectors'][0][:4]}")

    # Optional: write vectors to a self-describing parquet file
    # (requires `pip install 'coralbricks-context-prep[graph]'` for pyarrow).
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            on_disk = embed(chunks, embedder=_FakeEmbedder(), output_dir=tmpdir)
            print(f"\nwrote {on_disk.metadata['parquet_path']}")
    except RuntimeError as exc:
        print(f"\n(skipped parquet sink: {exc})")


if __name__ == "__main__":
    main()
