"""Tests for the embedder factory.

These don't actually call any embedding API. They check:
- The factory module is importable without optional API SDKs.
- ``coralbricks.context_prep.embedders`` lazy-imports heavy backends only when accessed.
- ``prep.embed()`` can run end-to-end against a fake embedder.
"""

from __future__ import annotations

import pytest

from coralbricks.context_prep import embed
from coralbricks.context_prep.embedders import (
    BaseEmbedder,
    create_embedder,
    list_supported_models,
)


def test_factory_rejects_unknown_model():
    with pytest.raises(ValueError):
        create_embedder("totally-fake-model", dimension=128)


def test_list_supported_models_runs():
    try:
        models = list_supported_models()
    except (ImportError, ModuleNotFoundError):
        pytest.skip("optional embedder backend not installed")
    assert "openai" in models
    assert "bedrock" in models


class _FakeEmbedder(BaseEmbedder):
    def __init__(self, dim: int = 4):
        self._dim = dim

    def embed_texts(self, texts):
        return [[0.1 * i] * self._dim for i, _ in enumerate(texts)], {
            "prompt_tokens": 0,
            "total_tokens": 0,
        }

    def get_model_name(self):
        return "fake"

    def get_dimension(self):
        return self._dim


def test_prep_embed_uses_provided_embedder():
    fake = _FakeEmbedder(dim=8)
    art = embed(["alpha", "beta", "gamma"], embedder=fake)
    assert art.metadata["model"] == "fake"
    assert art.metadata["dimension"] == 8
    assert art.record_count == 3
    assert len(art.metadata["vectors"]) == 3
    assert all(len(v) == 8 for v in art.metadata["vectors"])


def test_prep_embed_dry_run_with_string_input():
    art = embed("hello", model="coral_embed", dry_run=True)
    assert "vectors" not in art.metadata
    assert art.metadata["model"] == "coral_embed"


def test_prep_chunk_then_embed_pipes_text():
    from coralbricks.context_prep import chunk

    text = "Coral Bricks is a memory layer. " * 40
    chunks_art = chunk(text, strategy="sliding_token", target_tokens=32, overlap=4)
    materialised = chunks_art.metadata["chunks"]
    assert materialised
    fake = _FakeEmbedder(dim=4)
    vec_art = embed(materialised, embedder=fake)
    assert vec_art.record_count == len(materialised)


def test_prep_embed_writes_vectors_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    from coralbricks.context_prep import chunk

    text = "Coral Bricks is a memory layer. " * 20
    chunks_art = chunk(text, strategy="sliding_token", target_tokens=32, overlap=4)
    fake = _FakeEmbedder(dim=4)
    vec_art = embed(chunks_art, embedder=fake, output_dir=str(tmp_path))

    parquet_path = vec_art.metadata["parquet_path"]
    assert parquet_path is not None
    assert vec_art.uri == parquet_path
    assert (tmp_path / "vectors.parquet").exists()

    table = pq.read_table(parquet_path)
    assert table.num_rows == vec_art.record_count
    assert {"doc_id", "chunk_index", "text", "vector"}.issubset(set(table.schema.names))

    vector_field = table.schema.field("vector")
    assert pa.types.is_fixed_size_list(vector_field.type)
    assert vector_field.type.list_size == 4

    md = table.schema.metadata or {}
    assert md.get(b"model") == b"fake"
    assert md.get(b"dimension") == b"4"
    assert md.get(b"count") == str(vec_art.record_count).encode()
    assert md.get(b"batch_size") == b"128"

    rows = table.to_pylist()
    assert all(r["text"] for r in rows)
    assert all(len(r["vector"]) == 4 for r in rows)
    assert all(r["doc_id"] is not None for r in rows)


def test_prep_embed_writes_parquet_for_plain_strings(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")

    fake = _FakeEmbedder(dim=4)
    vec_art = embed(["alpha", "beta"], embedder=fake, output_dir=str(tmp_path))

    table = pq.read_table(vec_art.metadata["parquet_path"])
    rows = table.to_pylist()
    assert [r["text"] for r in rows] == ["alpha", "beta"]
    assert all(r["doc_id"] is None for r in rows)
    assert all(r["chunk_index"] is None for r in rows)


def test_prep_embed_rejects_stray_kwargs():
    """A5: embed() no longer silently drops unknown kwargs."""
    with pytest.raises(TypeError):
        embed(["alpha"], embedder=_FakeEmbedder(), opnai_api_key="oops")  # type: ignore[call-arg]


def test_prep_embed_forwards_embedder_kwargs(monkeypatch):
    """embedder_kwargs={...} flows verbatim to create_embedder."""
    seen: dict = {}

    def fake_create_embedder(**kwargs):
        seen.update(kwargs)
        return _FakeEmbedder(dim=4)

    import coralbricks.context_prep.embedders as emb_pkg

    monkeypatch.setattr(emb_pkg, "create_embedder", fake_create_embedder)

    embed(
        ["alpha"],
        model="openai:text-embedding-3-large",
        embedder_kwargs={"openai_api_key": "sk-test", "device": "cpu"},
    )
    assert seen["openai_api_key"] == "sk-test"
    assert seen["device"] == "cpu"
    assert seen["model_id"] == "openai:text-embedding-3-large"


def test_write_vectors_parquet_validates_lengths(tmp_path):
    pytest.importorskip("pyarrow")
    from coralbricks.context_prep.embedders import write_vectors_parquet

    with pytest.raises(ValueError):
        write_vectors_parquet(
            ["a", "b"], [[0.1, 0.2, 0.3]], tmp_path, model="fake", dimension=3
        )
    with pytest.raises(ValueError):
        write_vectors_parquet(
            ["a"], [[0.1, 0.2]], tmp_path, model="fake", dimension=3
        )
