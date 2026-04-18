"""Unit tests for the coralbricks.context_prep DSL.

Covers both:
- Dry-run shape: each verb returns an Artifact with the right kind and
  metadata, even without doing real work. Used to compose recipes.
- Live execution: each verb actually produces data on concrete record
  inputs (list[dict] / list[str] / single string).

Live trafilatura, embedder backends, and pyarrow installation are
covered in their own targeted tests.
"""

from __future__ import annotations

from coralbricks.context_prep import (
    Artifact,
    ArtifactKind,
    Recipe,
    chunk,
    clean,
    embed,
    enrich,
    hydrate,
    join,
)


# ---------------------------------------------------------------------------
# Dry-run shape
# ---------------------------------------------------------------------------


def test_chunk_dry_run_links_inputs() -> None:
    upstream = clean(["<html>x</html>"], dry_run=True)
    chunks = chunk(upstream, target_tokens=256, overlap=32)
    assert chunks.kind == ArtifactKind.CHUNKS
    assert chunks.inputs == (upstream.artifact_id,)
    assert chunks.metadata["target_tokens"] == 256


def test_full_pipeline_dry_run_chains_correctly() -> None:
    cleaned = clean(["<html>x</html>"], dry_run=True)
    chunks = chunk(cleaned, strategy="semantic")
    vecs = embed(chunks, model="bge-m3")
    enriched = enrich(cleaned, extractors=["urls", "tickers"])
    g = hydrate(enriched, graph="graph_combined")

    assert vecs.kind == ArtifactKind.VECTORS
    assert vecs.inputs == (chunks.artifact_id,)
    assert enriched.inputs == (cleaned.artifact_id,)
    assert g.kind == ArtifactKind.GRAPH
    assert g.inputs == (enriched.artifact_id,)


def test_recipe_collects_calls_and_plans() -> None:
    r = Recipe("sec_filings_2026q1")
    cleaned = clean(["<html>x</html>"], dry_run=True)
    r.add(clean, ["<html>x</html>"], dry_run=True)
    r.add(chunk, cleaned, strategy="semantic", target_tokens=512, dry_run=True)
    r.add(embed, cleaned, model="bge-m3", dry_run=True)

    assert len(r.calls) == 3
    assert r.calls[0].name == "clean"
    assert r.calls[1].name == "chunk"
    assert r.calls[2].name == "embed"


def test_recipe_run_executes_in_order() -> None:
    seen: list[str] = []

    def fake_verb(name: str) -> Artifact:
        seen.append(name)
        return Artifact(artifact_id=name, kind=ArtifactKind.DOCS, produced_by=name)

    r = Recipe("test")
    r.add(fake_verb, "first")
    r.add(fake_verb, "second")
    r.add(fake_verb, "third")
    results = r.run()

    assert seen == ["first", "second", "third"]
    assert [a.artifact_id for a in results] == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# Live verbs over records
# ---------------------------------------------------------------------------


def test_chunk_live_on_records() -> None:
    art = chunk(
        [
            {"id": "d1", "text": "a " * 200},
            {"id": "d2", "text": "b " * 50},
        ],
        strategy="fixed_token",
        target_tokens=20,
    )
    assert art.kind == ArtifactKind.CHUNKS
    assert art.metadata["doc_count"] == 2
    assert art.record_count > 1


def test_chunk_live_on_string() -> None:
    art = chunk("a " * 200, strategy="fixed_token", target_tokens=20)
    assert art.record_count > 1
    assert isinstance(art.metadata["chunks"], list)


def test_enrich_live_extractors_default() -> None:
    art = enrich(["see https://example.com about $AAPL on 2026-04-18"])
    docs = art.metadata["documents"]
    assert len(docs) == 1
    extractions = docs[0]["metadata"]["extractions"]
    assert any(r["value"] == "https://example.com" for r in extractions["urls"])
    assert any(r["value"] == "AAPL" for r in extractions["tickers"])
    assert any("2026" in r["value"] for r in extractions["dates"])


def test_enrich_then_chunk_chain() -> None:
    enriched = enrich(["a long bit of text $AAPL " * 80])
    chunks_art = chunk(enriched, strategy="fixed_token", target_tokens=40)
    assert chunks_art.inputs == (enriched.artifact_id,)
    assert chunks_art.record_count >= 1


def test_join_live_dict_records() -> None:
    left = [{"id": 1, "text": "alpha"}, {"id": 2, "text": "beta"}]
    right = [{"id": 1, "ticker": "AAPL"}, {"id": 3, "ticker": "MSFT"}]
    art = join(left, right, on="id", how="left")
    assert art.kind == ArtifactKind.JOINED
    rows = art.metadata["records"]
    assert len(rows) == 2
    by_id = {r["id"]: r for r in rows}
    assert by_id[1]["ticker"] == "AAPL"
    assert "ticker" not in by_id[2]


def test_hydrate_live_from_enrich() -> None:
    enriched = enrich(
        [
            {"id": "d1", "text": "$AAPL is up; see https://example.com"},
            {"id": "d2", "text": "$AAPL and $MSFT both rallied"},
        ],
        extractors=["tickers"],
    )
    g = hydrate(enriched, graph="cb_test")
    assert g.kind == ArtifactKind.GRAPH
    assert g.metadata["node_count"] >= 3
    assert g.metadata["edge_count"] >= 2


def test_clean_dry_run() -> None:
    art = clean(["<html>x</html>"], dry_run=True)
    assert art.kind == ArtifactKind.CLEANED
    assert art.metadata["cleaner"] == "trafilatura"
