"""Tests for coralbricks.context_prep.joiners and context_prep.graph."""

from __future__ import annotations

from coralbricks.context_prep.enrichers import TickerExtractor, enrich_documents
from coralbricks.context_prep.graph import (
    CooccurrenceExtractor,
    EntityCooccurrenceExtractor,
    hydrate_graph,
    merge_graphs,
)
from coralbricks.context_prep.joiners import join_records


def test_join_inner() -> None:
    left = [{"k": 1, "a": "x"}, {"k": 2, "a": "y"}]
    right = [{"k": 2, "b": "Y"}, {"k": 3, "b": "Z"}]
    joined = join_records(left, right, on="k", how="inner")
    assert joined == [{"k": 2, "a": "y", "b": "Y"}]


def test_join_left_keeps_unmatched() -> None:
    left = [{"k": 1, "a": "x"}, {"k": 2, "a": "y"}]
    right = [{"k": 2, "b": "Y"}]
    joined = join_records(left, right, on="k", how="left")
    assert {r["k"] for r in joined} == {1, 2}
    assert next(r for r in joined if r["k"] == 1).get("b") is None


def test_join_outer_includes_both_sides() -> None:
    left = [{"k": 1}]
    right = [{"k": 2}]
    joined = join_records(left, right, on="k", how="outer")
    assert {r["k"] for r in joined} == {1, 2}


def test_join_with_callable_key() -> None:
    left = [{"id": "AAPL"}, {"id": "msft"}]
    right = [{"sym": "aapl", "px": 100}]
    joined = join_records(
        left,
        right,
        on=lambda r: (r.get("id") or "").lower(),
        right_on=lambda r: (r.get("sym") or "").lower(),
        how="inner",
    )
    assert joined == [{"id": "AAPL", "sym": "aapl", "px": 100}]


def test_hydrate_graph_dedupes_nodes_and_accumulates_weight() -> None:
    docs = [
        {"id": "d1", "text": "$AAPL up"},
        {"id": "d2", "text": "$AAPL down"},
    ]
    enriched = enrich_documents(docs, [TickerExtractor()])
    graph = hydrate_graph(
        enriched,
        [CooccurrenceExtractor({"tickers": "Ticker"})],
    )
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    assert "Ticker:AAPL" in nodes_by_id
    edges_by_key = {(e["src"], e["dst"], e["relation"]): e for e in graph["edges"]}
    aapl_edges = [
        e
        for (src, dst, rel), e in edges_by_key.items()
        if dst == "Ticker:AAPL" and rel == "mentions"
    ]
    assert len(aapl_edges) == 2


def test_entity_cooccurrence_inline_extractor() -> None:
    docs = [{"id": "d1", "text": "$AAPL and $MSFT both popped"}]
    graph = hydrate_graph(docs, [EntityCooccurrenceExtractor([TickerExtractor()])])
    assert graph["node_count"] >= 3
    assert any(e["relation"] == "co_occurs_with" for e in graph["edges"])


# ---------------------------------------------------------------------------
# merge_graphs: distributed-hydration reduce step
# ---------------------------------------------------------------------------


def _hydrate_shard(records):
    enriched = enrich_documents(records, [TickerExtractor()])
    return hydrate_graph(enriched, [CooccurrenceExtractor({"tickers": "Ticker"})])


def test_merge_graphs_handles_empty() -> None:
    assert merge_graphs() == {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}


def test_merge_graphs_dedupes_nodes() -> None:
    a = _hydrate_shard([{"id": "d1", "text": "$AAPL up"}])
    b = _hydrate_shard([{"id": "d2", "text": "$AAPL down"}])
    merged = merge_graphs(a, b)
    node_ids = {n["id"] for n in merged["nodes"]}
    assert "Ticker:AAPL" in node_ids
    aapl_count = sum(1 for n in merged["nodes"] if n["id"] == "Ticker:AAPL")
    assert aapl_count == 1


def test_merge_graphs_sums_edge_weights() -> None:
    a = _hydrate_shard([{"id": "d1", "text": "$AAPL"}])
    b = _hydrate_shard([{"id": "d1", "text": "$AAPL"}])
    merged = merge_graphs(a, b)
    aapl_edges = [
        e for e in merged["edges"] if e["dst"] == "Ticker:AAPL" and e["relation"] == "mentions"
    ]
    assert len(aapl_edges) == 1
    assert aapl_edges[0]["weight"] == 2.0


def test_merge_graphs_combines_disjoint_shards() -> None:
    a = _hydrate_shard([{"id": "d1", "text": "$AAPL only"}])
    b = _hydrate_shard([{"id": "d2", "text": "$MSFT only"}])
    merged = merge_graphs(a, b)
    assert merged["node_count"] == a["node_count"] + b["node_count"]
    assert merged["edge_count"] == a["edge_count"] + b["edge_count"]
