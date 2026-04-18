"""Show the distributed-hydration pattern: build per-shard graphs, then reduce.

In a real job each ``hydrate_graph(shard, ...)`` call would happen in a
Ray actor / Spark executor / Prefect task. Here we just use the
multiprocessing pool for illustration.

Run:

    pip install coralbricks-context-prep
    python examples/distributed_hydrate.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any

from coralbricks.context_prep.enrichers import TickerExtractor, enrich_documents
from coralbricks.context_prep.graph import (
    CooccurrenceExtractor,
    hydrate_graph,
    merge_graphs,
)


def hydrate_shard(records: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = enrich_documents(records, [TickerExtractor()])
    return hydrate_graph(enriched, [CooccurrenceExtractor({"tickers": "Ticker"})])


SHARDS = [
    [
        {"id": "n1", "text": "$AAPL and $MSFT both rallied"},
        {"id": "n2", "text": "$AAPL beat earnings"},
    ],
    [
        {"id": "n3", "text": "$AAPL and $GOOGL up"},
        {"id": "n4", "text": "$NVDA, $AAPL, and $MSFT hit highs"},
    ],
]


def main() -> None:
    with ProcessPoolExecutor() as pool:
        partials = list(pool.map(hydrate_shard, SHARDS))

    print("Per-shard sizes:")
    for i, p in enumerate(partials):
        print(f"  shard {i}: {p['node_count']} nodes / {p['edge_count']} edges")

    full = merge_graphs(*partials)
    print(f"\nmerged: {full['node_count']} nodes / {full['edge_count']} edges")

    aapl_mentions = [
        e for e in full["edges"] if e["dst"] == "Ticker:AAPL" and e["relation"] == "mentions"
    ]
    print(f"AAPL mention edges (deduped): {len(aapl_mentions)}")
    print(
        "Total AAPL mention weight (summed across shards): "
        f"{sum(e['weight'] for e in aapl_mentions)}"
    )


if __name__ == "__main__":
    main()
