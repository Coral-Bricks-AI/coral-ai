"""Build a small entity co-occurrence graph from news-like records.

Run:

    pip install coralbricks-context-prep
    python examples/knowledge_graph.py
"""

from __future__ import annotations

from coralbricks.context_prep import enrich, hydrate

NEWS = [
    {"id": "n1", "text": "$AAPL and $MSFT both rallied on the Fed announcement."},
    {"id": "n2", "text": "$AAPL beat earnings; $GOOGL also up."},
    {"id": "n3", "text": "$NVDA hit a new high; $MSFT followed."},
]


def main() -> None:
    enriched = enrich(NEWS, extractors=["tickers"])
    g = hydrate(enriched, graph="news_cooccurrence")

    print(f"nodes: {g.metadata['node_count']}")
    print(f"edges: {g.metadata['edge_count']}")
    print()

    print("Top 5 entity nodes:")
    for n in g.metadata["nodes"][:5]:
        print(f"  - {n['id']}  label={n['label']}  value={n['value']}")

    print()
    print("Top 5 edges:")
    for e in g.metadata["edges"][:5]:
        print(
            f"  - {e['src']} --[{e['relation']} w={e['weight']}]--> {e['dst']}"
        )


if __name__ == "__main__":
    main()
