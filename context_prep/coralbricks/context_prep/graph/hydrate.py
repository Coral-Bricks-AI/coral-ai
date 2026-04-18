"""Aggregate triples into a deduplicated nodes + edges graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence, Union

from .._records import normalize_records
from .triples import (
    BaseTripleExtractor,
    Edge,
    Node,
    Triple,
)


def _aggregate(triples: Iterable[Triple]) -> tuple[list[Node], list[Edge]]:
    nodes: dict[str, Node] = {}
    edge_acc: dict[tuple[str, str, str], Edge] = {}

    for t in triples:
        src = t.src
        dst = t.dst
        if not src or not dst:
            continue
        if src not in nodes:
            nodes[src] = Node(id=src, label=t.subject_label, value=t.subject_value)
        if dst not in nodes:
            nodes[dst] = Node(id=dst, label=t.object_label, value=t.object_value)

        key = (src, dst, t.predicate)
        if key in edge_acc:
            existing = edge_acc[key]
            existing.weight += t.weight
        else:
            edge_acc[key] = Edge(src=src, dst=dst, relation=t.predicate, weight=t.weight)

    return list(nodes.values()), list(edge_acc.values())


def hydrate_graph(
    docs: Union[dict, Iterable[Any]],
    extractors: Sequence[BaseTripleExtractor],
    *,
    output_dir: Union[str, Path, None] = None,
) -> dict[str, Any]:
    """Run triple extractors over records and return a {nodes, edges} graph.

    Args:
        docs: Iterable of dict records (or a single dict).
        extractors: One or more :class:`BaseTripleExtractor`.
        output_dir: When given, write ``nodes.parquet`` + ``edges.parquet``
            to this **local** directory (requires pyarrow). For S3/GCS,
            write to a scratch dir first, then upload (e.g. ``aws s3 cp
            --recursive``). Direct ``s3://`` / ``gs://`` URIs are
            planned for 0.2.0.

    Returns:
        ``{"nodes": [...], "edges": [...], "node_count": N, "edge_count": M}``
        where each node / edge is a plain dict.
    """
    if isinstance(docs, dict):
        records = normalize_records([docs])
    else:
        records = normalize_records(list(docs))

    triples: list[Triple] = []
    for rec in records:
        for extractor in extractors:
            triples.extend(extractor.extract(rec))

    nodes, edges = _aggregate(triples)
    result: dict[str, Any] = {
        "nodes": [
            {"id": n.id, "label": n.label, "value": n.value, "metadata": n.metadata}
            for n in nodes
        ],
        "edges": [
            {
                "src": e.src,
                "dst": e.dst,
                "relation": e.relation,
                "weight": e.weight,
                "metadata": e.metadata,
            }
            for e in edges
        ],
        "node_count": len(nodes),
        "edge_count": len(edges),
    }

    if output_dir:
        write_graph_parquet(result, output_dir)
        result["output_dir"] = str(output_dir)

    return result


def merge_graphs(*graphs: dict) -> dict:
    """Combine partial graphs from distributed hydration into one graph.

    Each input is a graph dict shaped like the output of
    :func:`hydrate_graph` (``{"nodes": [...], "edges": [...], ...}``).
    Nodes are deduplicated by id (last writer wins for ``label``/``value``
    metadata; node metadata is shallow-merged). Edges are keyed by
    ``(src, dst, relation)`` and their ``weight`` values are summed.

    The intended pattern is::

        # in each Ray / Spark / Prefect worker:
        partial = hydrate_graph(shard, extractors=...)

        # in the reducer:
        full = merge_graphs(*partials)
    """
    if not graphs:
        return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}

    nodes: dict[str, dict] = {}
    edge_acc: dict[tuple[str, str, str], dict] = {}

    for g in graphs:
        for n in g.get("nodes", []) or []:
            nid = n.get("id")
            if not nid:
                continue
            existing = nodes.get(nid)
            if existing is None:
                nodes[nid] = {
                    "id": nid,
                    "label": n.get("label"),
                    "value": n.get("value"),
                    "metadata": dict(n.get("metadata") or {}),
                }
            else:
                if n.get("label") and not existing.get("label"):
                    existing["label"] = n["label"]
                if n.get("value") and not existing.get("value"):
                    existing["value"] = n["value"]
                existing["metadata"].update(dict(n.get("metadata") or {}))

        for e in g.get("edges", []) or []:
            src = e.get("src")
            dst = e.get("dst")
            rel = e.get("relation")
            if not (src and dst and rel):
                continue
            key = (src, dst, rel)
            weight = float(e.get("weight", 1.0) or 0.0)
            if key in edge_acc:
                existing_e = edge_acc[key]
                existing_e["weight"] = float(existing_e.get("weight", 0.0)) + weight
                existing_e["metadata"].update(dict(e.get("metadata") or {}))
            else:
                edge_acc[key] = {
                    "src": src,
                    "dst": dst,
                    "relation": rel,
                    "weight": weight,
                    "metadata": dict(e.get("metadata") or {}),
                }

    return {
        "nodes": list(nodes.values()),
        "edges": list(edge_acc.values()),
        "node_count": len(nodes),
        "edge_count": len(edge_acc),
    }


def write_graph_parquet(graph: dict, output_dir: Union[str, Path]) -> dict[str, str]:
    """Persist nodes + edges to ``<output_dir>/nodes.parquet`` and ``edges.parquet``.

    ``output_dir`` is a **local** filesystem path. To land in S3 / GCS,
    write to a scratch dir first then upload — e.g. ``aws s3 cp
    --recursive <output_dir> s3://bucket/prefix/``. Native object-store
    URIs are planned for 0.2.0.
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "write_graph_parquet requires pyarrow; install with `pip install pyarrow` "
            "(or `pip install 'coralbricks-context-prep[graph]'`)."
        ) from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / "nodes.parquet"
    edges_path = out_dir / "edges.parquet"

    nodes_table = pa.Table.from_pylist(
        [
            {"id": n["id"], "label": n["label"], "value": n["value"]}
            for n in graph.get("nodes", [])
        ]
    )
    edges_table = pa.Table.from_pylist(
        [
            {
                "src": e["src"],
                "dst": e["dst"],
                "relation": e["relation"],
                "weight": float(e.get("weight", 1.0)),
            }
            for e in graph.get("edges", [])
        ]
    )
    pq.write_table(nodes_table, nodes_path)
    pq.write_table(edges_table, edges_path)
    return {"nodes": str(nodes_path), "edges": str(edges_path)}
