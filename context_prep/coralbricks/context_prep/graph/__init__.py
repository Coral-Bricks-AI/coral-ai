"""Graph hydration: turn documents into nodes + edges suitable for a
graph index (e.g. cuGraph-loadable parquet).

Three-step model:

1. **Triple extraction** — pluggable per-document extractors emit
   ``Triple(subject, predicate, object)`` rows. A default
   :class:`CooccurrenceExtractor` builds edges from co-occurring entities.
2. **Hydration** — :func:`hydrate_graph` deduplicates triples into
   ``nodes`` (with stable label-prefixed ids) and ``edges`` (with
   ``src``, ``dst``, ``relation``, ``weight``). Optionally persists to
   parquet via pyarrow.
3. **Merge (distributed)** — :func:`merge_graphs` combines partial
   graphs from many workers (Ray/Spark/Prefect) into one graph,
   summing edge weights. This is the only step where graph hydration
   isn't embarrassingly parallel, so we own the merge primitive.
"""

from __future__ import annotations

from .triples import (
    BaseTripleExtractor,
    CooccurrenceExtractor,
    EntityCooccurrenceExtractor,
    Node,
    Edge,
    Triple,
)
from .hydrate import hydrate_graph, merge_graphs, write_graph_parquet

__all__ = [
    "Triple",
    "Node",
    "Edge",
    "BaseTripleExtractor",
    "CooccurrenceExtractor",
    "EntityCooccurrenceExtractor",
    "hydrate_graph",
    "merge_graphs",
    "write_graph_parquet",
]
