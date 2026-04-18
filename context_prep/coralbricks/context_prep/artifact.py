"""Artifact = the typed output of a single prep verb.

Verbs produce artifacts that subsequent verbs consume. An ``Artifact``
is intentionally lightweight: it describes *what* was produced
(records, vectors, a graph slice) and *where* it landed, plus an
optional ``metadata`` payload carrying the actual data when the verb
ran in-process.

Artifacts are immutable handles; the materialised payload (if any)
lives under :attr:`metadata`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ArtifactKind(str, Enum):
    DOCS = "docs"  # raw normalized records (input shape)
    CHUNKS = "chunks"  # chunk() output: chunked records
    VECTORS = "vectors"  # embed() output: float vectors
    ENRICHED = "enriched"  # enrich() output: chunks + extracted facts
    GRAPH = "graph"  # hydrate() output: nodes/edges
    CLEANED = "cleaned"  # clean() output: cleaned records
    JOINED = "joined"  # join() output: joined records


@dataclass(frozen=True)
class Artifact:
    """Handle to one prep stage's output.

    ``uri`` is the canonical location once materialised (``s3://...``,
    local path, in-memory key). ``record_count`` is best-effort; verbs
    that don't know yet may leave it ``None``. ``produced_by`` is the
    verb name; ``inputs`` is the list of upstream Artifact ids that fed
    it (for lineage).
    """

    artifact_id: str
    kind: ArtifactKind
    uri: str | None = None
    record_count: int | None = None
    schema_hint: dict[str, Any] = field(default_factory=dict)
    produced_by: str = ""
    inputs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
