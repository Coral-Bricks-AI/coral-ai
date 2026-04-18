"""Triple / Node / Edge primitives + co-occurrence extractor."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from ..enrichers.base import BaseExtractor


def _slug(value: str) -> str:
    return re.sub(r"\s+", "_", (value or "").strip())[:200]


def node_id(label: str, identifier: str) -> str:
    """Stable id used by both nodes and edge endpoints."""
    n = _slug(identifier)
    if not n:
        return ""
    return f"{label}:{n}"


@dataclass(frozen=True)
class Triple:
    """(subject, predicate, object) plus optional weight + provenance."""

    subject_label: str
    subject_value: str
    predicate: str
    object_label: str
    object_value: str
    weight: float = 1.0
    metadata: tuple[tuple[str, Any], ...] = ()

    @property
    def src(self) -> str:
        return node_id(self.subject_label, self.subject_value)

    @property
    def dst(self) -> str:
        return node_id(self.object_label, self.object_value)


@dataclass
class Node:
    id: str
    label: str
    value: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTripleExtractor(ABC):
    name: str = "base"

    @abstractmethod
    def extract(self, record: dict[str, Any]) -> list[Triple]: ...


class CooccurrenceExtractor(BaseTripleExtractor):
    """Connect document → entity, and entity ↔ entity for entities that
    co-occur within a single document. Useful when you have already
    enriched documents with named entities and want a starter graph.

    Args:
        sources: Mapping from extraction-bucket name (the keys produced
            by :func:`enrich_documents`) to graph node label. E.g.
            ``{"tickers": "Ticker", "spacy_entities": "Entity"}``.
        document_label: Label assigned to the document node itself.
        relation_doc_to_entity: Predicate for doc → entity edges.
        relation_entity_entity: Predicate for entity ↔ entity edges.
        include_cooccurrence: Emit entity-entity edges when True (default).
    """

    name = "cooccurrence"

    def __init__(
        self,
        sources: dict[str, str],
        *,
        document_label: str = "Document",
        relation_doc_to_entity: str = "mentions",
        relation_entity_entity: str = "co_occurs_with",
        include_cooccurrence: bool = True,
    ):
        self._sources = dict(sources)
        self._doc_label = document_label
        self._rel_doc = relation_doc_to_entity
        self._rel_pair = relation_entity_entity
        self._include_pairs = include_cooccurrence

    def extract(self, record: dict[str, Any]) -> list[Triple]:
        meta = record.get("metadata") or {}
        extractions = meta.get("extractions") or {}
        entities: list[tuple[str, str]] = []
        for bucket, label in self._sources.items():
            for hit in extractions.get(bucket, []):
                value = hit.get("value") if isinstance(hit, dict) else getattr(hit, "value", None)
                if value:
                    entities.append((label, str(value)))

        seen: set[tuple[str, str]] = set()
        unique_entities: list[tuple[str, str]] = []
        for ent in entities:
            if ent in seen:
                continue
            seen.add(ent)
            unique_entities.append(ent)

        triples: list[Triple] = []
        doc_id = str(record.get("id") or "")
        for label, value in unique_entities:
            triples.append(
                Triple(
                    subject_label=self._doc_label,
                    subject_value=doc_id,
                    predicate=self._rel_doc,
                    object_label=label,
                    object_value=value,
                )
            )

        if self._include_pairs and len(unique_entities) >= 2:
            for (la, va), (lb, vb) in combinations(unique_entities, 2):
                triples.append(
                    Triple(
                        subject_label=la,
                        subject_value=va,
                        predicate=self._rel_pair,
                        object_label=lb,
                        object_value=vb,
                    )
                )

        return triples


class EntityCooccurrenceExtractor(CooccurrenceExtractor):
    """Run a list of :class:`BaseExtractor` inline (no prior enrich step
    required) and treat their outputs as graph entities."""

    def __init__(
        self,
        extractors: Sequence[BaseExtractor],
        *,
        document_label: str = "Document",
        relation_doc_to_entity: str = "mentions",
        relation_entity_entity: str = "co_occurs_with",
    ):
        sources = {ex.name: ex.name.title() for ex in extractors}
        super().__init__(
            sources,
            document_label=document_label,
            relation_doc_to_entity=relation_doc_to_entity,
            relation_entity_entity=relation_entity_entity,
        )
        self._extractors = list(extractors)

    def extract(self, record: dict[str, Any]) -> list[Triple]:
        meta = record.setdefault("metadata", {})
        bucket = meta.setdefault("extractions", {})
        for ex in self._extractors:
            bucket.setdefault(ex.name, [r.to_dict() for r in ex.extract(record.get("text", ""))])
        return super().extract(record)
