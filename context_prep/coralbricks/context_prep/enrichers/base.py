"""Extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionResult:
    """A single extracted value with its span and optional metadata."""

    value: str
    start: int
    end: int
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"value": self.value, "start": self.start, "end": self.end}
        if self.label:
            d["label"] = self.label
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d


class BaseExtractor(ABC):
    """Subclasses extract a list of :class:`ExtractionResult` from text."""

    name: str = "base"

    @abstractmethod
    def extract(self, text: str) -> list[ExtractionResult]: ...

    def extract_many(self, texts: Iterable[str]) -> list[list[ExtractionResult]]:
        return [self.extract(t) for t in texts]
