"""Shared types for multi-source retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    """A single retrieval result from any backend."""

    id: str
    text: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("SearchHit.id must be non-empty")


@dataclass
class RetrievalResult:
    """Aggregated output from multi-source retrieval."""

    hits: list[SearchHit]
    sources_queried: list[str]
    fusion_strategy: str | None = None

    @property
    def top(self) -> SearchHit | None:
        return self.hits[0] if self.hits else None

    def texts(self, limit: int | None = None) -> list[str]:
        """Return hit texts, optionally capped at *limit*."""
        subset = self.hits[:limit] if limit else self.hits
        return [h.text for h in subset]
