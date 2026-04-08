"""Backend protocol — the contract every retrieval source must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import SearchHit


@runtime_checkable
class RetrievalBackend(Protocol):
    """Minimal interface for a retrieval data source.

    Implement ``name`` and ``search`` to plug any backend into the
    :class:`~coral_retrieval.MultiSourceRetriever`.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this backend (e.g. ``"vector"``, ``"graph"``)."""
        ...

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        """Return up to *top_k* hits ranked by relevance."""
        ...
