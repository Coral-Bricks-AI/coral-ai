"""CoralBricksMemory: high-level memory helper for LangChain applications.

Wraps CoralBricksClient with save, search, and delete convenience methods
so callers never need to touch the raw HTTP client directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .client import CoralBricksClient


class CoralBricksMemory:
    """High-level wrapper around :class:`CoralBricksClient`.

    All operations are automatically scoped to ``project_id`` and
    ``session_id`` when provided.

    Args:
        client: A configured :class:`CoralBricksClient` instance.
        project_id: Optional project namespace (shared across crews/agents).
        session_id: Optional session or user namespace.
    """

    def __init__(
        self,
        client: CoralBricksClient,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.client = client
        self.project_id = project_id
        self.session_id = session_id

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def save_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Embed *text* and persist it as a memory item.

        Returns the CoralBricks memory id assigned by the server.
        """
        item: Dict[str, Any] = {"text": text}
        if metadata:
            item["metadata"] = metadata
        saved = self.client.save(
            items=[item],
            project_id=self.project_id,
            session_id=self.session_id,
        )
        if not saved:
            raise RuntimeError("CoralBricks /v1/memory/save returned no items")
        mem_id = saved[0].get("id")
        if not isinstance(mem_id, str):
            raise RuntimeError("CoralBricks /v1/memory/save did not return an id")
        return mem_id

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search memory by natural-language *query*.

        Returns a list of hit dicts each containing at minimum
        ``id``, ``text``, ``metadata``, ``created_at``, and ``score``.
        """
        return self.client.query(
            query=query,
            top_k=top_k,
            project_id=self.project_id,
            session_id=self.session_id,
            filters=filters,
            include_score=True,
        )

    def delete_memory(self, ids: List[str]) -> int:
        """Delete one or more memory items by their IDs.

        Returns the number of records confirmed deleted by the server.
        """
        return self.client.delete(ids)

    # ------------------------------------------------------------------
    # Lower-level helpers (callers that manage embeddings themselves)
    # ------------------------------------------------------------------

    def store_with_embedding(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Bypass the server-side embed step and store a pre-computed vector."""
        return self.client.store(
            text=text,
            embedding=embedding,
            project_id=self.project_id,
            session_id=self.session_id,
            metadata=metadata,
        )

    def search_with_embedding(
        self,
        embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search with a pre-computed query embedding."""
        return self.client.query(
            embedding=embedding,
            top_k=top_k,
            project_id=self.project_id,
            session_id=self.session_id,
            include_score=True,
        )
