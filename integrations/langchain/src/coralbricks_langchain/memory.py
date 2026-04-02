"""CoralBricksMemory: high-level memory helper for LangChain applications.

Creates its own CoralBricksClient internally — callers only provide
api_key and (optionally) base_url.  Mirrors the crewAI CoralBricksMemory API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .client import CoralBricksClient


class CoralBricksMemory:
    """High-level memory wrapper for LangChain applications.

    Args:
        api_key: Your CoralBricks API key.
        base_url: Root URL of the memory-api service.
        project_id: Optional project namespace (shared across agents).
        session_id: Optional session or user namespace.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = CoralBricksClient.DEFAULT_BASE_URL,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.client = CoralBricksClient(api_key=api_key, base_url=base_url)
        self.project_id = project_id
        self.session_id = session_id
        self.store_name: Optional[str] = None

    def set_project_id(self, project_id: str) -> None:
        self.project_id = project_id

    def set_session_id(self, session_id: str) -> None:
        self.session_id = session_id

    # ------------------------------------------------------------------
    # Store management
    # ------------------------------------------------------------------

    def create_memory_store(self, store_name: str) -> "CoralBricksMemory":
        """Create a new memory store.

        All subsequent save/search/forget calls will target this store.
        Raises if the store already exists.
        """
        self.client.create_memory_store(store_name)
        self.store_name = store_name
        return self

    def get_or_create_memory_store(self, store_name: str) -> "CoralBricksMemory":
        """Attach to an existing memory store, or create it if it doesn't exist.

        Idempotent — safe to call on every startup.
        """
        self.client.get_or_create_memory_store(store_name)
        self.store_name = store_name
        return self

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
        return self.client.store(
            text=text,
            project_id=self.project_id,
            session_id=self.session_id,
            metadata=metadata,
            store_name=self.store_name,
        )

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memory by natural-language *query*.

        Returns a list of hit dicts each containing at minimum
        ``text``, ``score``, and ``id``.
        """
        return self.client.search(
            query=query,
            top_k=top_k,
            project_id=self.project_id,
            session_id=self.session_id,
            store_name=self.store_name,
        )

    def forget_memory(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Forget memories that match a semantic query.

        Finds the top_k closest memories to the query and deletes them.
        Returns dict with forgotten_count and forgotten_ids.
        """
        return self.client.forget(
            query=query,
            top_k=top_k,
            project_id=self.project_id,
            session_id=self.session_id,
            store_name=self.store_name,
        )
