"""HTTP client for the CoralBricks Memory API.

Provides the same core surface as the crewAI client (store, search, forget,
store management) plus LangChain-specific chat-history endpoints:

  POST /v1/memory/chat  – append a chat message
  GET  /v1/memory/chat  – list chat messages for a conversation
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import requests


class CoralBricksClient:
    """Thin HTTP client wrapping the CoralBricks Memory API.

    Args:
        api_key: Your CoralBricks API key (sent as ``x-api-key`` header).
        base_url: Root URL of the memory-api service.
                  Defaults to ``https://memory.coralbricks.ai``.
    """

    DEFAULT_BASE_URL = "https://memory.coralbricks.ai"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
        }

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, json: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = requests.post(url, headers=self._headers, json=json, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Unexpected response from CoralBricks ({path}): {data!r}"
            )
        return data

    def _get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = requests.get(
            url, headers=self._headers, params=params or {}, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Unexpected response from CoralBricks ({path}): {data!r}"
            )
        return data

    # ------------------------------------------------------------------
    # Store management
    # ------------------------------------------------------------------

    def create_memory_store(self, store_name: str) -> Dict[str, Any]:
        """Create a new memory store (TurboPuffer namespace).

        Returns dict with store_name, namespace, and created flag.
        Raises if the store already exists (HTTP 409).
        """
        return self._post("/v1/memory/stores", {"store_name": store_name})

    def get_or_create_memory_store(self, store_name: str) -> Dict[str, Any]:
        """Get an existing memory store or create it. Idempotent.

        Returns dict with store_name, namespace, and created flag.
        """
        return self._post(
            "/v1/memory/stores/get_or_create", {"store_name": store_name}
        )

    # ------------------------------------------------------------------
    # Core memory operations
    # ------------------------------------------------------------------

    def store(
        self,
        text: str,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        store_name: Optional[str] = None,
    ) -> str:
        """Store a memory item. Returns the new memory id."""
        payload: Dict[str, Any] = {"text": text}
        if project_id is not None:
            payload["project_id"] = project_id
        if session_id is not None:
            payload["session_id"] = session_id
        if metadata is not None:
            payload["metadata"] = metadata
        if store_name is not None:
            payload["store"] = store_name
        data = self._post("/store", payload)
        mem_id = data.get("id")
        if not isinstance(mem_id, str):
            raise RuntimeError("CoralBricks /store did not return an id")
        return mem_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        store_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search memories by query text. Returns list of result dicts."""
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        payload: Dict[str, Any] = {"query": query, "top_k": top_k}
        if project_id is not None:
            payload["project_id"] = project_id
        if session_id is not None:
            payload["session_id"] = session_id
        if store_name is not None:
            payload["store"] = store_name
        data = self._post("/search", payload)
        results = data.get("results")
        if not isinstance(results, list):
            return []
        return [r for r in results if isinstance(r, dict)]

    def forget(
        self,
        query: str,
        top_k: int = 5,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        store_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forget memories matching a semantic query.

        Returns dict with forgotten_count and forgotten_ids.
        """
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        payload: Dict[str, Any] = {"query": query, "top_k": top_k}
        if project_id is not None:
            payload["project_id"] = project_id
        if session_id is not None:
            payload["session_id"] = session_id
        if store_name is not None:
            payload["store"] = store_name
        return self._post("/v1/memory/forget", payload)

    # ------------------------------------------------------------------
    # Chat history endpoints (LangChain-specific)
    # ------------------------------------------------------------------

    def append_chat(
        self,
        conversation_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        chunk_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Append a message to a conversation via ``POST /v1/memory/chat``."""
        if not conversation_id or not conversation_id.strip():
            raise ValueError("conversation_id must be a non-empty string")
        if role not in ("user", "assistant", "system"):
            raise ValueError("role must be one of: user, assistant, system")
        payload: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
        }
        if chunk_ids is not None:
            payload["chunk_ids"] = chunk_ids
        return self._post("/v1/memory/chat", payload)

    def list_chat(
        self,
        conversation_id: str,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """List messages for a conversation via ``GET /v1/memory/chat``."""
        if not conversation_id or not conversation_id.strip():
            raise ValueError("conversation_id must be a non-empty string")
        data = self._get(
            "/v1/memory/chat",
            params={"conversation_id": conversation_id, "limit": limit},
        )
        messages = data.get("messages")
        if not isinstance(messages, list):
            return []
        return [m for m in messages if isinstance(m, dict)]
