"""HTTP client for the CoralBricks Memory API.

Extends the base client used by the crewAI integration with additional
methods for the v1 REST surface:

  POST /v1/memory/save    – batch-save memory items
  POST /v1/memory/query   – semantic search (v1)
  POST /v1/memory/delete  – delete memories by ID
  POST /v1/memory/chat    – append a chat message
  GET  /v1/memory/chat    – list chat messages for a conversation

The legacy /embed, /store, /search endpoints are also present and behave
identically to the crewAI client for parity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import requests


class CoralBricksClient:
    """Thin HTTP client wrapping the CoralBricks Memory API.

    Args:
        api_key: Your CoralBricks API key (sent as ``x-api-key`` header).
        base_url: Root URL of the memory-api service.
                  Defaults to ``https://cw.coralbricks.ai``.
    """

    DEFAULT_BASE_URL = "https://cw.coralbricks.ai"

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

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    # Legacy endpoints (parity with crewAI client)
    # ------------------------------------------------------------------

    def embed(self, text: str) -> List[float]:
        """Call ``/embed`` to obtain an embedding vector for *text*."""
        data = self._post("/embed", {"text": text})
        emb = data.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError("CoralBricks /embed returned invalid embedding")
        return [float(x) for x in emb]

    def store(
        self,
        text: str,
        embedding: Optional[List[float]] = None,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a memory item via ``/store``. Returns the new memory id."""
        payload: Dict[str, Any] = {"text": text}
        if embedding is not None:
            payload["embedding"] = embedding
        if project_id is not None:
            payload["project_id"] = project_id
        if session_id is not None:
            payload["session_id"] = session_id
        if metadata is not None:
            payload["metadata"] = metadata
        data = self._post("/store", payload)
        mem_id = data.get("id")
        if not isinstance(mem_id, str):
            raise RuntimeError("CoralBricks /store did not return an id")
        return mem_id

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search memories via ``/search``. Returns a list of result dicts."""
        if embedding is None and (query is None or not query.strip()):
            raise ValueError("Either query text or embedding must be provided")
        payload: Dict[str, Any] = {"top_k": top_k}
        if query is not None:
            payload["query"] = query
        if embedding is not None:
            payload["embedding"] = embedding
        if project_id is not None:
            payload["project_id"] = project_id
        if session_id is not None:
            payload["session_id"] = session_id
        data = self._post("/search", payload)
        results = data.get("results")
        if not isinstance(results, list):
            return []
        return [r for r in results if isinstance(r, dict)]

    # ------------------------------------------------------------------
    # v1 endpoints
    # ------------------------------------------------------------------

    def save(
        self,
        items: List[Dict[str, Any]],
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Batch-save memory items via ``POST /v1/memory/save``.

        Each item must have at least a ``text`` key. Optional keys:
        ``type``, ``metadata``, ``client_id``.

        Returns a list of ``{id, client_id, status}`` dicts.
        """
        payload: Dict[str, Any] = {"items": items}
        if project_id is not None:
            payload["project_id"] = project_id
        if session_id is not None:
            payload["session_id"] = session_id
        data = self._post("/v1/memory/save", payload)
        return data.get("items", [])

    def query(
        self,
        query: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        top_k: int = 5,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_score: bool = True,
    ) -> List[Dict[str, Any]]:
        """Semantic search via ``POST /v1/memory/query``.

        Returns a list of hit dicts each containing ``id``, ``text``,
        ``metadata``, ``created_at``, and optionally ``score``.
        """
        if embedding is None and (query is None or not query.strip()):
            raise ValueError("Either query text or embedding must be provided")
        payload: Dict[str, Any] = {"top_k": top_k, "include_score": include_score}
        if query is not None:
            payload["query"] = query
        if embedding is not None:
            payload["embedding"] = embedding
        if project_id is not None:
            payload["project_id"] = project_id
        if session_id is not None:
            payload["session_id"] = session_id
        if filters is not None:
            payload["filters"] = filters
        data = self._post("/v1/memory/query", payload)
        hits = data.get("hits")
        if not isinstance(hits, list):
            return []
        return [h for h in hits if isinstance(h, dict)]

    def delete(self, ids: List[str]) -> int:
        """Delete memories by ID via ``POST /v1/memory/delete``.

        Returns the number of records deleted (``deleted_count`` from the API).
        Note: currently returns 0 as the API stub is best-effort.
        """
        if not ids:
            raise ValueError("ids must be a non-empty list")
        data = self._post("/v1/memory/delete", {"ids": ids})
        return int(data.get("deleted_count", 0))

    def append_chat(
        self,
        conversation_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        chunk_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Append a message to a conversation via ``POST /v1/memory/chat``.

        Returns API response dict with ``message_id``, ``conversation_id``,
        ``role``, ``message_index``, and ``created_at``.
        """
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
        """List messages for a conversation via ``GET /v1/memory/chat``.

        Returns list of message dicts ordered by ``message_index``, each
        containing ``message_id``, ``conversation_id``, ``role``, ``content``,
        ``created_at``, and ``chunk_ids``.
        """
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
