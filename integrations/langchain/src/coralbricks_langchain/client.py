"""HTTP client for the CoralBricks Memory API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
        item: Dict[str, Any] = {"text": text}
        if metadata is not None:
            item["metadata"] = metadata

        payload: Dict[str, Any] = {"items": [item]}
        if project_id is not None:
            payload["project_id"] = project_id
        if session_id is not None:
            payload["session_id"] = session_id
        if store_name is not None:
            payload["store"] = store_name

        data = self._post("/v1/memory/save", payload)
        items = data.get("items")
        if not isinstance(items, list) or len(items) == 0:
            raise RuntimeError("CoralBricks /v1/memory/save did not return items")
        mem_id = items[0].get("id")
        if not isinstance(mem_id, str):
            raise RuntimeError("CoralBricks /v1/memory/save did not return an id")
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
        data = self._post("/v1/memory/query", payload)
        hits = data.get("hits")
        if not isinstance(hits, list):
            return []
        return [h for h in hits if isinstance(h, dict)]

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
