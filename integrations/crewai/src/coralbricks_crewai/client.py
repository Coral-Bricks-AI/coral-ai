"""HTTP client for the CoralBricks Memory API."""

from __future__ import annotations

from typing import Any, Dict, List

import requests


class CoralBricksClient:
  DEFAULT_BASE_URL = "https://memory.coralbricks.ai"

  def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL) -> None:
    self.api_key = api_key
    self.base_url = base_url.rstrip("/")
    self._headers = {
      "Content-Type": "application/json",
      "x-api-key": api_key,
    }

  # Low-level helpers -----------------------------------------------------

  def _post(self, path: str, json: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{self.base_url}{path}"
    resp = requests.post(url, headers=self._headers, json=json, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
      raise RuntimeError(f"Unexpected response from CoralBricks ({path}): {data!r}")
    return data

  # Store management ------------------------------------------------------

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
    return self._post("/v1/memory/stores/get_or_create", {"store_name": store_name})

  # Public API ------------------------------------------------------------

  def store(
    self,
    text: str,
    project_id: str | None = None,
    session_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
    store_name: str | None = None,
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
    project_id: str | None = None,
    session_id: str | None = None,
    store_name: str | None = None,
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
    project_id: str | None = None,
    session_id: str | None = None,
    store_name: str | None = None,
  ) -> Dict[str, Any]:
    """Forget memories matching a semantic query. Returns forgotten count and ids."""
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
