"""HTTP client for the CoralBricks Memory API.

This client talks to the internal CoralBricks memory-api service, which exposes
simple /embed, /store, and /search endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List

import requests


class CoralBricksClient:
  def __init__(self, api_key: str, base_url: str) -> None:
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

  # Public API ------------------------------------------------------------

  def embed(self, text: str) -> List[float]:
    """Call /embed to get an embedding vector for text."""
    data = self._post("/embed", {"text": text})
    emb = data.get("embedding")
    if not isinstance(emb, list):
      raise RuntimeError("CoralBricks /embed returned invalid embedding")
    return [float(x) for x in emb]

  def store(
    self,
    text: str,
    embedding: List[float] | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
  ) -> str:
    """Store a memory item. Returns the new memory id."""
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
    query: str | None = None,
    embedding: List[float] | None = None,
    top_k: int = 5,
    project_id: str | None = None,
    session_id: str | None = None,
  ) -> List[Dict[str, Any]]:
    """Search memories by text or embedding. Returns list of result dicts."""
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
    cleaned: List[Dict[str, Any]] = []
    for r in results:
      if isinstance(r, dict):
        cleaned.append(r)
    return cleaned

