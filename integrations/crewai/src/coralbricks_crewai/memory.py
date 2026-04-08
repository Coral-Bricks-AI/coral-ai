"""Simple CoralBricks-backed memory helper for CrewAI apps.

This is not the CrewAI unified Memory itself; instead, it is a thin helper
that wraps CoralBricksClient with save/search methods you can call from agents
or tools.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .client import CoralBricksClient


class CoralBricksMemory:
  def __init__(
    self,
    api_key: str,
    base_url: str = CoralBricksClient.DEFAULT_BASE_URL,
    project_id: str | None = None,
    session_id: str | None = None,
  ) -> None:
    self.client = CoralBricksClient(api_key=api_key, base_url=base_url)
    self.project_id = project_id
    self.session_id = session_id
    self.store_name: str | None = None

  def set_project_id(self, project_id: str) -> None:
    self.project_id = project_id

  def set_session_id(self, session_id: str) -> None:
    self.session_id = session_id

  # Store management ------------------------------------------------------

  def get_or_create_memory_store(self, store_name: str) -> "CoralBricksMemory":
    """Attach to an existing memory store, or create it if it doesn't exist.

    Idempotent — safe to call on every startup.
    """
    self.client.get_or_create_memory_store(store_name)
    self.store_name = store_name
    return self

  # High-level helpers ----------------------------------------------------

  def save_memory(self, text: str, metadata: Dict[str, Any] | None = None) -> str:
    """Store a memory. Returns the CoralBricks memory id."""
    return self.client.store(
      text=text,
      project_id=self.project_id,
      session_id=self.session_id,
      metadata=metadata,
      store_name=self.store_name,
    )

  def search_memory(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search memory by meaning. Returns list of {text, score, ...}."""
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
