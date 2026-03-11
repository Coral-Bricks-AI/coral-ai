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
    client: CoralBricksClient,
    project_id: str | None = None,
    session_id: str | None = None,
  ) -> None:
    self.client = client
    self.project_id = project_id
    self.session_id = session_id

  # High-level helpers ----------------------------------------------------

  def save_memory(self, text: str, metadata: Dict[str, Any] | None = None) -> str:
    """Embed and store a single memory.

    Returns the CoralBricks memory id.
    """
    embedding = self.client.embed(text)
    return self.client.store(
      text=text,
      embedding=embedding,
      project_id=self.project_id,
      session_id=self.session_id,
      metadata=metadata,
    )

  def search_memory(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search memory by query string. Returns list of {text, score, ...}."""
    return self.client.search(
      query=query,
      embedding=None,
      top_k=top_k,
      project_id=self.project_id,
      session_id=self.session_id,
    )

  # Lower-level helpers if callers want to manage embeddings themselves ----

  def store_with_embedding(
    self,
    text: str,
    embedding: List[float],
    metadata: Dict[str, Any] | None = None,
  ) -> str:
    return self.client.store(
      text=text,
      embedding=embedding,
      project_id=self.project_id,
      session_id=self.session_id,
      metadata=metadata,
    )

  def search_with_embedding(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    return self.client.search(
      query=None,
      embedding=embedding,
      top_k=top_k,
      project_id=self.project_id,
      session_id=self.session_id,
    )

