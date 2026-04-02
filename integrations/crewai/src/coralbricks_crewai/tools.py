"""CrewAI tools for CoralBricks memory."""

from __future__ import annotations

from typing import ClassVar

from crewai.tools import BaseTool
from pydantic import Field

from .memory import CoralBricksMemory


class SearchCoralBricksMemoryTool(BaseTool):
  name: str = "search_coralbricks_memory"
  description: str = "Search CoralBricks memory for relevant context given a natural language query."

  memory: CoralBricksMemory = Field(exclude=True)
  top_k: int = 5

  class Config:
    arbitrary_types_allowed: ClassVar[bool] = True

  def _run(self, query: str) -> str:  # type: ignore[override]
    results = self.memory.search_memory(query, top_k=self.top_k)
    lines: list[str] = []
    for r in results:
      text = str(r.get("text", ""))
      score = r.get("score")
      if isinstance(score, (int, float)):
        lines.append(f"[{score:.3f}] {text}")
      else:
        lines.append(text)
    return "\n".join(lines)
