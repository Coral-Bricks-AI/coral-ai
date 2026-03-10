"""CrewAI tools for CoralBricks memory."""

from __future__ import annotations

from typing import Optional

from crewai.tools import BaseTool

from .memory import CoralBricksMemory


_memory: Optional[CoralBricksMemory] = None


def set_global_memory(memory: CoralBricksMemory) -> None:
  """Configure the global CoralBricksMemory instance used by tools."""
  global _memory
  _memory = memory


class SearchCoralBricksMemoryTool(BaseTool):
  name: str = "search_coralbricks_memory"
  description: str = "Search CoralBricks memory for relevant context given a natural language query."

  def _run(self, query: str) -> str:  # type: ignore[override]
    if _memory is None:
      return "CoralBricks memory is not configured. Call set_global_memory() first."
    results = _memory.search_memory(query, top_k=5)
    lines: list[str] = []
    for r in results:
      text = str(r.get("text", ""))
      score = r.get("score")
      if isinstance(score, (int, float)):
        lines.append(f"[{score:.3f}] {text}")
      else:
        lines.append(text)
    return "\n".join(lines)


def search_coralbricks_memory() -> SearchCoralBricksMemoryTool:
  """Factory returning a configured tool instance for use in agents."""
  return SearchCoralBricksMemoryTool()


