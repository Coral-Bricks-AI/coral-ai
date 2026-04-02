"""Coralbricks + CrewAI integration package.

Provides:
- CoralBricksClient: low-level HTTP client for the Coralbricks Memory API.
- CoralBricksMemory: simple memory helper (store + search).
- SearchCoralBricksMemoryTool: CrewAI tool for agents.
"""

from .client import CoralBricksClient
from .memory import CoralBricksMemory
from .tools import SearchCoralBricksMemoryTool

__all__ = [
    "CoralBricksClient",
    "CoralBricksMemory",
    "SearchCoralBricksMemoryTool",
]

