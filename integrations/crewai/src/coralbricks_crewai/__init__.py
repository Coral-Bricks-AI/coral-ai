"""Coralbricks + CrewAI integration package.

Provides:
- CoralBricksClient: low-level HTTP client for the Coralbricks Memory API.
- CoralBricksMemory: simple memory helper (embed + store + search).
- search_coralbricks_memory: optional CrewAI tool for agents.
"""

from .client import CoralBricksClient
from .memory import CoralBricksMemory
from .tools import search_coralbricks_memory, set_global_memory

__all__ = [
    "CoralBricksClient",
    "CoralBricksMemory",
    "search_coralbricks_memory",
    "set_global_memory",
]

