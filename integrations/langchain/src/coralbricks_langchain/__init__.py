"""coralbricks-langchain – CoralBricks + LangChain integration.

Public API
----------
CoralBricksClient
    Low-level HTTP client for the CoralBricks Memory API.
CoralBricksMemory
    High-level helper: save_memory, search_memory, forget_memory.
CoralBricksRetriever
    LangChain BaseRetriever for LCEL RAG pipelines.
get_tools
    Factory returning [store, search, forget] tools bound to a memory instance.
"""

from .client import CoralBricksClient
from .memory import CoralBricksMemory
from .retriever import CoralBricksRetriever
from .tools import get_tools

__all__ = [
    "CoralBricksClient",
    "CoralBricksMemory",
    "CoralBricksRetriever",
    "get_tools",
]
