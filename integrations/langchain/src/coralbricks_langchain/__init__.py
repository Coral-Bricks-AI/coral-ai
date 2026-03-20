"""coralbricks-langchain – CoralBricks + LangChain integration.

Public API
----------
CoralBricksClient
    Low-level HTTP client for the CoralBricks Memory API.
CoralBricksMemory
    High-level helper: save_memory, search_memory, delete_memory.
CoralBricksRetriever
    LangChain BaseRetriever for LCEL RAG pipelines.
CoralBricksChatMessageHistory
    LangChain BaseChatMessageHistory backed by CoralBricks chat storage.
StoreMemoryTool
    LangChain tool to embed and store a memory item.
SearchMemoryTool
    LangChain tool to semantic-search memories.
DeleteMemoryTool
    LangChain tool to delete memories by ID.
set_global_memory
    Configure the global CoralBricksMemory instance used by tools.
get_tools
    Factory returning [StoreMemoryTool, SearchMemoryTool, DeleteMemoryTool].
"""

from .chat_history import CoralBricksChatMessageHistory
from .client import CoralBricksClient
from .memory import CoralBricksMemory
from .retriever import CoralBricksRetriever
from .tools import (
    delete_coralbricks_memory,
    get_tools,
    search_coralbricks_memory,
    set_global_memory,
    store_coralbricks_memory,
)

__all__ = [
    "CoralBricksClient",
    "CoralBricksMemory",
    "CoralBricksRetriever",
    "CoralBricksChatMessageHistory",
    "store_coralbricks_memory",
    "search_coralbricks_memory",
    "delete_coralbricks_memory",
    "set_global_memory",
    "get_tools",
]
