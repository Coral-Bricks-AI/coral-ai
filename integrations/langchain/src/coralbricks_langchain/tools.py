"""LangChain tools for CoralBricks memory.

Three tools are provided using the @tool decorator (recommended in langchain >= 1.0):

* store_coralbricks_memory  – embed and persist a text snippet.
* search_coralbricks_memory – semantic search over stored memories.
* delete_coralbricks_memory – delete memories by comma-separated IDs.

Quick-start::

    from coralbricks_langchain import (
        CoralBricksClient, CoralBricksMemory,
        set_global_memory, get_tools,
    )

    client = CoralBricksClient(api_key="cb-...")
    memory = CoralBricksMemory(client, project_id="my-project")
    set_global_memory(memory)

    tools = get_tools()
"""

from __future__ import annotations

import json
from typing import List, Optional

from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field

from .memory import CoralBricksMemory


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_memory: Optional[CoralBricksMemory] = None


def set_global_memory(memory: CoralBricksMemory) -> None:
    """Configure the global :class:`CoralBricksMemory` instance used by all tools.

    Must be called before any tool is invoked.
    """
    global _memory
    _memory = memory


def _require_memory() -> CoralBricksMemory:
    if _memory is None:
        raise RuntimeError(
            "CoralBricks memory is not configured. Call set_global_memory() first."
        )
    return _memory


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class _StoreInput(BaseModel):
    text: str = Field(description="The text or fact to embed and store as a memory item.")
    metadata_json: Optional[str] = Field(
        default=None,
        description=(
            "Optional JSON string of key/value metadata to attach, "
            'e.g. \'{"source": "user", "topic": "pricing"}\'. Leave blank if not needed.'
        ),
    )


class _SearchInput(BaseModel):
    query: str = Field(description="Natural language query to search memories with.")
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return (1–50). Defaults to 5.",
    )


class _DeleteInput(BaseModel):
    ids: str = Field(
        description=(
            "Comma-separated memory IDs to delete, e.g. 'mem_abc123' or "
            "'mem_abc123, mem_def456'. IDs are returned by the store tool."
        )
    )


# ---------------------------------------------------------------------------
# Tools (@tool decorator — recommended pattern in langchain >= 1.0)
# ---------------------------------------------------------------------------

@tool(args_schema=_StoreInput)
def store_coralbricks_memory(text: str, metadata_json: Optional[str] = None) -> str:
    """Embed a piece of text and persist it in CoralBricks long-term memory.

    Use this when you want to remember a fact, policy, or user preference for later.
    Returns the assigned memory ID.
    """
    mem = _require_memory()
    metadata = None
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            return f"Error: metadata_json is not valid JSON – {exc}"
    try:
        mem_id = mem.save_memory(text=text, metadata=metadata)
        return f"Memory stored successfully. ID: {mem_id}"
    except Exception as exc:  # noqa: BLE001
        return f"Error storing memory: {exc}"


@tool(args_schema=_SearchInput)
def search_coralbricks_memory(query: str, top_k: int = 5) -> str:
    """Search CoralBricks long-term memory for relevant context.

    Always use this before answering questions that may involve stored facts
    or preferences. Returns the top matching memories with relevance scores.
    """
    mem = _require_memory()
    try:
        results = mem.search_memory(query=query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        return f"Error searching memory: {exc}"
    if not results:
        return "No relevant memories found."
    lines: List[str] = []
    for r in results:
        text = str(r.get("text", ""))
        score = r.get("score")
        mem_id = r.get("id", "")
        if isinstance(score, (int, float)):
            lines.append(f"[score={score:.3f}] [id={mem_id}] {text}")
        else:
            lines.append(f"[id={mem_id}] {text}")
    return "\n".join(lines)


@tool(args_schema=_DeleteInput)
def delete_coralbricks_memory(ids: str) -> str:
    """Delete one or more CoralBricks memory items by their IDs.

    Use this to remove outdated or incorrect information from memory.
    Memory IDs are returned by the store tool (format: 'mem_xxxxxxxx').
    """
    mem = _require_memory()
    ids = ids.strip()
    if ids.startswith("["):
        try:
            parsed = json.loads(ids)
            id_list = [str(i).strip() for i in parsed if str(i).strip()]
        except json.JSONDecodeError:
            id_list = [i.strip().strip('"').strip("'") for i in ids.strip("[]").split(",") if i.strip()]
    else:
        id_list = [i.strip().strip('"').strip("'") for i in ids.split(",") if i.strip()]
    if not id_list:
        return "Error: no valid IDs provided."
    try:
        deleted_count = mem.delete_memory(id_list)
        return (
            f"Delete request sent for {len(id_list)} ID(s). "
            f"Server confirmed {deleted_count} deleted. IDs: {id_list}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error deleting memories: {exc}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_tools() -> List[BaseTool]:
    """Return all three CoralBricks memory tools as a list.

    ``set_global_memory()`` must be called before any tool is *invoked*.

    Returns:
        ``[store_coralbricks_memory, search_coralbricks_memory, delete_coralbricks_memory]``
    """
    return [
        store_coralbricks_memory,
        search_coralbricks_memory,
        delete_coralbricks_memory,
    ]
