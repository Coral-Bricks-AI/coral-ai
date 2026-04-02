"""LangChain tools for CoralBricks memory.

Three tools are created via ``get_tools(memory)`` — no global state required:

* store_coralbricks_memory  – persist a text snippet.
* search_coralbricks_memory – semantic search over stored memories.
* forget_coralbricks_memory – forget memories by semantic query.

Quick-start::

    from coralbricks_langchain import CoralBricksMemory, get_tools

    memory = CoralBricksMemory(api_key="cb-...")
    memory.get_or_create_memory_store("langchain:my-app")

    tools = get_tools(memory)
"""

from __future__ import annotations

import json
from typing import List, Optional

from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field

from .memory import CoralBricksMemory


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class _StoreInput(BaseModel):
    text: str = Field(
        description="The text or fact to embed and store as a memory item."
    )
    metadata_json: Optional[str] = Field(
        default=None,
        description=(
            "Optional JSON string of key/value metadata to attach, "
            'e.g. \'{"source": "user", "topic": "pricing"}\'. '
            "Leave blank if not needed."
        ),
    )


class _SearchInput(BaseModel):
    query: str = Field(
        description="Natural language query to search memories with."
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return (1–50). Defaults to 5.",
    )


class _ForgetInput(BaseModel):
    query: str = Field(
        description="Natural language description of the memories to forget."
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Max number of matching memories to remove (1–50). Defaults to 5.",
    )


# ---------------------------------------------------------------------------
# Tool factory — no global state
# ---------------------------------------------------------------------------


def get_tools(memory: CoralBricksMemory) -> List[BaseTool]:
    """Return CoralBricks memory tools bound to *memory*.

    Args:
        memory: A configured :class:`CoralBricksMemory` instance.

    Returns:
        ``[store_coralbricks_memory, search_coralbricks_memory, forget_coralbricks_memory]``
    """

    @tool(args_schema=_StoreInput)
    def store_coralbricks_memory(
        text: str, metadata_json: Optional[str] = None
    ) -> str:
        """Embed a piece of text and persist it in CoralBricks long-term memory.

        Use this when you want to remember a fact, policy, or user preference
        for later. Returns the assigned memory ID.
        """
        metadata = None
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError as exc:
                return f"Error: metadata_json is not valid JSON – {exc}"
        try:
            mem_id = memory.save_memory(text=text, metadata=metadata)
            return f"Memory stored successfully. ID: {mem_id}"
        except Exception as exc:  # noqa: BLE001
            return f"Error storing memory: {exc}"

    @tool(args_schema=_SearchInput)
    def search_coralbricks_memory(query: str, top_k: int = 5) -> str:
        """Search CoralBricks long-term memory for relevant context.

        Always use this before answering questions that may involve stored
        facts or preferences. Returns the top matching memories.
        """
        try:
            results = memory.search_memory(query=query, top_k=top_k)
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

    @tool(args_schema=_ForgetInput)
    def forget_coralbricks_memory(query: str, top_k: int = 5) -> str:
        """Forget CoralBricks memories that match a semantic query.

        Use this to remove outdated or incorrect information from memory.
        Finds the closest memories to the query and deletes them.
        """
        try:
            result = memory.forget_memory(query=query, top_k=top_k)
            count = result.get("forgotten_count", 0)
            ids = result.get("forgotten_ids", [])
            return f"Forgot {count} memory item(s). IDs: {ids}"
        except Exception as exc:  # noqa: BLE001
            return f"Error forgetting memories: {exc}"

    return [
        store_coralbricks_memory,
        search_coralbricks_memory,
        forget_coralbricks_memory,
    ]
