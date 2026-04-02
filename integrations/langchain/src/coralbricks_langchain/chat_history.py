"""CoralBricksChatMessageHistory – LangChain BaseChatMessageHistory backed
by the CoralBricks /v1/memory/chat endpoints.

Compatible with RunnableWithMessageHistory and any LangChain component
that accepts a BaseChatMessageHistory.

Usage::

    from coralbricks_langchain import CoralBricksMemory, CoralBricksChatMessageHistory

    memory = CoralBricksMemory(api_key="cb-...")
    history = CoralBricksChatMessageHistory(
        client=memory.client,
        conversation_id="conv-001",
    )
"""

from __future__ import annotations

from typing import List, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import ConfigDict

from .client import CoralBricksClient


def _row_to_message(row: dict) -> BaseMessage:
    """Convert a CoralBricks chat row dict to the appropriate LangChain message type."""
    role: str = row.get("role", "user")
    content: str = str(row.get("content", ""))
    if role == "assistant":
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    return HumanMessage(content=content)


class CoralBricksChatMessageHistory(BaseChatMessageHistory):
    """Persistent chat message history stored in the CoralBricks Memory API.

    Messages are appended to the remote service on every ``add_message`` call
    and fetched lazily when the ``messages`` property is accessed.

    Args:
        client: A :class:`~coralbricks_langchain.client.CoralBricksClient`
                (accessible via ``memory.client``).
        conversation_id: Unique string identifying this conversation.
        limit: Maximum number of messages to retrieve (default 500).

    Note:
        ``clear()`` is a no-op — the CoralBricks chat API has no bulk-delete
        endpoint. Use a new ``conversation_id`` to start fresh.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: CoralBricksClient
    conversation_id: str
    limit: int = 500

    # ------------------------------------------------------------------
    # BaseChatMessageHistory interface
    # ------------------------------------------------------------------

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        """Fetch and return all messages for this conversation in order."""
        rows = self.client.list_chat(
            conversation_id=self.conversation_id,
            limit=self.limit,
        )
        return [_row_to_message(r) for r in rows]

    def add_message(self, message: BaseMessage) -> None:
        """Append a single :class:`~langchain_core.messages.BaseMessage`."""
        if isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "user"
        self.client.append_chat(
            conversation_id=self.conversation_id,
            role=role,  # type: ignore[arg-type]
            content=str(message.content),
        )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append multiple messages sequentially."""
        for msg in messages:
            self.add_message(msg)

    def clear(self) -> None:
        """No-op: chat bulk-delete is not supported by the CoralBricks API.

        To start a fresh conversation, instantiate a new
        ``CoralBricksChatMessageHistory`` with a different ``conversation_id``.
        """
