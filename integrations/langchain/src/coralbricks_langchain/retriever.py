"""CoralBricksRetriever – LangChain BaseRetriever backed by CoralBricks Memory.

Drop this into any LCEL chain exactly like you would use a vector-store retriever:

    retriever = CoralBricksRetriever(memory=mem, top_k=5)
    chain = retriever | format_docs | prompt | llm | StrOutputParser()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .memory import CoralBricksMemory


class CoralBricksRetriever(BaseRetriever):
    """LangChain retriever that fetches relevant documents from CoralBricks.

    Each hit returned by the CoralBricks ``/v1/memory/query`` endpoint is
    converted to a :class:`~langchain_core.documents.Document` whose
    ``page_content`` is the stored text and whose ``metadata`` mirrors the
    hit's metadata dict, augmented with ``id``, ``score``, and ``created_at``.

    Args:
        memory: A configured :class:`~coralbricks_langchain.memory.CoralBricksMemory`.
        top_k: Number of results to retrieve per query (default 5).
        filters: Optional server-side filter dict forwarded verbatim.

    Example::

        from coralbricks_langchain import CoralBricksClient, CoralBricksMemory, CoralBricksRetriever

        client = CoralBricksClient(api_key="cb-...")
        memory = CoralBricksMemory(client, project_id="my-project")
        retriever = CoralBricksRetriever(memory=memory, top_k=5)

        docs = retriever.invoke("What is the return policy?")
    """

    # Pydantic v2: CoralBricksMemory is a plain Python class, not a BaseModel
    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory: CoralBricksMemory
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        hits = self.memory.search_memory(
            query=query,
            top_k=self.top_k,
            filters=self.filters,
        )
        docs: List[Document] = []
        for hit in hits:
            meta: Dict[str, Any] = dict(hit.get("metadata") or {})
            meta["id"] = hit.get("id", "")
            meta["created_at"] = hit.get("created_at", "")
            score = hit.get("score")
            if score is not None:
                meta["score"] = score
            docs.append(
                Document(
                    page_content=str(hit.get("text", "")),
                    metadata=meta,
                )
            )
        return docs
