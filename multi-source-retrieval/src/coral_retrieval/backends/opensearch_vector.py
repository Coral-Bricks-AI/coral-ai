"""OpenSearch kNN (semantic) vector search backend.

Dense retrieval via approximate nearest-neighbor search on pre-computed
embedding vectors. Requires ``pip install coral-retrieval[opensearch]``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..types import SearchHit

logger = logging.getLogger(__name__)


class OpenSearchVectorBackend:
    """kNN semantic search against an OpenSearch vector index.

    Args:
        client: An ``opensearchpy.OpenSearch`` instance.
        index: Index name to query.
        embed_fn: Callable that maps a query string to a list of floats.
        vector_field: Field containing the embedding vector.
        text_field: Field containing the document text.
        source_name: Name reported in :attr:`SearchHit.source`.
    """

    def __init__(
        self,
        client: Any,
        index: str,
        embed_fn: Callable[[str], list[float]],
        vector_field: str = "embedding",
        text_field: str = "text",
        source_name: str = "vector",
    ) -> None:
        self._client = client
        self._index = index
        self._embed_fn = embed_fn
        self._vector_field = vector_field
        self._text_field = text_field
        self._source_name = source_name

    @property
    def name(self) -> str:
        return self._source_name

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        vector = self._embed_fn(query)
        body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "knn": {
                    self._vector_field: {"vector": vector, "k": top_k},
                },
            },
        }
        resp = self._client.search(index=self._index, body=body)
        return self._parse_hits(resp)

    def _parse_hits(self, resp: dict[str, Any]) -> list[SearchHit]:
        raw_hits = resp.get("hits", {}).get("hits", [])
        results: list[SearchHit] = []
        for h in raw_hits:
            source = h.get("_source", {})
            doc_id = h.get("_id", "")
            text = source.get(self._text_field, "")
            score = float(h.get("_score", 0.0))
            meta = {
                k: v for k, v in source.items()
                if k not in (self._text_field, self._vector_field)
            }
            results.append(
                SearchHit(id=doc_id, text=text, score=score, source=self.name, metadata=meta)
            )
        return results
