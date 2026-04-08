"""OpenSearch BM25 (keyword / lexical) search backend.

Sparse retrieval via term matching. No embedding function required.
Requires ``pip install coral-retrieval[opensearch]``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..types import SearchHit

logger = logging.getLogger(__name__)


class OpenSearchKeywordBackend:
    """BM25 lexical search against an OpenSearch index.

    Args:
        client: An ``opensearchpy.OpenSearch`` instance.
        index: Index name to query.
        text_field: Field to match against.
        source_name: Name reported in :attr:`SearchHit.source`.
    """

    def __init__(
        self,
        client: Any,
        index: str,
        text_field: str = "text",
        source_name: str = "keyword",
    ) -> None:
        self._client = client
        self._index = index
        self._text_field = text_field
        self._source_name = source_name

    @property
    def name(self) -> str:
        return self._source_name

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "match": {self._text_field: {"query": query}},
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
                if k != self._text_field
            }
            results.append(
                SearchHit(id=doc_id, text=text, score=score, source=self.name, metadata=meta)
            )
        return results
