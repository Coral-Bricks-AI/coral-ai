"""In-memory brute-force vector backend for demos and testing.

No external dependencies required.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from ..types import SearchHit


class InMemoryVectorBackend:
    """Brute-force cosine-similarity backend for demos and testing.

    No external dependencies. Stores documents in-memory.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        source_name: str = "vector",
    ) -> None:
        self._embed_fn = embed_fn
        self._source_name = source_name
        self._docs: list[dict[str, Any]] = []
        self._vectors: list[list[float]] = []

    @property
    def name(self) -> str:
        return self._source_name

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        self._docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})
        self._vectors.append(self._embed_fn(text))
        return doc_id

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        if not self._docs:
            return []
        q_vec = self._embed_fn(query)
        scored = []
        for i, d_vec in enumerate(self._vectors):
            score = self._cosine_sim(q_vec, d_vec)
            scored.append((score, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[SearchHit] = []
        for score, idx in scored[:top_k]:
            doc = self._docs[idx]
            results.append(
                SearchHit(
                    id=doc["id"],
                    text=doc["text"],
                    score=score,
                    source=self.name,
                    metadata=doc["metadata"],
                )
            )
        return results

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
