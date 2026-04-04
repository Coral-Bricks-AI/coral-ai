"""Multi-source retrieval orchestrator.

The :class:`MultiSourceRetriever` fans queries out to registered backends,
collects ranked results, and fuses them into a single
:class:`~coral_retrieval.RetrievalResult`.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Callable, Literal

from .backends.base import RetrievalBackend
from .fusion import reciprocal_rank_fusion, weighted_score_fusion
from .types import RetrievalResult, SearchHit

logger = logging.getLogger(__name__)

FusionFn = Callable[[list[list[SearchHit]]], list[SearchHit]]

_BUILTIN_STRATEGIES: dict[str, FusionFn] = {
    "rrf": reciprocal_rank_fusion,
    "wsf": weighted_score_fusion,
}


class MultiSourceRetriever:
    """Orchestrate retrieval across heterogeneous backends.

    Usage::

        retriever = MultiSourceRetriever()
        retriever.add(vector_backend)
        retriever.add(graph_backend)
        retriever.add(sql_backend)

        result = retriever.search("recent acquisitions in fintech", top_k=10)
        for hit in result.hits:
            print(f"[{hit.source}] {hit.score:.4f}  {hit.text[:80]}")

    Args:
        fusion: Built-in strategy name (``"rrf"`` or ``"wsf"``) or a
                callable with signature
                ``(list[list[SearchHit]]) -> list[SearchHit]``.
        parallel: If True (default), query backends concurrently.
        timeout: Per-backend timeout in seconds when running in parallel.
    """

    def __init__(
        self,
        fusion: Literal["rrf", "wsf"] | FusionFn = "rrf",
        parallel: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self._backends: list[RetrievalBackend] = []

        if isinstance(fusion, str):
            if fusion not in _BUILTIN_STRATEGIES:
                raise ValueError(
                    f"Unknown fusion strategy {fusion!r}. "
                    f"Choose from {list(_BUILTIN_STRATEGIES)} or pass a callable."
                )
            self._fusion_fn = _BUILTIN_STRATEGIES[fusion]
            self._fusion_name = fusion
        else:
            self._fusion_fn = fusion
            self._fusion_name = getattr(fusion, "__name__", "custom")

        self._parallel = parallel
        self._timeout = timeout

    @property
    def backends(self) -> list[str]:
        """Names of registered backends."""
        return [b.name for b in self._backends]

    def add(self, backend: RetrievalBackend) -> "MultiSourceRetriever":
        """Register a retrieval backend. Returns self for chaining."""
        self._backends.append(backend)
        return self

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        sources: list[str] | None = None,
    ) -> RetrievalResult:
        """Query backends and return fused results.

        Args:
            query: The search query.
            top_k: Maximum results to return after fusion.
            sources: If provided, only query backends whose ``name``
                     is in this list. Otherwise query all.

        Returns:
            A :class:`RetrievalResult` with fused hits.
        """
        targets = self._backends
        if sources:
            targets = [b for b in self._backends if b.name in sources]
            if not targets:
                logger.warning("No backends matched sources=%s", sources)
                return RetrievalResult(hits=[], sources_queried=[], fusion_strategy=self._fusion_name)

        ranked_lists = self._fan_out(query, targets, top_k)
        fused = self._fusion_fn(ranked_lists)

        return RetrievalResult(
            hits=fused[:top_k],
            sources_queried=[b.name for b in targets],
            fusion_strategy=self._fusion_name,
        )

    def _fan_out(
        self,
        query: str,
        backends: list[RetrievalBackend],
        top_k: int,
    ) -> list[list[SearchHit]]:
        """Query all backends, optionally in parallel."""
        if not self._parallel or len(backends) == 1:
            return [self._safe_search(b, query, top_k) for b in backends]

        ranked_lists: list[list[SearchHit]] = [[] for _ in backends]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(backends)) as pool:
            futures = {
                pool.submit(self._safe_search, b, query, top_k): i
                for i, b in enumerate(backends)
            }
            for future in concurrent.futures.as_completed(futures, timeout=self._timeout):
                idx = futures[future]
                ranked_lists[idx] = future.result()

        return ranked_lists

    @staticmethod
    def _safe_search(
        backend: RetrievalBackend, query: str, top_k: int
    ) -> list[SearchHit]:
        """Call backend.search with error isolation."""
        try:
            return backend.search(query, top_k=top_k)
        except Exception:
            logger.exception("Backend %s failed for query=%r", backend.name, query)
            return []
