"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009).

Merges multiple ranked lists by summing 1/(k + rank) for each document
across all lists. Documents appearing in more lists and at higher ranks
receive higher fused scores. The constant *k* (default 60) controls how
much weight is given to top vs. lower ranks.
"""

from __future__ import annotations

from ..types import SearchHit


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchHit]],
    k: int = 60,
) -> list[SearchHit]:
    """Fuse *ranked_lists* using RRF and return a single merged ranking.

    Args:
        ranked_lists: One list of :class:`SearchHit` per backend.
        k: Smoothing constant. Higher values flatten rank differences.

    Returns:
        Merged hits sorted by fused score (descending). Each hit's
        ``score`` is replaced with the RRF score.
    """
    scores: dict[str, float] = {}
    best_hit: dict[str, SearchHit] = {}

    for result_list in ranked_lists:
        for rank, hit in enumerate(result_list):
            rrf_contrib = 1.0 / (k + rank + 1)
            scores[hit.id] = scores.get(hit.id, 0.0) + rrf_contrib

            prev = best_hit.get(hit.id)
            if prev is None or hit.score > prev.score:
                best_hit[hit.id] = hit

    fused: list[SearchHit] = []
    for hit_id, fused_score in scores.items():
        original = best_hit[hit_id]
        fused.append(
            SearchHit(
                id=original.id,
                text=original.text,
                score=fused_score,
                source=original.source,
                metadata={**original.metadata, "rrf_score": fused_score},
            )
        )

    fused.sort(key=lambda h: h.score, reverse=True)
    return fused
