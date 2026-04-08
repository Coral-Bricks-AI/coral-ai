"""Weighted Score Fusion.

Normalizes per-backend scores to [0, 1] via min-max scaling, applies
per-backend weights, and sums. Useful when backend scores are on
different scales (e.g. BM25 vs cosine similarity).
"""

from __future__ import annotations

from ..types import SearchHit


def weighted_score_fusion(
    ranked_lists: list[list[SearchHit]],
    weights: list[float] | None = None,
) -> list[SearchHit]:
    """Fuse *ranked_lists* using weighted, normalized scores.

    Args:
        ranked_lists: One list of :class:`SearchHit` per backend.
        weights: Per-backend multipliers. Defaults to equal weight (1.0).
                 Length must match *ranked_lists*.

    Returns:
        Merged hits sorted by fused score (descending).
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match "
            f"ranked_lists length ({len(ranked_lists)})"
        )

    normalized = [_min_max_normalize(rl) for rl in ranked_lists]

    scores: dict[str, float] = {}
    best_hit: dict[str, SearchHit] = {}

    for weight, result_list in zip(weights, normalized):
        for hit in result_list:
            scores[hit.id] = scores.get(hit.id, 0.0) + weight * hit.score

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
                metadata={**original.metadata, "wsf_score": fused_score},
            )
        )

    fused.sort(key=lambda h: h.score, reverse=True)
    return fused


def _min_max_normalize(hits: list[SearchHit]) -> list[SearchHit]:
    """Scale scores to [0, 1] within a single ranked list.

    When all scores are identical (including single-item lists), every
    hit receives a normalized score of 1.0.
    """
    if not hits:
        return []
    scores = [h.score for h in hits]
    lo, hi = min(scores), max(scores)
    spread = hi - lo
    return [
        SearchHit(
            id=h.id,
            text=h.text,
            score=(h.score - lo) / spread if spread > 0 else 1.0,
            source=h.source,
            metadata=h.metadata,
        )
        for h in hits
    ]
