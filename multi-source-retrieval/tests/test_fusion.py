"""Tests for fusion strategies."""

from coral_retrieval.types import SearchHit
from coral_retrieval.fusion.rrf import reciprocal_rank_fusion
from coral_retrieval.fusion.wsf import weighted_score_fusion


def _hit(id: str, score: float, source: str = "s") -> SearchHit:
    return SearchHit(id=id, text=f"doc-{id}", score=score, source=source)


class TestRRF:
    def test_single_list_preserves_order(self):
        hits = [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)]
        fused = reciprocal_rank_fusion([hits])
        assert [h.id for h in fused] == ["a", "b", "c"]

    def test_two_lists_boosts_overlap(self):
        list_a = [_hit("a", 0.9), _hit("b", 0.8)]
        list_b = [_hit("b", 0.95), _hit("c", 0.7)]
        fused = reciprocal_rank_fusion([list_a, list_b])
        # "b" appears in both lists — should rank highest
        assert fused[0].id == "b"

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_k_parameter_affects_scores(self):
        hits = [_hit("a", 0.9)]
        fused_low_k = reciprocal_rank_fusion([hits], k=1)
        fused_high_k = reciprocal_rank_fusion([hits], k=100)
        assert fused_low_k[0].score > fused_high_k[0].score


class TestWSF:
    def test_equal_weights_symmetric(self):
        list_a = [_hit("a", 0.9), _hit("b", 0.8), _hit("x", 0.1)]
        list_b = [_hit("b", 0.9), _hit("c", 0.8), _hit("y", 0.1)]
        fused = weighted_score_fusion([list_a, list_b], weights=[1.0, 1.0])
        # "b" appears in both lists with high scores — should rank first
        assert fused[0].id == "b"

    def test_weights_shift_ranking(self):
        list_a = [_hit("a", 1.0)]
        list_b = [_hit("b", 1.0)]
        fused_a_heavy = weighted_score_fusion([list_a, list_b], weights=[10.0, 1.0])
        fused_b_heavy = weighted_score_fusion([list_a, list_b], weights=[1.0, 10.0])
        assert fused_a_heavy[0].id == "a"
        assert fused_b_heavy[0].id == "b"

    def test_empty_lists(self):
        assert weighted_score_fusion([]) == []
        assert weighted_score_fusion([[], []]) == []

    def test_mismatched_weights_raises(self):
        import pytest
        with pytest.raises(ValueError, match="weights length"):
            weighted_score_fusion([[_hit("a", 1.0)]], weights=[1.0, 2.0])
