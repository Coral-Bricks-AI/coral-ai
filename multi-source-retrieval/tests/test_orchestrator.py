"""Tests for the MultiSourceRetriever orchestrator."""

import pytest

from coral_retrieval import MultiSourceRetriever, SearchHit
from coral_retrieval.backends.base import RetrievalBackend


class StubBackend:
    """Deterministic backend for testing."""

    def __init__(self, name: str, hits: list[SearchHit]) -> None:
        self._name = name
        self._hits = hits

    @property
    def name(self) -> str:
        return self._name

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        return self._hits[:top_k]


class FailingBackend:
    """Backend that always raises."""

    @property
    def name(self) -> str:
        return "failing"

    def search(self, query: str, *, top_k: int = 10) -> list[SearchHit]:
        raise RuntimeError("boom")


def _hits(source: str, ids: list[str]) -> list[SearchHit]:
    return [
        SearchHit(id=id_, text=f"text-{id_}", score=1.0 - i * 0.1, source=source)
        for i, id_ in enumerate(ids)
    ]


class TestMultiSourceRetriever:
    def test_single_backend(self):
        r = MultiSourceRetriever(fusion="rrf")
        r.add(StubBackend("v", _hits("v", ["a", "b"])))
        result = r.search("test")
        assert len(result.hits) == 2
        assert result.sources_queried == ["v"]

    def test_multi_backend_fusion(self):
        r = MultiSourceRetriever(fusion="rrf")
        r.add(StubBackend("v", _hits("v", ["a", "b", "c"])))
        r.add(StubBackend("g", _hits("g", ["b", "d"])))
        result = r.search("test", top_k=5)
        ids = [h.id for h in result.hits]
        assert "b" in ids
        assert result.sources_queried == ["v", "g"]
        assert result.fusion_strategy == "rrf"

    def test_source_filtering(self):
        r = MultiSourceRetriever(fusion="rrf")
        r.add(StubBackend("v", _hits("v", ["a"])))
        r.add(StubBackend("g", _hits("g", ["b"])))
        result = r.search("test", sources=["g"])
        assert result.sources_queried == ["g"]
        assert all(h.source == "g" for h in result.hits)

    def test_no_matching_sources_returns_empty(self):
        r = MultiSourceRetriever()
        r.add(StubBackend("v", _hits("v", ["a"])))
        result = r.search("test", sources=["nonexistent"])
        assert result.hits == []
        assert result.sources_queried == []

    def test_failing_backend_isolated(self):
        r = MultiSourceRetriever(fusion="rrf")
        r.add(StubBackend("v", _hits("v", ["a", "b"])))
        r.add(FailingBackend())
        result = r.search("test")
        assert len(result.hits) == 2
        assert result.sources_queried == ["v", "failing"]

    def test_top_k_respected(self):
        r = MultiSourceRetriever()
        r.add(StubBackend("v", _hits("v", ["a", "b", "c", "d", "e"])))
        result = r.search("test", top_k=2)
        assert len(result.hits) == 2

    def test_chaining(self):
        r = MultiSourceRetriever()
        r.add(StubBackend("a", [])).add(StubBackend("b", []))
        assert r.backends == ["a", "b"]

    def test_parallel_flag(self):
        r_parallel = MultiSourceRetriever(parallel=True)
        r_serial = MultiSourceRetriever(parallel=False)
        for r in (r_parallel, r_serial):
            r.add(StubBackend("v", _hits("v", ["a"])))
            r.add(StubBackend("g", _hits("g", ["b"])))
        assert r_parallel.search("test").hits
        assert r_serial.search("test").hits

    def test_wsf_strategy(self):
        r = MultiSourceRetriever(fusion="wsf")
        r.add(StubBackend("v", _hits("v", ["a"])))
        result = r.search("test")
        assert result.fusion_strategy == "wsf"

    def test_custom_fusion(self):
        def my_fusion(ranked_lists):
            all_hits = [h for rl in ranked_lists for h in rl]
            all_hits.sort(key=lambda h: h.id)
            return all_hits

        r = MultiSourceRetriever(fusion=my_fusion)
        r.add(StubBackend("v", _hits("v", ["c", "a"])))
        result = r.search("test")
        assert [h.id for h in result.hits] == ["a", "c"]

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown fusion"):
            MultiSourceRetriever(fusion="nonexistent")
