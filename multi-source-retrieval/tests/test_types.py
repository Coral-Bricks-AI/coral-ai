"""Tests for core types."""

import pytest

from coral_retrieval.types import SearchHit, RetrievalResult


class TestSearchHit:
    def test_basic_creation(self):
        hit = SearchHit(id="1", text="hello", score=0.9, source="test")
        assert hit.id == "1"
        assert hit.text == "hello"
        assert hit.score == 0.9
        assert hit.source == "test"
        assert hit.metadata == {}

    def test_with_metadata(self):
        hit = SearchHit(id="1", text="hi", score=0.5, source="s", metadata={"k": "v"})
        assert hit.metadata == {"k": "v"}

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SearchHit(id="", text="hi", score=0.5, source="s")


class TestRetrievalResult:
    def _hits(self, n: int) -> list[SearchHit]:
        return [
            SearchHit(id=str(i), text=f"doc {i}", score=1.0 - i * 0.1, source="s")
            for i in range(n)
        ]

    def test_top_returns_best(self):
        result = RetrievalResult(hits=self._hits(3), sources_queried=["s"])
        assert result.top is not None
        assert result.top.id == "0"

    def test_top_returns_none_when_empty(self):
        result = RetrievalResult(hits=[], sources_queried=[])
        assert result.top is None

    def test_texts(self):
        result = RetrievalResult(hits=self._hits(5), sources_queried=["s"])
        assert len(result.texts()) == 5
        assert len(result.texts(limit=2)) == 2
        assert result.texts(limit=2) == ["doc 0", "doc 1"]
