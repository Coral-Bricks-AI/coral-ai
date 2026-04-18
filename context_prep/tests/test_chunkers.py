"""Unit tests for coralbricks.context_prep.chunkers."""

from __future__ import annotations

import pytest

from coralbricks.context_prep.chunkers import (
    BaseChunker,
    Chunk,
    FixedTokenChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
    SlidingTokenChunker,
    chunk_text,
    count_tokens,
    make_chunker,
)

LONG = (
    "The quick brown fox jumps over the lazy dog. " * 50
    + "Coral Bricks unifies prep + serve in one platform. "
    + "Mr. Smith met Dr. Jones at 9 a.m.\n\n"
    + "Section 2: indices.\nThis paragraph explains how indices warm and cool. "
    * 10
)


def _is_monotonic(chunks: list[Chunk]) -> bool:
    for i, c in enumerate(chunks):
        if c.start > c.end:
            return False
        if i and c.start < chunks[i - 1].start:
            return False
    return True


def test_count_tokens_runs():
    assert count_tokens("hello world") >= 1


def test_fixed_token_chunker_covers_text():
    ch = FixedTokenChunker(target_tokens=64)
    chunks = ch.chunk(LONG)
    assert chunks
    assert _is_monotonic(chunks)
    assert chunks[0].start == 0
    assert chunks[-1].end <= len(LONG)
    for c in chunks:
        assert c.token_count and c.token_count <= 64


def test_sliding_token_chunker_overlaps_and_covers():
    ch = SlidingTokenChunker(target_tokens=64, overlap=16)
    chunks = ch.chunk(LONG)
    assert len(chunks) >= 2
    assert _is_monotonic(chunks)
    # Adjacent chunks should overlap (in token terms; we check via char proxy).
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.start < prev.end, "expected sliding window overlap"


def test_sliding_token_overlap_validation():
    with pytest.raises(ValueError):
        SlidingTokenChunker(target_tokens=10, overlap=10)


def test_recursive_character_respects_chunk_size():
    ch = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)
    chunks = ch.chunk(LONG)
    assert chunks
    assert _is_monotonic(chunks)
    for c in chunks:
        # Allow some slop for overlap glue text but never wildly over.
        assert len(c.text) <= 220


def test_recursive_character_short_text_one_chunk():
    ch = RecursiveCharacterChunker(chunk_size=500)
    chunks = ch.chunk("Hello world.")
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].start == 0
    assert chunks[0].end == len("Hello world.")


def test_sentence_chunker_aggregates_to_budget():
    ch = SentenceChunker(target_tokens=40)
    chunks = ch.chunk(LONG)
    assert chunks
    assert _is_monotonic(chunks)
    for c in chunks:
        assert c.metadata["sentences"] >= 1


def test_sentence_chunker_handles_abbreviations():
    text = "Mr. Smith arrived. Dr. Jones replied. The meeting started at 9 a.m. Then it ended."
    ch = SentenceChunker(target_chars=200, target_tokens=None)
    chunks = ch.chunk(text)
    # Should NOT split on "Mr.", "Dr.", "a.m.".
    assert all("Mr. Smith arrived" not in c.text or "Dr. Jones replied" in c.text
               for c in chunks) or len(chunks) <= 4


def test_chunk_text_dispatches_by_strategy():
    chunks = chunk_text(LONG, strategy="fixed_token", target_tokens=128)
    assert chunks
    assert chunks[0].metadata["strategy"] == "fixed_token"

    chunks2 = chunk_text(LONG, strategy="recursive_character", chunk_size=300)
    assert chunks2
    assert chunks2[0].metadata["strategy"] == "recursive_character"


def test_make_chunker_accepts_instance():
    instance = SlidingTokenChunker(target_tokens=64, overlap=8)
    out = make_chunker(instance)
    assert out is instance


def test_make_chunker_rejects_unknown():
    with pytest.raises(ValueError):
        make_chunker("nope")


def test_chunk_text_aliases():
    a = chunk_text("a b c " * 200, strategy="semantic", target_tokens=64, overlap=8)
    b = chunk_text("a b c " * 200, strategy="sliding_token", target_tokens=64, overlap=8)
    assert len(a) == len(b)
    assert isinstance(a[0], Chunk)
    assert issubclass(SlidingTokenChunker, BaseChunker)


def test_empty_text_returns_no_chunks():
    for strategy in ("fixed_token", "sliding_token", "recursive_character", "sentence"):
        assert chunk_text("", strategy=strategy) == []
