"""Sliding-window token chunker.

Like ``FixedTokenChunker`` but with a configurable overlap so adjacent
chunks share a tail / head of context. This is the standard chunking
strategy for embedding-based RAG.
"""

from __future__ import annotations

from coralbricks.context_prep.chunkers._tokens import get_tokenizer
from coralbricks.context_prep.chunkers.base import BaseChunker, Chunk


class SlidingTokenChunker(BaseChunker):
    name = "sliding_token"

    def __init__(
        self,
        *,
        target_tokens: int = 512,
        overlap: int = 64,
        encoding: str = "cl100k_base",
    ):
        if target_tokens <= 0:
            raise ValueError("target_tokens must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= target_tokens:
            raise ValueError("overlap must be < target_tokens")
        self.target_tokens = int(target_tokens)
        self.overlap = int(overlap)
        self.encoding = encoding
        self._tok = get_tokenizer(encoding)

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []

        ids = self._tok.encode(text)
        if not ids:
            return []

        n_tokens = len(ids)
        n_chars = len(text)
        step = self.target_tokens - self.overlap
        chunks: list[Chunk] = []

        i = 0
        while i < n_tokens:
            tok_start = i
            tok_end = min(i + self.target_tokens, n_tokens)
            char_start = (tok_start * n_chars) // n_tokens
            char_end = (tok_end * n_chars) // n_tokens
            char_end = max(char_end, char_start + 1)
            piece = text[char_start:char_end]
            chunks.append(
                Chunk(
                    text=piece,
                    start=char_start,
                    end=char_end,
                    index=len(chunks),
                    token_count=tok_end - tok_start,
                    metadata={"strategy": self.name, "overlap": self.overlap},
                )
            )
            if tok_end >= n_tokens:
                break
            i += step

        return chunks
