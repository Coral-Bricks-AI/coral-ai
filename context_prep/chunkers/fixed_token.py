"""Fixed-token chunker.

Splits a text into non-overlapping windows of approximately
``target_tokens`` tokens each, walking by character offset to keep
``Chunk.start`` / ``Chunk.end`` aligned with the source text.
"""

from __future__ import annotations

from context_prep.chunkers._tokens import get_tokenizer
from context_prep.chunkers.base import BaseChunker, Chunk


class FixedTokenChunker(BaseChunker):
    name = "fixed_token"

    def __init__(self, *, target_tokens: int = 512, encoding: str = "cl100k_base"):
        if target_tokens <= 0:
            raise ValueError("target_tokens must be > 0")
        self.target_tokens = int(target_tokens)
        self.encoding = encoding
        self._tok = get_tokenizer(encoding)

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []

        ids = self._tok.encode(text)
        if not ids:
            return []

        # We can't always reverse-decode token spans to character spans
        # (especially with the whitespace fallback), so we walk character
        # offsets in proportion to the token spans. This is an
        # approximation but is monotonic and covers the whole text.
        n_tokens = len(ids)
        n_chars = len(text)
        chunks: list[Chunk] = []

        for i in range(0, n_tokens, self.target_tokens):
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
                    metadata={"strategy": self.name},
                )
            )

        return chunks
