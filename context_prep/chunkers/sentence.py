"""Sentence chunker.

Splits text into sentences using a regex with common abbreviation
guards, then aggregates consecutive sentences up to ``target_tokens``
(or ``target_chars`` when token counts are unavailable).
"""

from __future__ import annotations

import re

from context_prep.chunkers._tokens import count_tokens
from context_prep.chunkers.base import BaseChunker, Chunk

# Sentence boundaries: end-punctuation followed by whitespace + capital,
# guarded against common abbreviations.
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "st",
    "jr",
    "sr",
    "vs",
    "etc",
    "i.e",
    "e.g",
    "u.s",
    "u.k",
    "no",
    "fig",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return list of ``(start, end, text)`` sentence spans."""
    spans: list[tuple[int, int, str]] = []
    last = 0
    for match in _SENT_SPLIT.finditer(text):
        end = match.start()
        candidate = text[last:end].rstrip()
        if not candidate:
            last = match.end()
            continue
        # Abbreviation guard: if the token before the period is a known
        # abbreviation, glue this sentence to the next one.
        token = candidate.split()[-1].rstrip(".").lower()
        if token in _ABBREVIATIONS and len(spans) > 0:
            prev_start, _, prev_text = spans[-1]
            spans[-1] = (prev_start, end, prev_text + " " + candidate)
        else:
            spans.append((last, end, candidate))
        last = match.end()
    tail = text[last:].rstrip()
    if tail:
        spans.append((last, last + len(tail), tail))
    return spans


class SentenceChunker(BaseChunker):
    name = "sentence"

    def __init__(
        self,
        *,
        target_tokens: int | None = 256,
        target_chars: int | None = None,
        encoding: str = "cl100k_base",
    ):
        if target_tokens is None and target_chars is None:
            raise ValueError("provide target_tokens or target_chars")
        if target_tokens is not None and target_tokens <= 0:
            raise ValueError("target_tokens must be > 0")
        if target_chars is not None and target_chars <= 0:
            raise ValueError("target_chars must be > 0")
        self.target_tokens = target_tokens
        self.target_chars = target_chars
        self.encoding = encoding

    def _measure(self, text: str) -> int:
        if self.target_tokens is not None:
            return count_tokens(text, self.encoding)
        return len(text)

    def _budget(self) -> int:
        return self.target_tokens if self.target_tokens is not None else int(self.target_chars or 0)

    def chunk(self, text: str) -> list[Chunk]:
        if not text:
            return []

        sentences = _split_sentences(text)
        if not sentences:
            return []

        budget = self._budget()
        chunks: list[Chunk] = []
        buf: list[tuple[int, int, str]] = []
        buf_size = 0

        def _flush() -> None:
            nonlocal buf, buf_size
            if not buf:
                return
            start = buf[0][0]
            end = buf[-1][1]
            piece = " ".join(s[2] for s in buf)
            chunks.append(
                Chunk(
                    text=piece,
                    start=start,
                    end=end,
                    index=len(chunks),
                    token_count=self._measure(piece) if self.target_tokens else None,
                    metadata={"strategy": self.name, "sentences": len(buf)},
                )
            )
            buf = []
            buf_size = 0

        for span in sentences:
            size = self._measure(span[2])
            if buf and (buf_size + size) > budget:
                _flush()
            buf.append(span)
            buf_size += size

        _flush()
        return chunks
