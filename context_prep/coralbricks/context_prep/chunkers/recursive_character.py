"""Recursive character chunker.

Splits text by a hierarchy of separators, falling back from coarse to
fine boundaries until each piece fits within ``chunk_size``.
This is a clean re-implementation of the LangChain
``RecursiveCharacterTextSplitter`` algorithm without the dependency.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from coralbricks.context_prep.chunkers.base import BaseChunker, Chunk

DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
)


class RecursiveCharacterChunker(BaseChunker):
    name = "recursive_character"

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: Optional[Sequence[str]] = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.separators: tuple[str, ...] = tuple(separators or DEFAULT_SEPARATORS)

    def chunk(self, text: str) -> List[Chunk]:
        if not text:
            return []
        pieces = self._split_recursive(text, list(self.separators))
        merged = self._merge_with_overlap(pieces)

        chunks: List[Chunk] = []
        cursor = 0
        for piece in merged:
            start = text.find(piece, cursor)
            if start == -1:
                start = cursor
            end = start + len(piece)
            chunks.append(
                Chunk(
                    text=piece,
                    start=start,
                    end=end,
                    index=len(chunks),
                    metadata={
                        "strategy": self.name,
                        "chunk_overlap": self.chunk_overlap,
                    },
                )
            )
            cursor = max(0, end - self.chunk_overlap)
        return chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []
        # Pick the first separator that actually appears.
        sep = ""
        rest: List[str] = []
        for idx, candidate in enumerate(separators):
            if candidate == "":
                sep = ""
                rest = []
                break
            if candidate in text:
                sep = candidate
                rest = separators[idx + 1 :]
                break
        else:
            sep = ""
            rest = []

        if sep == "":
            # Hard slice into chunk-size windows.
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        splits = text.split(sep)
        good: List[str] = []
        for piece in splits:
            if not piece:
                continue
            piece_with_sep = piece + sep if piece is not splits[-1] else piece
            if len(piece_with_sep) <= self.chunk_size:
                good.append(piece_with_sep)
            else:
                # Recurse with the remaining (finer) separators.
                good.extend(self._split_recursive(piece_with_sep, rest or [""]))
        return good

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        merged: List[str] = []
        current = ""
        for piece in pieces:
            if not piece:
                continue
            if not current:
                current = piece
                continue
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
                continue

            merged.append(current)
            tail = current[-self.chunk_overlap :] if self.chunk_overlap else ""
            # Avoid infinite growth: if a single piece is already bigger
            # than chunk_size (after recursive split fell through), emit
            # it standalone without an overlap header.
            if len(piece) > self.chunk_size:
                merged.append(piece)
                current = ""
            else:
                current = (tail + piece) if tail else piece
        if current:
            merged.append(current)
        return merged
