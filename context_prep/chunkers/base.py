"""Base chunker interface + ``Chunk`` dataclass.

A chunker takes a single text and returns an ordered list of ``Chunk``s.
Every chunk records its byte offsets back into the source text so
downstream verbs (embed, hydrate) can reconstruct context windows or
attribute hits to source spans.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single chunk of a source document.

    Attributes:
        text: The chunk's textual content.
        start: Inclusive character offset in the source text.
        end: Exclusive character offset in the source text.
        index: 0-based position within the chunked output.
        token_count: Optional token count (set by token-aware chunkers).
        metadata: Free-form per-chunk metadata (e.g. section heading).
    """

    text: str
    start: int
    end: int
    index: int
    token_count: int | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "index": self.index,
            "token_count": self.token_count,
            "metadata": dict(self.metadata),
        }


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""

    name: str = "base"

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """Split ``text`` into an ordered list of ``Chunk``s."""

    def chunk_many(self, texts: list[str]) -> list[list[Chunk]]:
        """Convenience: chunk a batch of texts independently."""
        return [self.chunk(t) for t in texts]
