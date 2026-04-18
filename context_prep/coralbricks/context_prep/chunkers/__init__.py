"""Text chunking algorithms for Coral Bricks prep.

Public surface
--------------

* ``Chunk`` -- dataclass returned by every chunker.
* ``BaseChunker`` -- abstract base.
* Concrete chunkers:

  - ``FixedTokenChunker`` -- non-overlapping token windows.
  - ``SlidingTokenChunker`` -- overlapping token windows (RAG default).
  - ``RecursiveCharacterChunker`` -- LangChain-style hierarchical splitter.
  - ``SentenceChunker`` -- regex sentence boundary + budget-bound aggregation.

* ``chunk_text(text, *, strategy="sliding_token", **kwargs)`` --
  one-shot dispatcher. ``strategy`` accepts a string name or a
  ``BaseChunker`` instance (in which case kwargs are ignored).

* ``count_tokens(text, encoding="cl100k_base")`` -- shared helper.

These chunkers are intentionally dependency-light. ``tiktoken`` is used
when available for accurate token counts; otherwise we fall back to a
whitespace tokenizer that keeps boundaries deterministic but makes
``token_count`` an approximation.
"""

from __future__ import annotations

from typing import Any

from coralbricks.context_prep.chunkers._tokens import count_tokens, get_tokenizer
from coralbricks.context_prep.chunkers.base import BaseChunker, Chunk
from coralbricks.context_prep.chunkers.fixed_token import FixedTokenChunker
from coralbricks.context_prep.chunkers.recursive_character import (
    DEFAULT_SEPARATORS,
    RecursiveCharacterChunker,
)
from coralbricks.context_prep.chunkers.sentence import SentenceChunker
from coralbricks.context_prep.chunkers.sliding_token import SlidingTokenChunker

__all__ = [
    "Chunk",
    "BaseChunker",
    "FixedTokenChunker",
    "SlidingTokenChunker",
    "RecursiveCharacterChunker",
    "SentenceChunker",
    "DEFAULT_SEPARATORS",
    "chunk_text",
    "make_chunker",
    "count_tokens",
    "get_tokenizer",
]


_STRATEGIES: dict[str, type[BaseChunker]] = {
    "fixed_token": FixedTokenChunker,
    "sliding_token": SlidingTokenChunker,
    "recursive_character": RecursiveCharacterChunker,
    "sentence": SentenceChunker,
    # Friendly aliases.
    "semantic": SlidingTokenChunker,
    "sliding": SlidingTokenChunker,
    "fixed": FixedTokenChunker,
    "recursive": RecursiveCharacterChunker,
}


def make_chunker(strategy: str | BaseChunker, **kwargs: Any) -> BaseChunker:
    """Resolve ``strategy`` (string name or instance) into a ``BaseChunker``."""
    if isinstance(strategy, BaseChunker):
        if kwargs:
            raise TypeError("kwargs not allowed when strategy is already a BaseChunker instance")
        return strategy
    try:
        cls = _STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError(
            f"unknown chunking strategy {strategy!r}; choose one of {sorted(set(_STRATEGIES))}"
        ) from exc
    return cls(**kwargs)


def chunk_text(
    text: str,
    *,
    strategy: str | BaseChunker = "sliding_token",
    **kwargs: Any,
) -> list[Chunk]:
    """Chunk a single ``text`` according to ``strategy``.

    Examples::

        from coralbricks.context_prep.chunkers import chunk_text

        # 512-token sliding window with 64-token overlap (default).
        chunks = chunk_text(big_doc)

        # Recursive character splitter (LangChain-style).
        chunks = chunk_text(big_doc, strategy="recursive_character",
                            chunk_size=1000, chunk_overlap=100)

        # Sentence-aggregated, ~256-token chunks.
        chunks = chunk_text(big_doc, strategy="sentence", target_tokens=256)
    """
    chunker = make_chunker(strategy, **kwargs)
    return chunker.chunk(text)
