"""Tokenizer adapter for chunkers.

Token-aware chunkers prefer ``tiktoken`` (the OpenAI tokenizer used as
a generic proxy for transformer subword counts). When ``tiktoken`` isn't
installed we fall back to a whitespace tokenizer; this preserves
correctness of the chunk boundaries (we still chunk on character
positions) but makes ``token_count`` an approximation.
"""

from __future__ import annotations

from typing import List, Protocol


class _Tokenizer(Protocol):
    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...


_DEFAULT_ENCODING = "cl100k_base"
_CACHED: dict[str, _Tokenizer] = {}


def get_tokenizer(encoding: str = _DEFAULT_ENCODING) -> _Tokenizer:
    """Return a tokenizer for ``encoding``. Cached per process.

    Falls back to a whitespace tokenizer if ``tiktoken`` is unavailable.
    """
    cached = _CACHED.get(encoding)
    if cached is not None:
        return cached

    tokenizer: _Tokenizer
    try:
        import tiktoken

        try:
            tokenizer = tiktoken.get_encoding(encoding)
        except Exception:
            tokenizer = tiktoken.get_encoding(_DEFAULT_ENCODING)
    except Exception:
        tokenizer = _WhitespaceTokenizer()

    _CACHED[encoding] = tokenizer
    return tokenizer


class _WhitespaceTokenizer:
    """Cheap fallback tokenizer used when ``tiktoken`` isn't installed.

    Tokens are 1:1 with whitespace-separated words. Decoding round-trips
    by joining with single spaces, which is lossy for original whitespace
    but adequate for ``token_count`` estimation.
    """

    def encode(self, text: str) -> List[int]:
        return list(range(len(text.split())))

    def decode(self, tokens: List[int]) -> str:
        # Decoder is informational; chunker uses character offsets.
        return " ".join(["<tok>"] * len(tokens))


def count_tokens(text: str, encoding: str = _DEFAULT_ENCODING) -> int:
    return len(get_tokenizer(encoding).encode(text))
