"""Base embedder interface.

All embedders must implement this interface.

Moved here from ``www/ml/embed/embedders/base.py`` so that the embedder
factory lives next to the ``coralbricks.context_prep`` DSL it powers. The old
location is preserved as a thin shim for back-compat.
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for all embedders."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each vector is a list of floats).
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier / name."""

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""

    def get_batch_size(self) -> int:
        """Recommended batch size (default: 32)."""
        return 32
