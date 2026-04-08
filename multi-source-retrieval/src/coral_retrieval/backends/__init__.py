"""Retrieval backend adapters.

Each adapter implements :class:`RetrievalBackend` — a minimal protocol with
a single ``search`` method. Plug in any data source by implementing this
protocol; the orchestrator treats all backends uniformly.
"""

from .base import RetrievalBackend

__all__ = ["RetrievalBackend"]
