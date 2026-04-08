"""coral-retrieval — multi-source retrieval orchestration.

Wire up vector search, graph traversal, and SQL backends behind a single
retriever. Fuse results with RRF or Weighted Score Fusion.
"""

from .types import SearchHit, RetrievalResult
from .orchestrator import MultiSourceRetriever

__all__ = [
    "SearchHit",
    "RetrievalResult",
    "MultiSourceRetriever",
]
