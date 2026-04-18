"""Document enrichment: extract structured signals from text.

Two flavours:

* **Regex extractors** (zero deps): tickers, URLs, emails, dates, money,
  hashtags, mentions. Reasonable defaults; deterministic and fast.
* **spaCy adapter** (lazy, optional): named-entity recognition via a
  user-supplied spaCy pipeline.

Use :func:`enrich_documents` to run a list of extractors over a list
of dict records. Extracted values land in
``record["metadata"]["extractions"][extractor_name]``.
"""

from __future__ import annotations

from .base import BaseExtractor, ExtractionResult
from .pipeline import enrich_documents, run_extractors
from .regex_extractors import (
    DateExtractor,
    EmailExtractor,
    HashtagExtractor,
    MentionExtractor,
    MoneyExtractor,
    PhoneExtractor,
    TickerExtractor,
    URLExtractor,
)
from .spacy_extractor import SpacyEntityExtractor

REGISTRY: dict[str, type[BaseExtractor]] = {
    "tickers": TickerExtractor,
    "urls": URLExtractor,
    "emails": EmailExtractor,
    "dates": DateExtractor,
    "money": MoneyExtractor,
    "phones": PhoneExtractor,
    "hashtags": HashtagExtractor,
    "mentions": MentionExtractor,
}


__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "REGISTRY",
    "enrich_documents",
    "run_extractors",
    "TickerExtractor",
    "URLExtractor",
    "EmailExtractor",
    "DateExtractor",
    "MoneyExtractor",
    "PhoneExtractor",
    "HashtagExtractor",
    "MentionExtractor",
    "SpacyEntityExtractor",
]
