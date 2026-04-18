"""Run a pipeline of extractors over records."""

from __future__ import annotations

from typing import Any, Iterable, Sequence, Union

from .._records import normalize_records
from .base import BaseExtractor, ExtractionResult


def _resolve_extractor(spec: Union[str, BaseExtractor, type]) -> BaseExtractor:
    if isinstance(spec, BaseExtractor):
        return spec
    if isinstance(spec, type) and issubclass(spec, BaseExtractor):
        return spec()
    if isinstance(spec, str):
        from . import REGISTRY

        if spec not in REGISTRY:
            raise ValueError(
                f"unknown extractor: {spec!r}; known: {sorted(REGISTRY)}"
            )
        return REGISTRY[spec]()
    raise TypeError(f"cannot resolve extractor from {spec!r}")


def run_extractors(
    text: str, extractors: Sequence[Union[str, BaseExtractor, type]]
) -> dict[str, list[ExtractionResult]]:
    """Apply each extractor to ``text``; return a {name: results} mapping."""
    out: dict[str, list[ExtractionResult]] = {}
    for spec in extractors:
        extractor = _resolve_extractor(spec)
        out[extractor.name] = extractor.extract(text)
    return out


def enrich_documents(
    docs: Union[str, dict, Iterable[Any]],
    extractors: Sequence[Union[str, BaseExtractor, type]],
) -> list[dict[str, Any]]:
    """Add ``metadata["extractions"][name] = [...]`` to each record.

    Returns a list of normalized record dicts (the input is normalized
    internally; callers may safely mutate the returned dicts).
    """
    if isinstance(docs, (str, dict)):
        items: list[Any] = [docs]
    else:
        items = list(docs)
    records = normalize_records(items)

    resolved = [_resolve_extractor(s) for s in extractors]

    for rec in records:
        bucket = rec["metadata"].setdefault("extractions", {})
        for extractor in resolved:
            bucket[extractor.name] = [
                r.to_dict() for r in extractor.extract(rec["text"])
            ]

    return records
