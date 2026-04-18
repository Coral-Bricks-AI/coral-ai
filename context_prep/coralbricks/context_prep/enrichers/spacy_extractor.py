"""spaCy NER adapter (lazy)."""

from __future__ import annotations

from typing import Any, Iterable

from .base import BaseExtractor, ExtractionResult


class SpacyEntityExtractor(BaseExtractor):
    """Wrap a spaCy ``Language`` pipeline as a CB extractor.

    Either pass a pre-loaded ``nlp`` instance (recommended for batches),
    or a model name like ``en_core_web_sm`` and the extractor will lazy
    load it on first use.

    ``labels`` filters which entity labels (e.g. ``{"ORG", "PERSON"}``)
    are returned. Defaults to all labels emitted by the pipeline.
    """

    name = "spacy_entities"

    def __init__(
        self,
        *,
        nlp: Any | None = None,
        model: str = "en_core_web_sm",
        labels: Iterable[str] | None = None,
    ):
        self._nlp = nlp
        self._model = model
        self._labels = {l.upper() for l in labels} if labels else None

    def _ensure_nlp(self) -> Any:
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "SpacyEntityExtractor requires spaCy. Install with `pip install spacy` "
                "and download a model, e.g. `python -m spacy download en_core_web_sm`."
            ) from exc
        try:
            self._nlp = spacy.load(self._model)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model {self._model!r} is not installed. "
                f"Run `python -m spacy download {self._model}`."
            ) from exc
        return self._nlp

    def extract(self, text: str) -> list[ExtractionResult]:
        nlp = self._ensure_nlp()
        doc = nlp(text)
        out: list[ExtractionResult] = []
        for ent in doc.ents:
            if self._labels and ent.label_.upper() not in self._labels:
                continue
            out.append(
                ExtractionResult(
                    value=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    label=ent.label_,
                )
            )
        return out

    def extract_many(self, texts: Iterable[str]) -> list[list[ExtractionResult]]:
        nlp = self._ensure_nlp()
        results: list[list[ExtractionResult]] = []
        for doc in nlp.pipe(list(texts)):
            entities: list[ExtractionResult] = []
            for ent in doc.ents:
                if self._labels and ent.label_.upper() not in self._labels:
                    continue
                entities.append(
                    ExtractionResult(
                        value=ent.text,
                        start=ent.start_char,
                        end=ent.end_char,
                        label=ent.label_,
                    )
                )
            results.append(entities)
        return results
