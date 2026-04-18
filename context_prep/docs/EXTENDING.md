# Extending coralbricks-context-prep

You can plug in your own chunkers, extractors, embedders, and triple
extractors without forking the library.

## Custom chunker

```python
from coralbricks.context_prep.chunkers import BaseChunker, Chunk

class FirstSentenceChunker(BaseChunker):
    name = "first_sentence"

    def chunk(self, text: str) -> list[Chunk]:
        # ... your logic ...
        return [Chunk(text=first, start=0, end=len(first), token_count=len(first.split()))]

# Use it directly with chunk():
from coralbricks.context_prep import chunk
art = chunk(records, strategy=FirstSentenceChunker())
```

## Custom extractor

```python
from coralbricks.context_prep.enrichers import BaseExtractor, ExtractionResult

class LawCitationExtractor(BaseExtractor):
    name = "law_citations"

    def extract(self, text: str) -> list[ExtractionResult]:
        # ... your regex / model ...
        return [ExtractionResult(value="42 U.S.C. § 1983", start=10, end=27, label="USC")]

from coralbricks.context_prep import enrich
art = enrich(records, extractors=[LawCitationExtractor()])
```

## Custom embedder

```python
from coralbricks.context_prep.embedders import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def __init__(self, dim=768):
        self._dim = dim

    def embed_texts(self, texts):
        vectors = [...]               # call your service
        usage = {"prompt_tokens": 0}
        return vectors, usage

    def get_model_name(self): return "my-embedder"
    def get_dimension(self):  return self._dim

from coralbricks.context_prep import embed
art = embed(chunks, embedder=MyEmbedder())
```

## Custom triple extractor (for graphs)

```python
from coralbricks.context_prep.graph import BaseTripleExtractor, Triple

class CitationTripleExtractor(BaseTripleExtractor):
    name = "citations"

    def extract(self, record):
        return [
            Triple(
                subject_label="Document", subject_value=record["id"],
                predicate="cites",
                object_label="Case", object_value="Roe v. Wade",
            ),
        ]

from coralbricks.context_prep import hydrate
graph = hydrate(records, graph="cases", extractors=[CitationTripleExtractor()])
```

## Registering a new embedder backend in the factory

If you'd like your backend to be addressable by string id, send a PR
extending `embedder_factory.py` — keep the import lazy so users
without your backend installed don't pay the cost.
