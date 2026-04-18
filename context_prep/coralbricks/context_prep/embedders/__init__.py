"""Multi-provider embedders for Coral Bricks prep.

Single entry point: ``create_embedder(model_id, dimension, ...)``.

Concrete embedders are exposed for callers that want to construct one
directly. Heavy embedders (sentence-transformers + torch) are imported
lazily by the factory.

"""

from coralbricks.context_prep.embedders.base import BaseEmbedder
from coralbricks.context_prep.embedders.embedder_factory import (
    create_embedder,
    list_supported_models,
)
from coralbricks.context_prep.embedders.parquet import write_vectors_parquet

__all__ = [
    "BaseEmbedder",
    "OpenAIEmbedder",
    "CoralEmbedder",
    "CoralGatewayEmbedder",
    "BedrockEmbedder",
    "DeepInfraEmbedder",
    "SentenceTransformerEmbedder",
    "create_embedder",
    "list_supported_models",
    "write_vectors_parquet",
]


# Concrete embedders are imported lazily so callers that don't need them
# don't pay the import-time cost of e.g. boto3 / sentence-transformers / torch.
_LAZY = {
    "OpenAIEmbedder": ("coralbricks.context_prep.embedders.openai_embedder", "OpenAIEmbedder"),
    "CoralEmbedder": ("coralbricks.context_prep.embedders.coral_embedder", "CoralEmbedder"),
    "CoralGatewayEmbedder": (
        "coralbricks.context_prep.embedders.coral_gateway_embedder",
        "CoralGatewayEmbedder",
    ),
    "BedrockEmbedder": (
        "coralbricks.context_prep.embedders.bedrock_embedder",
        "BedrockEmbedder",
    ),
    "DeepInfraEmbedder": (
        "coralbricks.context_prep.embedders.deepinfra_embedder",
        "DeepInfraEmbedder",
    ),
    "SentenceTransformerEmbedder": (
        "coralbricks.context_prep.embedders.sentence_transformer_embedder",
        "SentenceTransformerEmbedder",
    ),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_name, attr_name = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'coralbricks.context_prep.embedders' has no attribute {name!r}")
