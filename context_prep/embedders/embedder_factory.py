"""Factory for creating embedders based on model id.

Multi-provider entry point for the prep DSL. Supports:
- Coral inference server (model_id == ``"coral_embed"``)
- Coral Gateway production API (model_id == ``"coral_gateway"``)
- OpenAI embedding models
- AWS Bedrock embedding models (Cohere v3/v4, Titan v2)
- Sentence-transformer benchmark models (``st:<name>``)
- DeepInfra hosted models (``di:<name>``)
"""

from __future__ import annotations

from typing import Any

from context_prep.embedders.base import BaseEmbedder
from context_prep.embedders.coral_embedder import CoralEmbedder
from context_prep.embedders.coral_gateway_embedder import CoralGatewayEmbedder
from context_prep.embedders.deepinfra_embedder import (
    MODELS as DI_MODELS,
)
from context_prep.embedders.deepinfra_embedder import (
    DeepInfraEmbedder,
)
from context_prep.embedders.openai_embedder import OpenAIEmbedder

# Heavy backends (boto3 / sentence-transformers + torch) are imported on
# demand so callers that only need API-backed embedders don't pay the cost.
_ST_MODULE: Any | None = None
_BEDROCK_MODULE: Any | None = None


def _load_st():
    global _ST_MODULE
    if _ST_MODULE is None:
        from context_prep.embedders import sentence_transformer_embedder as st

        _ST_MODULE = st
    return _ST_MODULE


def _load_bedrock():
    global _BEDROCK_MODULE
    if _BEDROCK_MODULE is None:
        from context_prep.embedders import bedrock_embedder as br

        _BEDROCK_MODULE = br
    return _BEDROCK_MODULE


# Bedrock model id table is consulted by `create_embedder` for routing,
# without forcing boto3 import. Keep in sync with `BedrockEmbedder.SUPPORTED_MODELS`.
_BEDROCK_SUPPORTED_MODEL_IDS: tuple[str, ...] = (
    "cohere.embed-english-v3",
    "cohere.embed-v4:0",
    "amazon.titan-embed-text-v2:0",
)


def create_embedder(
    model_id: str,
    dimension: int,
    input_type: str = "product",
    coral_endpoint: str | None = None,
    coral_api_key: str | None = None,
    openai_api_key: str | None = None,
    aws_region: str = "us-east-1",
    batch_size: int = 32,
    coral_request_timeout: float | None = None,
    device: str | None = None,
    use_fp16: bool = False,
    tokenize_processes: int = 0,
    max_seq_length: int | None = None,
    **kwargs: Any,
) -> BaseEmbedder:
    """Create an embedder for ``model_id``.

    Routing rules:
    - ``di:<name>``  -> DeepInfra
    - ``st:<name>``  -> SentenceTransformer
    - ``coral_embed``   -> direct Coral inference server
    - ``coral_gateway`` -> Coral Gateway (production API)
    - OpenAI / Bedrock model ids -> respective providers

    See examples in the legacy module docstring (
    ``www/ml/embed/embedders/embedder_factory.py``) for full call shapes.
    """
    if model_id.startswith("di:"):
        di_model_name = model_id[3:]
        return DeepInfraEmbedder(
            model_name=di_model_name,
            batch_size=batch_size,
            input_type=input_type,
        )

    if model_id.startswith("st:"):
        st_model_name = model_id[3:]
        if kwargs:
            raise TypeError(
                f"create_embedder: unexpected kwargs for sentence-transformer: {sorted(kwargs)}"
            )
        st = _load_st()
        return st.SentenceTransformerEmbedder(
            model_name=st_model_name,
            input_type=input_type,
            batch_size=batch_size,
            device=device,
            use_fp16=use_fp16,
            tokenize_processes=tokenize_processes,
            max_seq_length=max_seq_length,
        )

    if model_id == "coral_embed":
        return CoralEmbedder(
            endpoint_url=coral_endpoint,
            batch_size=batch_size,
            input_type=input_type,
            api_key=coral_api_key,
            request_timeout=coral_request_timeout or 90.0,
        )

    if model_id == "coral_gateway":
        return CoralGatewayEmbedder(
            api_key=coral_api_key,
            endpoint_url=coral_endpoint or "https://api.coralbricks.ai",
            input_type=input_type,
            batch_size=batch_size,
        )

    if model_id in OpenAIEmbedder.SUPPORTED_MODELS:
        return OpenAIEmbedder(
            model_id=model_id,
            dimension=dimension,
            api_key=openai_api_key,
        )

    if model_id in _BEDROCK_SUPPORTED_MODEL_IDS:
        task = None
        if "cohere" in model_id.lower():
            task = "search_query" if input_type == "query" else "search_document"

        br = _load_bedrock()
        return br.BedrockEmbedder(
            model_id=model_id,
            aws_region=aws_region,
            task=task,
        )

    raise ValueError(
        f"Unknown model_id: {model_id}. Supported models:\n"
        f"  - coral_embed (direct inference server)\n"
        f"  - coral_gateway (production API)\n"
        f"  - OpenAI: {list(OpenAIEmbedder.SUPPORTED_MODELS.keys())}\n"
        f"  - Bedrock: {list(_BEDROCK_SUPPORTED_MODEL_IDS)}\n"
        f"  - SentenceTransformer (st:<name>): see prep.embedders.sentence_transformer_embedder.MODELS\n"
        f"  - DeepInfra (di:<name>): {list(DI_MODELS.keys())}"
    )


def list_supported_models() -> dict:
    """Return a directory of supported models by provider with dims."""
    st = _load_st()
    br = _load_bedrock()
    return {
        "coral": {"coral_embed": "Variable (from inference server)"},
        "openai": {model: f"{dim} dims" for model, dim in OpenAIEmbedder.SUPPORTED_MODELS.items()},
        "bedrock": {
            model: f"{dim} dims" for model, dim in br.BedrockEmbedder.SUPPORTED_MODELS.items()
        },
        "sentence_transformers": {
            f"st:{name}": f"{info['dimension']} dims" for name, info in st.MODELS.items()
        },
        "deepinfra": {
            f"di:{name}": f"{info['dimension']} dims" for name, info in DI_MODELS.items()
        },
    }
