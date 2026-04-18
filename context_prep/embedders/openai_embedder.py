"""OpenAI embedder implementation.

Supports:
- text-embedding-3-small (1536 dims)
- text-embedding-3-large (3072 dims, 256-3072 configurable)
"""

from __future__ import annotations

import os
import time

from context_prep.embedders._env import load_embed_dotenv
from context_prep.embedders.base import BaseEmbedder

load_embed_dotenv()


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI's embedding API.

    Features:
    - Batch processing (up to 2048 texts per request)
    - Automatic retries with exponential backoff
    """

    SUPPORTED_MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, model_id: str, dimension: int, api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            ) from exc

        self.model_id = model_id
        self.dimension = dimension

        if model_id not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported OpenAI model: {model_id}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        default_dim = self.SUPPORTED_MODELS[model_id]
        if dimension > default_dim:
            raise ValueError(
                f"Dimension {dimension} exceeds maximum {default_dim} for model {model_id}"
            )

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str], max_retries: int = 3) -> tuple[list[list[float]], dict]:
        """Embed a batch of texts using the OpenAI API.

        Returns:
            ``(embeddings, usage_info)`` — usage_info has
            ``prompt_tokens`` and ``total_tokens``.
        """
        if not texts:
            return [], {"prompt_tokens": 0, "total_tokens": 0}

        batch_size = min(len(texts), 2048)

        for attempt in range(max_retries):
            try:
                if self.dimension == self.SUPPORTED_MODELS[self.model_id]:
                    response = self.client.embeddings.create(
                        model=self.model_id,
                        input=texts[:batch_size],
                        encoding_format="float",
                    )
                else:
                    response = self.client.embeddings.create(
                        model=self.model_id,
                        input=texts[:batch_size],
                        dimensions=self.dimension,
                        encoding_format="float",
                    )

                embeddings: list = [None] * len(response.data)
                for item in response.data:
                    embeddings[item.index] = item.embedding

                usage_info = {
                    "prompt_tokens": response.usage.prompt_tokens
                    if hasattr(response, "usage")
                    else 0,
                    "total_tokens": response.usage.total_tokens
                    if hasattr(response, "usage")
                    else 0,
                }

                return embeddings, usage_info

            except Exception as exc:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(
                        f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {exc}; "
                        f"retrying in {wait_time}s"
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"OpenAI API failed after {max_retries} attempts: {exc}"
                    ) from exc

        raise RuntimeError("unreachable")

    def get_model_name(self) -> str:
        return self.model_id

    def get_dimension(self) -> int:
        return self.dimension

    def get_batch_size(self) -> int:
        return 100
