"""DeepInfra embedder using the OpenAI-compatible embeddings API.

Supported models:
  - BAAI/bge-m3: BGE-M3 | 1024 dim | 8192 token context | $0.01/1M tokens

Uses ``DEEPINFRA_API_KEY`` env var for authentication.
"""

from __future__ import annotations

import logging
import os
import time

from context_prep.embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)

MODELS = {
    "bge-m3": {
        "api_model": "BAAI/bge-m3",
        "dimension": 1024,
        "description": (
            "BGE-M3 | BAAI | XLM-RoBERTa | 568M params | 1024 dim | "
            "multi-granularity retrieval, 100+ languages, 8192 token context. "
            "Via DeepInfra API ($0.01/1M tokens)."
        ),
    },
}

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


class DeepInfraEmbedder(BaseEmbedder):
    """Embedder using DeepInfra's OpenAI-compatible embeddings API.

    Batches texts into API calls with retry + exponential backoff.
    """

    def __init__(
        self,
        model_name: str = "bge-m3",
        api_key: str | None = None,
        batch_size: int = 512,
        input_type: str = "product",
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Unknown DeepInfra model: {model_name}. Supported: {list(MODELS.keys())}"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package required. Install with: pip install openai") from exc

        self.model_name = model_name
        self.model_info = MODELS[model_name]
        self.batch_size = batch_size
        self.input_type = input_type

        api_key = api_key or os.environ.get("DEEPINFRA_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPINFRA_API_KEY environment variable is required for DeepInfra embeddings"
            )

        self.client = OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)
        logger.info(
            "DeepInfraEmbedder: model=%s dim=%d batch_size=%d",
            self.model_info["api_model"],
            self.model_info["dimension"],
            self.batch_size,
        )

    def embed_texts(self, texts: list[str], max_retries: int = 5) -> tuple[list[list[float]], dict]:
        if not texts:
            return [], {"prompt_tokens": 0, "total_tokens": 0}

        total_prompt_tokens = 0
        total_tokens = 0
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embs, usage = self._call_api(batch, max_retries)
            all_embeddings.extend(embs)
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)

        return all_embeddings, {
            "prompt_tokens": total_prompt_tokens,
            "total_tokens": total_tokens,
        }

    def _call_api(self, texts: list[str], max_retries: int) -> tuple[list[list[float]], dict]:
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_info["api_model"],
                    input=texts,
                    encoding_format="float",
                )
                embeddings: list = [None] * len(response.data)
                for item in response.data:
                    embeddings[item.index] = item.embedding

                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }
                return embeddings, usage

            except Exception as exc:
                wait = min(2**attempt, 60)
                if attempt < max_retries - 1:
                    logger.warning(
                        "DeepInfra API error (attempt %d/%d): %s; retrying in %ds",
                        attempt + 1,
                        max_retries,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"DeepInfra API failed after {max_retries} attempts: {exc}"
                    ) from exc

        raise RuntimeError("unreachable")

    def get_model_name(self) -> str:
        return f"di:{self.model_name}"

    def get_dimension(self) -> int:
        return self.model_info["dimension"]

    def get_batch_size(self) -> int:
        return self.batch_size
