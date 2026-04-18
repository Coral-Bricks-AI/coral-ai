"""Coral embedder implementation using OpenAI SDK.

Calls our custom inference server running on GPU via the
OpenAI-compatible ``/v1/embeddings`` endpoint.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from coralbricks.context_prep.embedders._env import load_embed_dotenv
from coralbricks.context_prep.embedders.base import BaseEmbedder

_log = logging.getLogger(__name__)
load_embed_dotenv()


class CoralEmbedder(BaseEmbedder):
    """Embedder using Coral inference server with OpenAI-compatible API.

    Features:
    - Uses OpenAI SDK for compatibility
    - Batch processing via /v1/embeddings endpoint
    - Automatic task routing (query / product / document)
    - Returns float arrays directly (no base64 decoding needed)
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        batch_size: int = 32,
        input_type: str = "product",
        api_key: str | None = None,
        request_timeout: float = 90.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            ) from exc

        self.endpoint_url = endpoint_url or os.getenv("CORAL_INFERENCE_URL")
        if not self.endpoint_url:
            raise ValueError(
                "Coral inference endpoint URL not provided. Set CORAL_INFERENCE_URL "
                "environment variable or pass endpoint_url parameter."
            )

        self.endpoint_url = self.endpoint_url.rstrip("/")
        self.batch_size = batch_size
        self._model_info: dict | None = None

        if input_type not in ("query", "product", "document"):
            raise ValueError(
                f"Invalid input_type '{input_type}'. Must be 'query', 'product', or 'document'."
            )

        self.input_type = input_type
        self.task = input_type

        self.api_key = api_key or os.getenv("CORAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Coral API key not provided. Set CORAL_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=f"{self.endpoint_url}/v1",
            timeout=request_timeout,
        )

        self._check_health()

    def _normalize_model_info(self, data: dict) -> None:
        """Accept both legacy Coral /health and search-saas-gateway shapes."""
        if "model_name" not in data and data.get("model"):
            m = str(data["model"]).replace("_", "-")
            data = {**data, "model_name": m}
        self._model_info = data

    def _check_health(self) -> None:
        import requests

        try:
            response = requests.get(f"{self.endpoint_url}/health", timeout=5)
            response.raise_for_status()
            self._normalize_model_info(response.json())
            _log.info(
                "Connected to OpenAI-compatible embed server model=%s dim=%s task=%s",
                self._model_info.get("model_name", "?"),
                self._model_info.get("dimension", "?"),
                self.task,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to Coral inference server at {self.endpoint_url}/health: {exc}"
            ) from exc

    def embed_texts(
        self,
        texts: list[str],
        max_retries: int = 3,
        **_: Any,
    ) -> tuple[list[list[float]], dict]:
        if not texts:
            return [], {"prompt_tokens": 0, "total_tokens": 0}

        import time

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.get_model_name(),
                    input=texts,
                    extra_body={"task": self.task},
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
                        f"Coral API error (attempt {attempt + 1}/{max_retries}): {exc}; "
                        f"retrying in {wait_time}s"
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Coral API failed after {max_retries} attempts: {exc}"
                    ) from exc

        raise RuntimeError("unreachable")

    def get_model_name(self) -> str:
        if self._model_info:
            return self._model_info.get("model_name", "coral_embed")
        return "coral_embed"

    def get_dimension(self) -> int:
        if self._model_info:
            return self._model_info.get("dimension", 768)
        return 768

    def get_batch_size(self) -> int:
        return self.batch_size
