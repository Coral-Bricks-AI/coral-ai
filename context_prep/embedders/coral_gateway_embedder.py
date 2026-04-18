"""Coral Gateway embedder implementation.

For the production API gateway at ``api.coralbricks.ai`` which uses
the OpenAI-compatible ``/v1/embeddings`` endpoint with API-key auth.
"""

from __future__ import annotations

import os

from context_prep.embedders.base import BaseEmbedder


class CoralGatewayEmbedder(BaseEmbedder):
    """Embedder using Coral Gateway API at ``api.coralbricks.ai``.

    Features:
    - Uses /v1/embeddings endpoint (OpenAI-compatible)
    - Requires API key authentication
    - Single-text embedding per request (gateway-side batching)
    - Returns float arrays directly
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint_url: str = "https://api.coralbricks.ai",
        input_type: str = "product",
        batch_size: int = 32,
    ):
        try:
            import requests  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "requests package not installed. Install with: pip install requests"
            ) from exc

        self.api_key = api_key or os.getenv("CORAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Coral API key not provided. Set CORAL_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.endpoint_url = endpoint_url.rstrip("/")
        self.batch_size = batch_size

        if input_type not in ("query", "product"):
            raise ValueError(f"Invalid input_type '{input_type}'. Must be 'query' or 'product'.")

        self.input_type = input_type
        self.task = input_type

    def embed_texts(self, texts: list[str], max_retries: int = 3) -> tuple[list[list[float]], dict]:
        import time

        import requests

        if not texts:
            return [], {"prompt_tokens": 0, "total_tokens": 0}

        embeddings: list[list[float]] = []
        total_tokens = 0

        for text in texts:
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.endpoint_url}/v1/embeddings",
                        json={
                            "input": [text],
                            "model": "coral_embed",
                            "task": self.task,
                        },
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        timeout=30.0,
                    )

                    response.raise_for_status()
                    result = response.json()

                    if "data" in result and len(result["data"]) > 0:
                        embeddings.append(result["data"][0]["embedding"])

                        if "usage" in result and "total_tokens" in result["usage"]:
                            total_tokens += result["usage"]["total_tokens"]
                        else:
                            total_tokens += len(text.split())
                        break
                    raise RuntimeError(f"Invalid response format: {result}")

                except Exception as exc:
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        print(
                            f"Coral Gateway API error (attempt {attempt + 1}/{max_retries}): {exc}; "
                            f"retrying in {wait_time}s"
                        )
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(
                            f"Coral Gateway API failed after {max_retries} attempts: {exc}"
                        ) from exc

        return embeddings, {"prompt_tokens": total_tokens, "total_tokens": total_tokens}

    def get_model_name(self) -> str:
        return "coral_gateway"

    def get_dimension(self) -> int:
        return 768

    def get_batch_size(self) -> int:
        return self.batch_size
