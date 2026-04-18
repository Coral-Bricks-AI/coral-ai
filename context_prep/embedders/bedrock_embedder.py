"""Bedrock embedder implementation.

Supports AWS Bedrock embedding models:
- Cohere Embed (v3, v4)
- AWS Titan Embed (v1, v2)
"""

from __future__ import annotations

import json
import time

from context_prep.embedders.base import BaseEmbedder


class BedrockEmbedder(BaseEmbedder):
    """Embedder using AWS Bedrock InvokeModel API.

    Supports:
    - cohere.embed-english-v3 (1024 dim, batch: 96)
    - cohere.embed-v4:0 (1536 dim, batch: 96)
    - amazon.titan-embed-text-v2:0 (1024 dim, parallel: 100)
    """

    SUPPORTED_MODELS = {
        "cohere.embed-english-v3": 1024,
        "cohere.embed-v4:0": 1536,
        "amazon.titan-embed-text-v2:0": 1024,
    }

    CROSS_REGION_PROFILES = {
        "cohere.embed-v4:0": "us.cohere.embed-v4:0",
    }

    def __init__(
        self,
        model_id: str,
        task: str | None = None,
        aws_region: str = "us-east-1",
        use_cross_region: bool = True,
    ):
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 package not installed. Install with: pip install boto3"
            ) from exc

        self.model_id = model_id
        self.aws_region = aws_region
        self.use_cross_region = use_cross_region

        if model_id not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported Bedrock model: {model_id}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.dimension = self.SUPPORTED_MODELS[model_id]

        self.task = task or "search_document"
        if self.model_id.startswith("cohere.") and self.task not in (
            "search_query",
            "search_document",
        ):
            raise ValueError(
                f"Invalid task '{self.task}' for Cohere model. "
                "Must be 'search_query' or 'search_document'."
            )

        if use_cross_region and model_id in self.CROSS_REGION_PROFILES:
            self.inference_model_id = self.CROSS_REGION_PROFILES[model_id]
        else:
            self.inference_model_id = model_id

        config = boto3.session.Config(
            read_timeout=300,
            connect_timeout=60,
            retries={"max_attempts": 3, "mode": "adaptive"},
            max_pool_connections=100,
        )
        self.client = boto3.client("bedrock-runtime", region_name=aws_region, config=config)

    def embed_texts(self, texts: list[str], max_retries: int = 3) -> tuple[list[list[float]], dict]:
        if not texts:
            return [], {"prompt_tokens": 0, "total_tokens": 0}

        max_batch_size = 100 if self.model_id.startswith("amazon.titan-embed-text-v2") else 96

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), max_batch_size):
            batch = texts[i : i + max_batch_size]
            for attempt in range(max_retries):
                try:
                    batch_embeddings = self._embed_batch(batch)
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as exc:
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        print(
                            f"Bedrock API error (attempt {attempt + 1}/{max_retries}): {exc}; "
                            f"retrying in {wait_time}s"
                        )
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(
                            f"Bedrock API failed after {max_retries} attempts: {exc}"
                        ) from exc

        return all_embeddings, {"prompt_tokens": 0, "total_tokens": 0}

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.model_id.startswith("amazon.titan-embed-text-v2"):
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def embed_single(text: str) -> list[float]:
                payload = {
                    "inputText": text,
                    "dimensions": self.dimension,
                    "normalize": True,
                }
                request_body = json.dumps(payload)
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=request_body,
                    contentType="application/json",
                    accept="application/json",
                )
                response_body = json.loads(response["body"].read())
                return response_body["embedding"]

            all_embeddings: list = [None] * len(texts)
            with ThreadPoolExecutor(max_workers=100) as executor:
                future_to_idx = {
                    executor.submit(embed_single, text): idx for idx, text in enumerate(texts)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    all_embeddings[idx] = future.result()
            return all_embeddings

        payload = {
            "texts": texts,
            "input_type": self.task or "search_document",
            "truncate": "END",
        }

        if "v4" in self.model_id:
            payload["embedding_types"] = ["float"]

        request_body = json.dumps(payload)

        response = self.client.invoke_model(
            modelId=self.inference_model_id,
            body=request_body,
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read())

        if "v4" in self.model_id:
            embeddings = response_body["embeddings"]["float"]
        else:
            embeddings = response_body["embeddings"]

        return [emb[: self.dimension] for emb in embeddings]

    def get_model_name(self) -> str:
        return self.model_id

    def get_dimension(self) -> int:
        return self.dimension

    def get_batch_size(self) -> int:
        if self.model_id.startswith("amazon.titan-embed-text-v2"):
            return 100
        return 96
