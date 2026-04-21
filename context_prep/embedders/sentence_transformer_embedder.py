"""Sentence Transformer embedder for benchmarking open-source models.

Loads HuggingFace models for benchmarking with the
``sentence-transformers`` library. By default, models are pulled
straight from the HuggingFace Hub (so the OSS package works
out-of-the-box for anyone with HF access -- no AWS account, no
private bucket needed). An optional S3 mirror is supported for
operators who want to pin model versions in their own account or
avoid hitting the public Hub from an air-gapped environment.

S3 mirror activation (all-or-nothing -- set both, or neither):

- ``CORAL_BENCHMARK_MODELS_S3_BUCKET``: bucket name (e.g.
  ``"my-org-models"``).
- ``CORAL_BENCHMARK_MODELS_S3_PREFIX``: optional key prefix
  (default: ``"benchmark-models/"``).

Or pass ``s3_mirror_bucket=...`` directly to the constructor for
programmatic control. When the mirror is configured we shell out to
``aws s3 sync`` (so this path requires the AWS CLI on PATH plus
working credentials in the boto3 default chain). When it isn't
configured we let ``sentence-transformers`` handle the download via
its built-in HF Hub support, which uses ``~/.cache/huggingface/`` by
default.

Supported models:
  - bge-m3: BGE-M3 (BAAI) | XLM-RoBERTa | 568M params | 1024 dim
  - snowflake-arctic-embed-m-v2.0: Arctic Embed M v2.0 (Snowflake) | ModernBERT | 305M | 768 dim
  - nomic-embed-v2-moe: Nomic Embed v2 MoE (Nomic AI) | ModernBERT MoE | 305M | 768 dim
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)

from coralbricks.context_prep.embedders.base import BaseEmbedder

# Each entry carries:
# - ``hf_repo``: canonical HuggingFace repo id; used for the default
#   (and only OSS-friendly) load path.
# - ``s3_key``: subpath under the operator-configured mirror bucket;
#   only consulted when the S3 mirror is explicitly enabled.
# - dimension / prefix / description: model behaviour metadata.
MODELS = {
    "bge-m3": {
        "hf_repo": "BAAI/bge-m3",
        "s3_key": "benchmark-models/bge-m3/",
        "dimension": 1024,
        "query_prefix": "",
        "document_prefix": "",
        "description": (
            "BGE-M3 | BAAI | XLM-RoBERTa | 568M params | 1024 dim | Jan 2024. "
            "Known for: multi-granularity retrieval (dense + sparse + ColBERT in one model), "
            "100+ languages, 8192 token context"
        ),
    },
    "snowflake-arctic-embed-m-v2.0": {
        "hf_repo": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "s3_key": "benchmark-models/snowflake-arctic-embed-m-v2.0/",
        "dimension": 768,
        "query_prefix": "",
        "document_prefix": "",
        "description": (
            "Arctic Embed M v2.0 | Snowflake | ModernBERT | 305M params | 768 dim | Feb 2025. "
            "Known for: #1 MTEB retrieval at base size, prefix-free, strong zero-shot generalization"
        ),
    },
    "nomic-embed-v2-moe": {
        "hf_repo": "nomic-ai/nomic-embed-text-v2-moe",
        "s3_key": "benchmark-models/nomic-embed-v2-moe/",
        "dimension": 768,
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: ",
        "description": (
            "Nomic Embed v2 MoE | Nomic AI | ModernBERT MoE | 305M params | 768 dim | Feb 2025. "
            "Known for: Mixture-of-Experts routing for 100+ languages, fully open-source"
        ),
    },
}

_S3_BUCKET_ENV = "CORAL_BENCHMARK_MODELS_S3_BUCKET"
_S3_PREFIX_ENV = "CORAL_BENCHMARK_MODELS_S3_PREFIX"
_DEFAULT_S3_PREFIX = "benchmark-models/"


def _resolve_s3_mirror(
    explicit_bucket: str | None,
    explicit_prefix: str | None,
) -> tuple[str, str] | None:
    """Decide whether the S3 mirror is active and return ``(bucket, prefix)``.

    Returns ``None`` when no mirror is configured (the default OSS
    case -- we'll load straight from HuggingFace Hub). Constructor
    args take precedence over env vars so notebooks / scripts can
    override deploy-time defaults locally.
    """
    bucket = explicit_bucket or (os.environ.get(_S3_BUCKET_ENV) or "").strip() or None
    if not bucket:
        return None
    prefix = (
        explicit_prefix
        or (os.environ.get(_S3_PREFIX_ENV) or "").strip()
        or _DEFAULT_S3_PREFIX
    )
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    return bucket, prefix


def _merge_tokenized_features(
    parts: list[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, Any]:
    """Concatenate tokenizer outputs from parallel chunks along batch dim.

    Pads each chunk to the same sequence length (max over chunks).
    """
    import torch

    if len(parts) == 1:
        return parts[0]

    max_len = max(int(p["input_ids"].shape[1]) for p in parts)
    merged_ids: list[Any] = []
    merged_mask: list[Any] = []
    merged_tti: list[Any] = []
    has_tti = all("token_type_ids" in p for p in parts)

    for p in parts:
        ids = p["input_ids"]
        mask = p["attention_mask"]
        cur = ids.shape[1]
        if cur < max_len:
            pad_w = max_len - cur
            ids = torch.nn.functional.pad(ids, (0, pad_w), value=pad_token_id)
            mask = torch.nn.functional.pad(mask, (0, pad_w), value=0)
        merged_ids.append(ids)
        merged_mask.append(mask)
        if has_tti:
            tti = p["token_type_ids"]
            if tti.shape[1] < max_len:
                tti = torch.nn.functional.pad(tti, (0, max_len - int(tti.shape[1])), value=0)
            merged_tti.append(tti)

    out: dict[str, Any] = {
        "input_ids": torch.cat(merged_ids, dim=0),
        "attention_mask": torch.cat(merged_mask, dim=0),
    }
    if has_tti:
        out["token_type_ids"] = torch.cat(merged_tti, dim=0)
    return out


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder for HuggingFace sentence-transformer models.

    Default load path is the HuggingFace Hub via the
    ``sentence-transformers`` library, which caches under
    ``~/.cache/huggingface/``. Operators can opt into a private S3
    mirror by setting ``CORAL_BENCHMARK_MODELS_S3_BUCKET`` (or
    passing ``s3_mirror_bucket=...``); see the module docstring.

    NOTE: For local GPU models, use ``--workers 1`` when calling
    ``embed/main.py``. Multiple workers create contention on the GPU
    without benefit.
    """

    def __init__(
        self,
        model_name: str,
        input_type: str = "product",
        device: str | None = None,
        batch_size: int = 64,
        cache_dir: str | None = None,
        use_fp16: bool = False,
        tokenize_processes: int = 0,
        max_seq_length: int | None = None,
        s3_mirror_bucket: str | None = None,
        s3_mirror_prefix: str | None = None,
    ):
        if model_name not in MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. Supported models: {list(MODELS.keys())}"
            )

        self.model_name = model_name
        self.model_info = MODELS[model_name]
        self.input_type = input_type
        self.batch_size = batch_size
        self.use_fp16 = bool(use_fp16)
        self.tokenize_processes = max(0, int(tokenize_processes))

        if input_type == "query":
            self.prefix = self.model_info["query_prefix"]
        else:
            self.prefix = self.model_info["document_prefix"]

        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/benchmark_models")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._s3_mirror = _resolve_s3_mirror(s3_mirror_bucket, s3_mirror_prefix)
        # Either a local path (S3 mirror was synced) or an HF repo id;
        # SentenceTransformer accepts both transparently.
        model_ref = self._resolve_model_ref(model_name)

        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._resolved_device = str(device)

        if str(device).startswith("cuda") and torch.cuda.is_available():
            if os.environ.get("SEC_EMBED_CUDNN_BENCHMARK", "1").strip().lower() not in (
                "0",
                "false",
                "no",
            ):
                torch.backends.cudnn.benchmark = True
            if os.environ.get("SEC_EMBED_MATMUL_HIGH", "1").strip().lower() not in (
                "0",
                "false",
                "no",
            ):
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass

        print(f"Loading SentenceTransformer model: {model_name}")
        print(f"  Source: {model_ref}")
        print(f"  Device: {device}")
        print(f"  Input type: {input_type}")
        if self.prefix:
            print(f"  Prefix: '{self.prefix}'")

        self.model = SentenceTransformer(model_ref, device=device, trust_remote_code=True)

        if max_seq_length is not None:
            cap = int(max_seq_length)
            if cap > 0:
                self.model.max_seq_length = cap
                print(f"  max_seq_length (capped): {cap}")

        print(f"  Dimension: {self.model.get_sentence_embedding_dimension()}")
        print("  Model loaded")

    def _resolve_model_ref(self, model_name: str) -> str:
        """Return the load reference (local path or HF repo id) for ``model_name``.

        - With an S3 mirror configured: sync into the local cache and
          return the local path. This preserves the air-gap-friendly
          flow for operators who don't want runtime traffic to HF.
        - Without a mirror: return the HF repo id and let
          ``sentence-transformers`` fetch + cache it via HF Hub. This
          is the default and works for any OSS user with internet
          access.
        """
        if self._s3_mirror is not None:
            return str(self._sync_from_s3(model_name))

        hf_repo = self.model_info.get("hf_repo")
        if not hf_repo:
            raise ValueError(
                f"Model {model_name!r} has no `hf_repo` registered and no "
                f"S3 mirror configured. Set ${_S3_BUCKET_ENV} to point at a "
                f"private mirror, or add a `hf_repo` to MODELS[{model_name!r}]."
            )
        return hf_repo

    def _sync_from_s3(self, model_name: str) -> Path:
        """Mirror the model down from the configured S3 bucket.

        Only invoked when the operator explicitly opts into the S3
        path. Uses ``aws s3 sync`` (rather than boto3 calls) because
        sentence-transformer model directories contain many files
        and the CLI handles parallelism + resume natively.
        """
        assert self._s3_mirror is not None  # narrowed by caller
        bucket, prefix = self._s3_mirror

        local_dir = self.cache_dir / model_name
        marker = local_dir / ".download_complete"

        if marker.exists():
            print(f"  Using cached model: {local_dir}")
            return local_dir

        # The s3_key may already start with the configured prefix
        # (legacy layout) or be a clean model-relative path; both
        # work because S3 ignores the difference between
        # ``benchmark-models/bge-m3/`` and ``bge-m3/`` once joined
        # under a matching prefix.
        s3_key = self.model_info["s3_key"].lstrip("/")
        if s3_key.startswith(prefix):
            s3_path = s3_key
        else:
            s3_path = prefix + s3_key
        s3_uri = f"s3://{bucket}/{s3_path}"
        print(f"  Downloading model from {s3_uri} ...")
        local_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["aws", "s3", "sync", s3_uri, str(local_dir), "--no-progress"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        marker.touch()
        print(f"  Downloaded to {local_dir}")
        return local_dir

    def _encode_with_threaded_tokenize(
        self,
        sentences: list[str],
        *,
        show_progress_bar: bool,
        use_amp: bool,
    ) -> np.ndarray:
        """Mirror SentenceTransformer single-device encode() batching, but run
        ``tokenize()`` on sub-batches in parallel threads.
        """
        import torch
        import torch.nn.functional as F
        from sentence_transformers.util import batch_to_device
        from tqdm.auto import trange

        model = self.model
        n_tok = max(1, int(self.tokenize_processes))
        bs = self.batch_size
        pad_id = model.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        length_sorted_idx = np.argsort([-model._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[int(idx)] for idx in length_sorted_idx]
        all_batches: list[np.ndarray] = []

        def _tokenize_chunk(chunk: list[str]) -> dict[str, Any]:
            return model.tokenize(chunk)

        for start_index in trange(
            0,
            len(sentences_sorted),
            bs,
            desc="Batches",
            disable=not show_progress_bar,
        ):
            sentences_batch = sentences_sorted[start_index : start_index + bs]
            n = len(sentences_batch)
            if n_tok <= 1 or n <= 1:
                features = model.tokenize(sentences_batch)
            else:
                n_workers = min(n_tok, n)
                sub_size = max(1, math.ceil(n / n_workers))
                chunks = [sentences_batch[i : i + sub_size] for i in range(0, n, sub_size)]
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    parts = list(executor.map(_tokenize_chunk, chunks))
                features = _merge_tokenized_features(parts, int(pad_id))

            features = batch_to_device(features, model.device)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out_features = model.forward(features)
            else:
                out_features = model.forward(features)

            emb = out_features["sentence_embedding"]
            emb = F.normalize(emb.detach(), p=2, dim=1)
            all_batches.append(emb.float().cpu().numpy())
            del features, out_features, emb

        stacked = np.concatenate(all_batches, axis=0)
        return stacked[np.argsort(length_sorted_idx)]

    def embed_texts(
        self,
        texts: list[str],
        max_retries: int = 1,
        *,
        show_progress_bar: bool = False,
    ) -> tuple:
        """Embed texts using the loaded sentence-transformer model.

        Applies the model-specific prefix (if any) and returns
        ``(embeddings, usage_info)``.
        """
        if not texts:
            return [], {"prompt_tokens": 0, "total_tokens": 0}

        if self.prefix:
            texts = [self.prefix + t for t in texts]

        import torch

        use_amp = (
            self.use_fp16 and torch.cuda.is_available() and self._resolved_device.startswith("cuda")
        )

        encode_kw: dict[str, Any] = {
            "batch_size": self.batch_size,
            "show_progress_bar": show_progress_bar,
            "normalize_embeddings": True,
        }

        with torch.no_grad():
            if self.tokenize_processes > 0:
                embeddings = self._encode_with_threaded_tokenize(
                    texts,
                    show_progress_bar=show_progress_bar,
                    use_amp=use_amp,
                )
            else:
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        embeddings = self.model.encode(texts, **encode_kw)
                else:
                    embeddings = self.model.encode(texts, **encode_kw)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = embeddings.tolist()
        return result, {"prompt_tokens": 0, "total_tokens": 0}

    def get_model_name(self) -> str:
        return f"st:{self.model_name}"

    def get_dimension(self) -> int:
        return self.model_info["dimension"]

    def get_batch_size(self) -> int:
        return self.batch_size
