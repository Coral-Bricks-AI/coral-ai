#!/usr/bin/env python3
"""
Model downloader with support for local paths, HuggingFace Hub, and S3.

Resolves a model path to a local directory:
  - Local path: used directly
  - HuggingFace Hub ID (e.g. "answerdotai/ModernBERT-base"): downloaded via snapshot_download
  - S3 path (s3://bucket/prefix): downloaded via boto3 (requires boto3 installed)
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def download_model(model_path: str, cache_dir: str = "/tmp/model_cache") -> str:
    """
    Resolve a model path to a local directory.

    Args:
        model_path: Local path, HuggingFace Hub repo ID, or S3 URI.
        cache_dir: Directory for caching downloaded models.

    Returns:
        Local filesystem path to the model directory.
    """
    if os.path.exists(model_path):
        logger.info(f"Using local model: {model_path}")
        return model_path

    if model_path.startswith("s3://"):
        return _download_from_s3(model_path, cache_dir)

    return _download_from_hub(model_path, cache_dir)


def _download_from_hub(repo_id: str, cache_dir: str) -> str:
    """Download model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading from HuggingFace Hub: {repo_id}")
    local_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
    )
    logger.info(f"Model downloaded to: {local_dir}")
    return local_dir


def _download_from_s3(s3_path: str, cache_dir: str) -> str:
    """Download model directory from S3 to local cache. Requires boto3."""
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 model downloads. "
            "Install it with: pip install boto3"
        )

    s3_path_stripped = s3_path.rstrip("/")
    without_prefix = s3_path_stripped[len("s3://"):]
    bucket = without_prefix.split("/")[0]
    prefix = without_prefix[len(bucket) :].lstrip("/")

    cache_name = without_prefix.replace("/", "_")
    local_model_dir = Path(cache_dir) / cache_name
    done_marker = local_model_dir / ".download_complete"

    if done_marker.exists():
        logger.info(f"Model found in cache: {local_model_dir}")
        return str(local_model_dir)

    logger.info(f"Downloading model from S3: {s3_path}")
    local_model_dir.mkdir(parents=True, exist_ok=True)

    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")

    files_downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            relative_path = key[len(prefix) :].lstrip("/")
            if not relative_path:
                continue
            local_file = local_model_dir / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Downloading: {relative_path}")
            s3_client.download_file(bucket, key, str(local_file))
            files_downloaded += 1

    if files_downloaded == 0:
        raise ValueError(f"No files found in S3 path: {s3_path}")

    done_marker.touch()
    logger.info(f"Downloaded {files_downloaded} files to cache")
    return str(local_model_dir)
