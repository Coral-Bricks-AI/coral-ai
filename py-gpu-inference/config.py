#!/usr/bin/env python3
"""
Configuration for the gpu_inference service.

All settings are configurable via environment variables.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .model_downloader import download_model

logger = logging.getLogger(__name__)


@dataclass
class ThroughputBackpressureConfig:
    """Throughput-based backpressure configuration."""
    max_queue_drain_time_ms: float = 2000.0
    max_queued_tokens: int = 131072
    ema_alpha: float = 0.3
    min_service_rate: float = 10000.0


@dataclass
class LatencyBackpressureConfig:
    """Latency-based backpressure configuration."""
    window_seconds: float = 1.0
    soft_latency_p95_ms: float = 100.0
    hard_latency_p95_ms: float = 200.0
    min_samples: int = 20


@dataclass
class BackpressureConfig:
    """
    Backpressure strategy selection.

    Use whichever has non-null config. If both are set, reject if EITHER triggers.
    """
    throughput: Optional[ThroughputBackpressureConfig] = field(
        default_factory=lambda: ThroughputBackpressureConfig()
        if os.getenv("BACKPRESSURE_THROUGHPUT_ENABLED", "true").lower() == "true"
        else None
    )
    latency: Optional[LatencyBackpressureConfig] = field(
        default_factory=lambda: LatencyBackpressureConfig()
        if os.getenv("BACKPRESSURE_LATENCY_ENABLED", "false").lower() == "true"
        else None
    )


@dataclass
class InferenceConfig:
    """Configuration for the gpu_inference service."""

    # Model path: local dir, HuggingFace repo ID, or s3:// URI
    model_path: str = field(
        default_factory=lambda: os.getenv("MODEL_PATH", "answerdotai/ModernBERT-base")
    )
    model_cache_dir: str = field(
        default_factory=lambda: os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache")
    )

    # Device
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Token-based batching
    max_batch_size: int = field(default_factory=lambda: int(os.getenv("MAX_BATCH_SIZE", "512")))
    max_tokens_per_batch: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS_PER_BATCH", "32768")))
    deadline_wait_ms: int = field(default_factory=lambda: int(os.getenv("DEADLINE_WAIT_MS", "50")))
    min_item_delay_ms: int = field(default_factory=lambda: int(os.getenv("MIN_ITEM_DELAY_MS", "5")))
    bucket_thresholds: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512])
    fixed_batch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256, 512])

    # Request timeout
    max_wait_time_s: int = field(default_factory=lambda: int(os.getenv("MAX_WAIT_TIME_S", "30")))

    # gRPC server
    grpc_host: str = field(default_factory=lambda: os.getenv("GRPC_HOST", "0.0.0.0"))
    grpc_port: int = field(default_factory=lambda: int(os.getenv("GRPC_PORT", "50051")))
    grpc_max_workers: int = field(default_factory=lambda: int(os.getenv("GRPC_MAX_WORKERS", "4")))

    # HTTP health endpoint
    http_health_port: int = field(default_factory=lambda: int(os.getenv("HTTP_HEALTH_PORT", "8001")))

    # GPU input queue cap
    gpu_queue_maxsize: int = field(default_factory=lambda: int(os.getenv("GPU_QUEUE_MAXSIZE", "4")))

    # Metrics
    metrics_enabled: bool = field(
        default_factory=lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true"
    )
    metrics_namespace: str = field(
        default_factory=lambda: os.getenv("METRICS_NAMESPACE", "GPUInference")
    )
    metrics_region: str = field(
        default_factory=lambda: os.getenv("METRICS_REGION", "us-east-1")
    )

    # Backpressure
    backpressure: BackpressureConfig = field(default_factory=BackpressureConfig)

    # Resolved local model path (set in __post_init__)
    local_model_path: str = field(init=False, default="")

    def __post_init__(self):
        """Validate configuration and download model if needed."""
        assert self.max_batch_size > 0, "max_batch_size must be positive"
        assert self.max_tokens_per_batch > 0, "max_tokens_per_batch must be positive"
        assert self.deadline_wait_ms > 0, "deadline_wait_ms must be positive"
        assert self.min_item_delay_ms >= 0, "min_item_delay_ms must be non-negative"
        assert self.min_item_delay_ms < self.deadline_wait_ms, (
            "min_item_delay_ms must be less than deadline_wait_ms"
        )
        assert self.max_wait_time_s > 0, "max_wait_time_s must be positive"
        assert len(self.bucket_thresholds) > 0, "bucket_thresholds must not be empty"
        assert self.bucket_thresholds == sorted(self.bucket_thresholds), (
            "bucket_thresholds must be sorted"
        )
        assert len(self.fixed_batch_sizes) > 0, "fixed_batch_sizes must not be empty"
        assert self.fixed_batch_sizes == sorted(self.fixed_batch_sizes), (
            "fixed_batch_sizes must be sorted"
        )
        assert self.gpu_queue_maxsize > 0, "gpu_queue_maxsize must be positive"

        self.local_model_path = download_model(self.model_path, self.model_cache_dir)

    def __str__(self):
        bp_throughput = "enabled" if self.backpressure.throughput else "disabled"
        bp_latency = "enabled" if self.backpressure.latency else "disabled"
        return (
            f"InferenceConfig(\n"
            f"  model_path={self.model_path},\n"
            f"  local_model_path={self.local_model_path},\n"
            f"  device={self.device},\n"
            f"  max_batch_size={self.max_batch_size},\n"
            f"  max_tokens_per_batch={self.max_tokens_per_batch},\n"
            f"  deadline_wait_ms={self.deadline_wait_ms},\n"
            f"  min_item_delay_ms={self.min_item_delay_ms},\n"
            f"  bucket_thresholds={self.bucket_thresholds},\n"
            f"  fixed_batch_sizes={self.fixed_batch_sizes},\n"
            f"  gpu_queue_maxsize={self.gpu_queue_maxsize},\n"
            f"  grpc_host={self.grpc_host},\n"
            f"  grpc_port={self.grpc_port},\n"
            f"  http_health_port={self.http_health_port},\n"
            f"  backpressure_throughput={bp_throughput},\n"
            f"  backpressure_latency={bp_latency},\n"
            f"  metrics_enabled={self.metrics_enabled},\n"
            f"  metrics_namespace={self.metrics_namespace}\n"
            f")"
        )
