#!/usr/bin/env python3
"""
Backpressure / admission control for the gpu_inference service.

Two strategies:
1. Throughput-based: reject if queued_tokens / service_rate > threshold
2. Latency-based: reject if P95 server-side latency > threshold

Factory function creates the appropriate manager based on config.

THREADING MODEL: All methods are called from the async event loop (single
thread), so no locking is needed:
- should_accept_and_update()  ← gRPC handler (async coroutine)
- record_drain()              ← Batcher (asyncio.Task)
- record_request_latency()    ← gRPC handler (async coroutine)
- get_metrics()               ← health endpoint (async coroutine)
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, Tuple

from .config import ThroughputBackpressureConfig, LatencyBackpressureConfig, BackpressureConfig

logger = logging.getLogger(__name__)


class BackpressureManager(ABC):
    """
    Base class for admission control that decides whether to accept or reject new requests.
    
    All methods run on the async event loop — no locks required.
    """
    
    def __init__(self, bucket_manager):
        self._bucket_manager = bucket_manager
    
    @abstractmethod
    def should_accept_and_update(self) -> Tuple[bool, str]:
        """
        Check if request should be accepted and update state.
        
        Queries the current queued tokens from bucket_manager internally.
        
        Returns:
            (accepted, reason): True if accepted, empty string if accepted
        """
        pass
    
    @abstractmethod
    def record_drain(self, padded_tokens: int):
        """
        Called by the Batcher after draining a batch from buckets.
        
        Used to compute the drain rate (proxy for GPU service rate).
        The batcher self-throttles on GPU queue depth, so drain rate
        naturally converges to the GPU's processing rate.
        
        Args:
            padded_tokens: batch_size * bucket_max_len
        """
        pass
    
    @abstractmethod
    def record_request_latency(self, latency_ms: float):
        """Record a completed request's server-side latency."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> dict:
        """Return current backpressure metrics for the health endpoint."""
        pass


class BackpressureManagerUsingThroughput(BackpressureManager):
    """
    Throughput-based admission control.
    
    Rejects requests if:
    - queued_tokens > max_queued_tokens (hard cap), OR
    - estimated_drain_time > max_queue_drain_time_ms (service rate check)
    
    Service rate is estimated from the batcher's drain rate (EMA).
    """
    
    def __init__(self, config: ThroughputBackpressureConfig, bucket_manager):
        super().__init__(bucket_manager)
        self.config = config
        
        self._service_rate: float = 0.0
        self._service_rate_initialized: bool = False
        self._last_drain_time: Optional[float] = None
    
    def should_accept_and_update(self) -> Tuple[bool, str]:
        """Check if request should be accepted based on throughput."""
        queued_tokens = self._bucket_manager.get_total_queued_tokens()
        return self._check_throughput(queued_tokens)
    
    def record_drain(self, padded_tokens: int):
        """
        Update service rate EMA from batcher drain event.
        
        Computes instantaneous drain rate from time since last drain,
        then applies EMA smoothing.
        """
        now = time.time()
        
        if self._last_drain_time is not None:
            elapsed_sec = now - self._last_drain_time
            if elapsed_sec > 0:
                instantaneous_rate = padded_tokens / elapsed_sec
                alpha = self.config.ema_alpha
                
                if not self._service_rate_initialized:
                    self._service_rate = instantaneous_rate
                    self._service_rate_initialized = True
                else:
                    self._service_rate = alpha * instantaneous_rate + (1 - alpha) * self._service_rate
        
        self._last_drain_time = now
    
    def record_request_latency(self, latency_ms: float):
        # Not used in throughput-based strategy
        pass
    
    def get_metrics(self) -> dict:
        queued_tokens = self._bucket_manager.get_total_queued_tokens()
        service_rate = self._service_rate
        accepted, _ = self._check_throughput(queued_tokens)
        
        if service_rate > 0:
            estimated_drain_time_ms = (queued_tokens / service_rate) * 1000
        else:
            estimated_drain_time_ms = 0.0
        
        # Determine status
        if not accepted:
            status = "overloaded"
        elif estimated_drain_time_ms > self.config.max_queue_drain_time_ms * 0.7:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "accepting_requests": accepted,
            "queued_tokens": queued_tokens,
            "service_rate_tokens_per_sec": round(service_rate, 1),
            "estimated_drain_time_ms": round(estimated_drain_time_ms, 1),
            "p95_server_side_latency_ms": None,
        }
    
    def _check_throughput(self, queued_tokens: int) -> Tuple[bool, str]:
        """Throughput-based rejection check."""
        # Hard cap on queued tokens
        if queued_tokens > self.config.max_queued_tokens:
            return False, f"queued_tokens={queued_tokens} > max={self.config.max_queued_tokens}"
        
        # Service rate check
        service_rate = max(self._service_rate, self.config.min_service_rate)
        if service_rate > 0:
            estimated_drain_time_ms = (queued_tokens / service_rate) * 1000
            if estimated_drain_time_ms > self.config.max_queue_drain_time_ms:
                return False, (
                    f"estimated_drain_time={estimated_drain_time_ms:.0f}ms "
                    f"> max={self.config.max_queue_drain_time_ms:.0f}ms"
                )
        
        return True, ""


class BackpressureManagerUsingLatency(BackpressureManager):
    """
    Latency-based admission control.
    
    Rejects requests if P95 latency exceeds thresholds:
    - Soft threshold: probabilistic rejection (25%)
    - Hard threshold: reject all
    """
    
    def __init__(self, config: LatencyBackpressureConfig, bucket_manager):
        super().__init__(bucket_manager)
        self.config = config
        
        # Ring buffer of (timestamp, latency_ms) tuples
        self._latency_samples: deque = deque(maxlen=1000)
    
    def should_accept_and_update(self) -> Tuple[bool, str]:
        """Check if request should be accepted based on latency."""
        return self._check_latency()
    
    def record_drain(self, padded_tokens: int):
        # Not used in latency-based strategy
        pass
    
    def record_request_latency(self, latency_ms: float):
        self._latency_samples.append((time.time(), latency_ms))
    
    def get_metrics(self) -> dict:
        p95 = self._compute_p95_latency_ms()
        accepted, _ = self._check_latency()
        
        # Determine status
        if not accepted:
            status = "overloaded"
        elif p95 and p95 > self.config.soft_latency_p95_ms * 0.7:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "accepting_requests": accepted,
            "queued_tokens": 0,
            "service_rate_tokens_per_sec": 0.0,
            "estimated_drain_time_ms": 0.0,
            "p95_server_side_latency_ms": round(p95, 1) if p95 is not None else None,
        }
    
    def _check_latency(self) -> Tuple[bool, str]:
        """Latency-based rejection check."""
        p95 = self._compute_p95_latency_ms()
        if p95 is None:
            return True, ""  # not enough samples
        
        # Hard threshold: reject all
        if p95 > self.config.hard_latency_p95_ms:
            return False, f"p95_latency={p95:.1f}ms > hard_threshold={self.config.hard_latency_p95_ms:.1f}ms"
        
        # Soft threshold: probabilistic shedding (25% rejection)
        if p95 > self.config.soft_latency_p95_ms:
            if random.random() < 0.25:
                return False, f"p95_latency={p95:.1f}ms > soft_threshold={self.config.soft_latency_p95_ms:.1f}ms (shed)"
        
        return True, ""
    
    def _compute_p95_latency_ms(self) -> Optional[float]:
        """Compute P95 latency from recent samples within the window."""
        window_sec = self.config.window_seconds
        min_samples = self.config.min_samples
        cutoff = time.time() - window_sec
        
        # Filter to samples within window
        recent = [lat for ts, lat in self._latency_samples if ts >= cutoff]
        
        if len(recent) < min_samples:
            return None
        
        recent.sort()
        idx = int(len(recent) * 0.95)
        idx = min(idx, len(recent) - 1)
        return recent[idx]


def create_backpressure_manager(config: BackpressureConfig, bucket_manager) -> BackpressureManager:
    """
    Factory function to create the appropriate BackpressureManager based on config.
    
    Args:
        config: BackpressureConfig with optional throughput and/or latency configs
        bucket_manager: TokenBucketManager instance for querying queued tokens
        
    Returns:
        BackpressureManager instance (throughput-based or latency-based)
        
    Raises:
        ValueError: If both or neither strategy is configured
    """
    has_throughput = config.throughput is not None
    has_latency = config.latency is not None
    
    if has_throughput and has_latency:
        raise ValueError(
            "Both throughput and latency backpressure strategies are configured. "
            "Please choose one. If you need both, use throughput-based as it's more predictive."
        )
    
    if not has_throughput and not has_latency:
        raise ValueError(
            "No backpressure strategy configured. "
            "Set either BACKPRESSURE_THROUGHPUT_ENABLED=true or BACKPRESSURE_LATENCY_ENABLED=true"
        )
    
    if has_throughput:
        logger.info("Using throughput-based backpressure manager")
        return BackpressureManagerUsingThroughput(config.throughput, bucket_manager)
    else:
        logger.info("Using latency-based backpressure manager")
        return BackpressureManagerUsingLatency(config.latency, bucket_manager)
