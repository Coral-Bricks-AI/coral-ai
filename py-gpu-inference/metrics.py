#!/usr/bin/env python3
"""
Metrics abstraction layer for inference server.

Provides a pluggable interface for emitting metrics to CloudWatch or other services.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


class MetricsClient(ABC):
    """Abstract base class for metrics clients."""
    
    @abstractmethod
    def put_metric(self, name: str, value: float, unit: str = "None", dimensions: Optional[Dict[str, str]] = None):
        """
        Put a single metric value.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Metric unit (e.g., "Milliseconds", "Count", "None")
            dimensions: Optional dimensions as key-value pairs
        """
        pass
    
    @abstractmethod
    def put_counter(self, name: str, value: float = 1.0, dimensions: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            value: Value to increment by (default 1.0)
            dimensions: Optional dimensions as key-value pairs
        """
        pass
    
    @abstractmethod
    def put_histogram(self, name: str, value: float, dimensions: Optional[Dict[str, str]] = None):
        """
        Put a histogram/distribution value for percentile calculation.
        
        Args:
            name: Histogram name
            value: Value to record
            dimensions: Optional dimensions as key-value pairs
        """
        pass
    
    @abstractmethod
    def put_timer(self, name: str, duration_ms: float, dimensions: Optional[Dict[str, str]] = None):
        """
        Put a timing metric in milliseconds.
        
        Args:
            name: Timer name
            duration_ms: Duration in milliseconds
            dimensions: Optional dimensions as key-value pairs
        """
        pass
    
    @abstractmethod
    def flush(self):
        """Flush any buffered metrics."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the metrics client and cleanup resources."""
        pass


class NoOpMetricsClient(MetricsClient):
    """No-op metrics client that does nothing (for local dev)."""
    
    def put_metric(self, name: str, value: float, unit: str = "None", dimensions: Optional[Dict[str, str]] = None):
        pass
    
    def put_counter(self, name: str, value: float = 1.0, dimensions: Optional[Dict[str, str]] = None):
        pass
    
    def put_histogram(self, name: str, value: float, dimensions: Optional[Dict[str, str]] = None):
        pass
    
    def put_timer(self, name: str, duration_ms: float, dimensions: Optional[Dict[str, str]] = None):
        pass
    
    def flush(self):
        pass
    
    def close(self):
        pass


class CloudWatchMetricsClient(MetricsClient):
    """
    CloudWatch metrics client with batching and flushing.
    
    Buffers metrics and flushes periodically to avoid rate limits.
    """
    
    def __init__(
        self,
        namespace: str = "GPUInference",
        region: str = "us-east-1",
        flush_interval: float = 0.2,
        # CloudWatch PutMetricData supports up to 1000 MetricData items per call.
        # Using a small batch size can cause the enqueue buffer to grow under load.
        max_batch_size: int = 1000,
        enqueue_warn_threshold: int = 5000,
        max_enqueue_buffer_size: int = 50000,
    ):
        """
        Initialize CloudWatch metrics client.
        
        Args:
            namespace: CloudWatch namespace
            region: AWS region
            flush_interval: Flush interval in seconds
            max_batch_size: Max metrics per batch (CloudWatch limit is 1000)
        """
        try:
            import boto3
            self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        except Exception as e:
            logger.warning(f"Failed to initialize CloudWatch client: {e}. Metrics will be disabled.")
            self.cloudwatch = None
        
        self.namespace = namespace
        self.flush_interval = flush_interval
        self.max_batch_size = max_batch_size
        self.enqueue_warn_threshold = enqueue_warn_threshold
        self.max_enqueue_buffer_size = max_enqueue_buffer_size
        
        # Double-buffer for metric data to minimize lock contention
        # One buffer for enqueuing (active), one for draining (being flushed)
        self._enqueue_buffer: deque = deque()
        self._drain_buffer: deque = deque()
        self._buffer_lock = threading.Lock()
        
        # Background flushing thread
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_flushing = threading.Event()
        self._warned_enqueue_full = False
        self._warned_dropping = False
        
        if self.cloudwatch:
            self._start_flush_thread()
            logger.info(f"CloudWatch metrics client initialized: namespace={namespace}, region={region}")
    
    def _start_flush_thread(self):
        """Start background thread for periodic flushing."""
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="MetricsFlushThread",
            daemon=True
        )
        self._flush_thread.start()
    
    def _flush_loop(self):
        """Background loop for periodic flushing."""
        while not self._stop_flushing.is_set():
            time.sleep(self.flush_interval)
            try:
                # Only flush if there's data in the enqueue buffer
                if len(self._enqueue_buffer) > 0:
                    # If buffer is large, flush multiple times
                    while len(self._enqueue_buffer) > 0 or len(self._drain_buffer) > 0:
                        self.flush()
                        # If buffers still have items, flush again immediately
                        if len(self._enqueue_buffer) > self.max_batch_size or len(self._drain_buffer) > 0:
                            continue
                        else:
                            break
            except Exception as e:
                logger.error(f"Error flushing metrics: {e}", exc_info=True)
    
    def _add_to_buffer(self, metric_data: Dict[str, Any]):
        """Add metric to buffer (fast path - only appends to enqueue buffer)."""
        if not self.cloudwatch:
            return
        
        with self._buffer_lock:
            # Prevent unbounded growth if CloudWatch is slow/unavailable.
            if len(self._enqueue_buffer) >= self.max_enqueue_buffer_size:
                # Drop newest metric to avoid memory blow-up.
                # (We still keep logging minimal by only warning when we cross the threshold.)
                if not self._warned_dropping:
                    logger.warning(
                        f"Metrics enqueue buffer reached max size ({self.max_enqueue_buffer_size}). "
                        "Dropping subsequent metrics until it drains."
                    )
                    self._warned_dropping = True
                return

            self._enqueue_buffer.append(metric_data)
            
            # Auto-flush if buffer is getting very full (to prevent memory issues)
            # Use a high threshold to avoid blocking the critical path
            if len(self._enqueue_buffer) >= self.enqueue_warn_threshold and not self._warned_enqueue_full:
                # Log warning but don't block - let background thread handle it
                logger.warning(
                    f"Metrics enqueue buffer is full ({len(self._enqueue_buffer)} items). "
                    "Consider increasing flush frequency or batch size."
                )
                self._warned_enqueue_full = True

            # Reset warning once we drain back below the threshold.
            if len(self._enqueue_buffer) < self.enqueue_warn_threshold:
                self._warned_enqueue_full = False
            if len(self._enqueue_buffer) < self.max_enqueue_buffer_size:
                self._warned_dropping = False
    
    def _flush_buffer(self):
        """
        Flush buffered metrics to CloudWatch using double-buffering.
        
        This implementation uses two buffers:
        1. enqueue_buffer: actively receiving new metrics (hot path)
        2. drain_buffer: being flushed to CloudWatch (background)
        
        The flush process:
        1. Hold lock briefly to swap buffers
        2. Release lock immediately
        3. Drain the buffer without holding the lock (network I/O)
        
        This minimizes lock contention - request threads only wait for buffer swap (~1μs),
        not for CloudWatch API calls (~100-500ms).
        """
        if not self.cloudwatch:
            return
        
        # Step 1: Swap buffers while holding lock (very fast operation)
        with self._buffer_lock:
            # If drain buffer still has items, don't swap yet
            if len(self._drain_buffer) > 0:
                # This shouldn't happen often, but if it does, we'll just process drain buffer first
                pass
            else:
                # Swap: enqueue becomes drain, drain becomes enqueue (now empty)
                self._enqueue_buffer, self._drain_buffer = self._drain_buffer, self._enqueue_buffer
        
        # Step 2: Process drain buffer WITHOUT holding the lock
        # Request threads can now enqueue to the new enqueue buffer in parallel
        if not self._drain_buffer:
            return
        
        # Process in batches up to max_batch_size
        while self._drain_buffer:
            batch = []
            # Take up to max_batch_size items
            while self._drain_buffer and len(batch) < self.max_batch_size:
                batch.append(self._drain_buffer.popleft())
            
            if batch:
                try:
                    self.cloudwatch.put_metric_data(
                        Namespace=self.namespace,
                        MetricData=batch
                    )
                except Exception as e:
                    logger.error(f"Failed to put metrics to CloudWatch: {e}")
    
    def put_metric(self, name: str, value: float, unit: str = "None", dimensions: Optional[Dict[str, str]] = None):
        """Put a single metric value."""
        metric_data = {
            'MetricName': name,
            'Value': value,
            'Unit': unit,
            'Timestamp': time.time()
        }
        
        if dimensions:
            metric_data['Dimensions'] = [
                {'Name': k, 'Value': str(v)} for k, v in dimensions.items()
            ]
        
        self._add_to_buffer(metric_data)
    
    def put_counter(self, name: str, value: float = 1.0, dimensions: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.put_metric(name, value, unit="Count", dimensions=dimensions)
    
    def put_histogram(self, name: str, value: float, dimensions: Optional[Dict[str, str]] = None):
        """Put a histogram/distribution value."""
        self.put_metric(name, value, unit="Milliseconds", dimensions=dimensions)
    
    def put_timer(self, name: str, duration_ms: float, dimensions: Optional[Dict[str, str]] = None):
        """Put a timing metric."""
        self.put_metric(name, duration_ms, unit="Milliseconds", dimensions=dimensions)
    
    def flush(self):
        """Flush any buffered metrics."""
        if not self.cloudwatch:
            return
        
        # No lock needed here - _flush_buffer handles its own locking
        self._flush_buffer()
    
    def close(self):
        """Close the metrics client and cleanup resources."""
        self._stop_flushing.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        
        # Final flush
        self.flush()
        logger.info("CloudWatch metrics client closed")


def create_metrics_client(
    enabled: bool = True,
    namespace: str = "GPUInference",
    region: str = "us-east-1"
) -> MetricsClient:
    """
    Factory function to create appropriate metrics client.
    
    Args:
        enabled: Whether to enable metrics
        namespace: CloudWatch namespace
        region: AWS region
        
    Returns:
        MetricsClient instance (CloudWatch or NoOp)
    """
    if not enabled:
        logger.info("Metrics disabled, using NoOpMetricsClient")
        return NoOpMetricsClient()
    
    # Allow runtime tuning without code changes.
    # These env vars are safe to omit; defaults are chosen to avoid buffer growth under load.
    import os
    flush_interval = float(os.getenv("METRICS_FLUSH_INTERVAL_S", "0.2"))
    max_batch_size = int(os.getenv("METRICS_MAX_BATCH_SIZE", "1000"))
    enqueue_warn_threshold = int(os.getenv("METRICS_ENQUEUE_WARN_THRESHOLD", "5000"))
    max_enqueue_buffer_size = int(os.getenv("METRICS_MAX_ENQUEUE_BUFFER_SIZE", "50000"))

    return CloudWatchMetricsClient(
        namespace=namespace,
        region=region,
        flush_interval=flush_interval,
        max_batch_size=max_batch_size,
        enqueue_warn_threshold=enqueue_warn_threshold,
        max_enqueue_buffer_size=max_enqueue_buffer_size,
    )

