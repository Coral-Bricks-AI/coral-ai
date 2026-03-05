#!/usr/bin/env python3
"""
Data models for the gpu_inference service.

Optimized pipeline:
- BatchChunk: pre-stacked tensors from one gRPC request (replaces per-item BucketItem)
- BatchJob: GPU batch with pre-concatenated tensors (no torch.stack needed)
- BatchResult: bulk completion with all embeddings (replaces per-item ItemResult)
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class InferRequest:
    """
    Represents one gRPC Infer call from the gateway.
    
    A single gateway API request may fan out to multiple InferRequests
    (one per bucket group), all sharing the same request_id.
    """
    request_id: str
    num_items: int  # number of items in this batch
    start_time: float = field(default_factory=time.time)
    
    # Results populated as chunks complete
    embeddings: List[Optional[np.ndarray]] = field(init=False)
    to_be_embedded: int = field(init=False)
    
    # Timing information
    end_time: Optional[float] = None
    
    # Future resolved when all items complete
    future: asyncio.Future = field(default_factory=asyncio.Future)
    
    # Thread safety lock for notify methods
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    
    def __post_init__(self):
        """Initialize embeddings list and counter."""
        self.embeddings = [None] * self.num_items
        self.to_be_embedded = self.num_items
    
    async def notify_chunk(self, embeddings_array: np.ndarray, start_index: int = 0):
        """
        Bulk notification: store a chunk of embeddings at once.
        
        Called by the output notifier when a batch completes.
        Much more efficient than per-item notify (1 lock acquisition
        instead of N, fewer event loop wakeups).
        
        Args:
            embeddings_array: [chunk_size, D] numpy array
            start_index: starting item index within this InferRequest
        """
        async with self._lock:
            n = len(embeddings_array)
            for i in range(n):
                self.embeddings[start_index + i] = embeddings_array[i]
            self.to_be_embedded -= n
            
            if self.to_be_embedded == 0:
                self.end_time = time.time()
                if not self.future.done():
                    self.future.set_result(self.embeddings)


@dataclass
class BatchChunk:
    """
    Pre-stacked group of items from one gRPC request, all same bucket.
    
    Replaces per-item BucketItem. Instead of N individual tensor objects
    and N queue operations, one BatchChunk holds the entire batch
    as a single pre-stacked tensor pair.
    """
    infer_request: InferRequest
    input_ids: torch.Tensor        # [N, seq_len] pre-stacked
    attention_mask: torch.Tensor   # [N, seq_len] pre-stacked
    num_items: int
    enqueue_time: float = field(default_factory=time.time)


@dataclass
class BatchJob:
    """
    A batch sent to the GPU worker with pre-concatenated tensors.
    
    The GPU worker just does .to(device) — no torch.stack needed.
    """
    chunks: List[BatchChunk]
    input_ids: torch.Tensor        # [total_items, seq_len] pre-concatenated
    attention_mask: torch.Tensor   # [total_items, seq_len] pre-concatenated
    bucket_max_len: int
    total_items: int
    batch_id: int = 0
    enqueue_time: float = field(default_factory=time.time)


@dataclass
class BatchResult:
    """
    Result of a GPU batch. Carries all embeddings + chunk references
    for routing back to the correct InferRequests.
    
    Replaces per-item ItemResult. Instead of N queue.put + N notify calls,
    one BatchResult triggers bulk notification.
    """
    chunks: List[BatchChunk]
    embeddings: np.ndarray         # [total_items, D] on CPU
