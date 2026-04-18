#!/usr/bin/env python3
"""
Batching orchestration for the gpu_inference service.

Optimized pipeline:
- Works with BatchChunks (pre-stacked tensors) instead of individual BucketItems
- Batcher concatenates chunk tensors with torch.cat (not torch.stack per item)
- Tracks total items per bucket for accurate batch size selection
"""

import asyncio
import logging
import queue
import time
from typing import Dict, List, Optional

import torch

from .metrics import MetricsClient
from .config import InferenceConfig
from .models import BatchChunk, BatchJob
from .backpressure import BackpressureManager

logger = logging.getLogger(__name__)


class TokenBucketManager:
    """
    Manages token-length-based buckets for efficient batching.
    
    Now works with BatchChunks (pre-stacked groups) instead of individual items.
    Tracks total items per bucket for accurate batch size selection.
    """
    
    def __init__(self, bucket_thresholds: List[int], metrics: Optional[MetricsClient] = None):
        self.bucket_thresholds = sorted(bucket_thresholds)
        self.metrics = metrics
        self.buckets: Dict[int, asyncio.Queue] = {
            threshold: asyncio.Queue() for threshold in self.bucket_thresholds
        }
        # Track total items per bucket (chunks can have varying num_items)
        self._bucket_item_counts: Dict[int, int] = {
            threshold: 0 for threshold in self.bucket_thresholds
        }
        logger.info(f"Initialized {len(self.buckets)} token buckets: {self.bucket_thresholds}")
        
    def get_bucket_for_token_count(self, token_count: int) -> int:
        """Determine which bucket a token count belongs to."""
        for threshold in self.bucket_thresholds:
            if token_count <= threshold:
                return threshold
        return self.bucket_thresholds[-1]
        
    async def enqueue_chunk(self, chunk: BatchChunk, bucket_threshold: int):
        """
        Enqueue a BatchChunk into the specified bucket.
        
        Single queue operation regardless of chunk size (1 put vs N puts).
        """
        await self.buckets[bucket_threshold].put(chunk)
        self._bucket_item_counts[bucket_threshold] += chunk.num_items
        
        if self.metrics:
            self.metrics.put_counter("heap_enqueued_items", float(chunk.num_items))
            self.metrics.put_counter("heap_enqueued_padded_tokens", float(bucket_threshold * chunk.num_items))
    
    def get_total_items(self, bucket_threshold: int) -> int:
        """Get total items (not chunks) queued in a bucket."""
        return self._bucket_item_counts.get(bucket_threshold, 0)
    
    def deduct_items(self, bucket_threshold: int, count: int):
        """Deduct item count after dequeuing chunks."""
        self._bucket_item_counts[bucket_threshold] = max(
            0, self._bucket_item_counts[bucket_threshold] - count
        )
    
    def get_total_queued_tokens(self) -> int:
        """Get total queued padded tokens across all buckets (for backpressure)."""
        total = 0
        for threshold in self.bucket_thresholds:
            total += threshold * self._bucket_item_counts[threshold]
        return total
    
    def get_stats(self) -> Dict[int, int]:
        """Get current item counts for all buckets."""
        return {threshold: self._bucket_item_counts[threshold] for threshold in self.bucket_thresholds}


class Batcher:
    """
    Batcher async task that drains bucket chunks and creates BatchJobs.
    
    Collects BatchChunks from bucket queues, concatenates their pre-stacked
    tensors with torch.cat, and sends BatchJobs to the GPU worker.
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        bucket_manager: TokenBucketManager,
        gpu_input_queue: queue.Queue,
        backpressure_manager: BackpressureManager
    ):
        self.config = config
        self.bucket_manager = bucket_manager
        self.gpu_input_queue = gpu_input_queue
        self.backpressure_manager = backpressure_manager
        self._stop = False
        self._task: Optional[asyncio.Task] = None
        self.batch_counter: int = 0
        
    async def start(self):
        """Start the batcher task."""
        self._stop = False
        self._task = asyncio.create_task(self._run())
        logger.info("Batcher started")
        
    async def stop(self):
        """Stop the batcher task."""
        self._stop = True
        if self._task:
            await self._task
        logger.info("Batcher stopped")
        
    async def _run(self):
        """Main batcher loop."""
        while not self._stop:
            try:
                cycle_start = time.time()
                
                # Check GPU queue depth - if backed up, wait
                gpu_queue_depth = self.gpu_input_queue.qsize()
                if gpu_queue_depth >= 2:
                    await asyncio.sleep(0.010)
                    continue
                
                # Try to form a batch
                batch_job = await self._collect_batch()
                
                if batch_job is None:
                    await asyncio.sleep(0.001)
                    continue
                
                self.batch_counter += 1
                batch_job.batch_id = self.batch_counter
                
                self.gpu_input_queue.put_nowait(batch_job)
                
                # Update backpressure service rate
                padded_tokens = batch_job.total_items * batch_job.bucket_max_len
                self.backpressure_manager.record_drain(padded_tokens)
                
                total_cycle_ms = (time.time() - cycle_start) * 1000
                gpu_queue_depth_after = self.gpu_input_queue.qsize()
                
                logger.info(
                    f"[Batch #{batch_job.batch_id}] "
                    f"Items: {batch_job.total_items} | "
                    f"Bucket: {batch_job.bucket_max_len} | "
                    f"Chunks: {len(batch_job.chunks)} | "
                    f"GPU queue: {gpu_queue_depth_after} | "
                    f"Cycle: {total_cycle_ms:.2f}ms"
                )
                
            except Exception as e:
                logger.error(f"Error in batcher: {e}", exc_info=True)
                await asyncio.sleep(0.01)
    
    def _select_bucket(self) -> Optional[int]:
        """
        Select which bucket to drain.
        
        Priority:
        1. Stale bucket (oldest chunk >= deadline_wait_ms)
        2. Full bucket (total items >= max fixed batch size)
        3. Largest non-empty bucket if GPU is idle or items are aged
        """
        now = time.time()
        bucket_manager = self.bucket_manager
        
        largest_bucket = None
        largest_items = 0
        max_age_ms = 0.0
        
        for threshold, bucket_queue in bucket_manager.buckets.items():
            if bucket_queue.qsize() == 0:
                continue
            
            total_items = bucket_manager.get_total_items(threshold)
            if total_items == 0:
                continue
            
            try:
                first_chunk = bucket_queue._queue[0]
                age_ms = (now - first_chunk.enqueue_time) * 1000
                
                if age_ms > max_age_ms:
                    max_age_ms = age_ms
                
                # Stale: return immediately
                if age_ms >= self.config.deadline_wait_ms:
                    return threshold
                    
            except (IndexError, AttributeError):
                pass
            
            if total_items > largest_items:
                largest_items = total_items
                largest_bucket = threshold
                
                # Full enough for max batch size
                max_batch_size = self.config.fixed_batch_sizes[-1]
                if largest_items >= max_batch_size:
                    return threshold
        
        if largest_bucket is not None:
            gpu_queue_depth = self.gpu_input_queue.qsize()
            if gpu_queue_depth == 0:
                return largest_bucket
            if max_age_ms >= self.config.min_item_delay_ms:
                return largest_bucket
        
        return None
    
    def _select_batch_size(self, bucket_threshold: int, available_items: int) -> int:
        """Select appropriate fixed batch size (round down)."""
        max_batch_by_tokens = self.config.max_tokens_per_batch // bucket_threshold
        max_allowed = min(max_batch_by_tokens, available_items, self.config.max_batch_size)
        
        selected_batch_size = None
        for batch_size in reversed(self.config.fixed_batch_sizes):
            if batch_size <= max_allowed:
                selected_batch_size = batch_size
                break
        
        if selected_batch_size is None:
            selected_batch_size = available_items if available_items > 0 else 0
        
        return selected_batch_size
                
    async def _collect_batch(self) -> Optional[BatchJob]:
        """Collect chunks from a bucket and form a BatchJob."""
        bucket_threshold = self._select_bucket()
        if bucket_threshold is None:
            return None
        
        available_items = self.bucket_manager.get_total_items(bucket_threshold)
        if available_items == 0:
            return None
        
        target_batch_size = self._select_batch_size(bucket_threshold, available_items)
        if target_batch_size == 0:
            return None
        
        bucket_queue = self.bucket_manager.buckets[bucket_threshold]
        chunks: List[BatchChunk] = []
        total_items = 0
        dequeue_time = time.time()
        
        while total_items < target_batch_size:
            try:
                chunk = bucket_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            
            # If adding this chunk exceeds target and we already have chunks, put it back
            if total_items + chunk.num_items > target_batch_size and chunks:
                # Put it back for the next batch cycle
                await bucket_queue.put(chunk)
                break
            
            chunks.append(chunk)
            total_items += chunk.num_items
            self.bucket_manager.deduct_items(bucket_threshold, chunk.num_items)
            
            if self.bucket_manager.metrics:
                queue_time_ms = (dequeue_time - chunk.enqueue_time) * 1000
                self.bucket_manager.metrics.put_histogram(
                    "time_spent_in_queue_ms",
                    queue_time_ms,
                    dimensions={"bucket_size": str(bucket_threshold)}
                )
        
        if not chunks:
            return None
        
        if self.bucket_manager.metrics:
            self.bucket_manager.metrics.put_counter("heap_drained_items", float(total_items))
        
        # Concatenate pre-stacked tensors (much faster than torch.stack per item)
        if len(chunks) == 1:
            input_ids = chunks[0].input_ids
            attention_mask = chunks[0].attention_mask
        else:
            input_ids = torch.cat([c.input_ids for c in chunks])
            attention_mask = torch.cat([c.attention_mask for c in chunks])
        
        return BatchJob(
            chunks=chunks,
            input_ids=input_ids,
            attention_mask=attention_mask,
            bucket_max_len=bucket_threshold,
            total_items=total_items,
        )
