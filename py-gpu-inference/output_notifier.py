#!/usr/bin/env python3
"""
Output notifier for the gpu_inference service.

Optimized: processes BatchResults (bulk) instead of per-item ItemResults.
One batch completion triggers one notify_chunk per chunk, instead of
N individual notify calls with N lock acquisitions.
"""

import asyncio
import logging
import time
from typing import Optional

from .models import BatchResult

logger = logging.getLogger(__name__)


class OutputNotifier:
    """
    Output notifier async task that drains the completion queue.
    
    Receives BatchResults from the GPU worker and routes embeddings
    back to the correct InferRequests via notify_chunk().
    """
    
    def __init__(self, completion_queue: asyncio.Queue):
        self.completion_queue = completion_queue
        self._stop = False
        self._task: Optional[asyncio.Task] = None
        self.notification_counter = 0
        
    async def start(self):
        """Start the notifier task."""
        self._stop = False
        self._task = asyncio.create_task(self._run())
        logger.info("Output notifier started")
        
    async def stop(self):
        """Stop the notifier task."""
        self._stop = True
        if self._task:
            await self._task
        logger.info("Output notifier stopped")
        
    async def _run(self):
        """Main notifier loop."""
        while not self._stop:
            try:
                try:
                    batch_result: BatchResult = await asyncio.wait_for(
                        self.completion_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Route embeddings to InferRequests in bulk
                # One notify_chunk per chunk (not per item)
                offset = 0
                for chunk in batch_result.chunks:
                    chunk_embeddings = batch_result.embeddings[offset:offset + chunk.num_items]
                    await chunk.infer_request.notify_chunk(chunk_embeddings)
                    offset += chunk.num_items
                
                self.notification_counter += 1
                
            except Exception as e:
                logger.error(f"Error in output notifier: {e}", exc_info=True)
                await asyncio.sleep(0.01)
