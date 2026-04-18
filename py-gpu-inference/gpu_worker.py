#!/usr/bin/env python3
"""
GPU worker thread for the gpu_inference service.

Optimized pipeline:
- Receives BatchJobs with pre-concatenated tensors (no torch.stack)
- Just does .to(device) + model forward pass
- Emits a single BatchResult per batch (not per-item ItemResult)
"""

import asyncio
import logging
import queue
import threading
import time
from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .control_tokens import CONTRASTIVE_CONTROL_TOKENS
from .metrics import MetricsClient
from .model_loader import mean_pooling
from .models import BatchJob, BatchResult

logger = logging.getLogger(__name__)


class GPUWorker:
    """
    GPU worker that runs in a dedicated thread.
    
    This is the ONLY component that touches the GPU. It receives pre-concatenated
    BatchJobs, moves tensors to GPU, runs forward passes, and emits BatchResults.
    
    Optimized vs previous version:
    - No torch.stack: tensors arrive pre-concatenated from the batcher
    - No per-item output: emits one BatchResult per batch (not N ItemResults)
    - Cached exclude_token_ids: avoid tokenizer lookup on every batch
    
    IMPORTANT: When using torch.compile(mode='reduce-overhead'), CUDA graph warmup
    MUST happen in this thread (not the main thread) because CUDA graphs use
    thread-local storage (TLS). Warmup captured in a different thread will cause
    AssertionError in cudagraph_trees.py.
    """
    
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        device: str,
        gpu_input_queue: queue.Queue,
        completion_queue: asyncio.Queue,
        event_loop: asyncio.AbstractEventLoop,
        metrics: Optional[MetricsClient] = None,
        warmup_config: Optional[dict] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gpu_input_queue = gpu_input_queue
        self.completion_queue = completion_queue
        self.event_loop = event_loop
        self.metrics = metrics
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._warmup_config = warmup_config  # {bucket_thresholds, batch_sizes}
        self._ready_event = threading.Event()
        
        # Cache exclude token IDs (avoid lookup on every batch)
        self._exclude_token_ids = self._resolve_exclude_token_ids()
        
    def _resolve_exclude_token_ids(self):
        """Pre-resolve control token strings to IDs (called once at init)."""
        ids = []
        for token in CONTRASTIVE_CONTROL_TOKENS:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                ids.append(token_id)
        logger.info(f"Cached {len(ids)} exclude token IDs for mean pooling")
        return ids
    
    def _build_exclude_ids_tensor(self):
        """
        Build a GPU tensor of all token IDs to exclude from mean pooling.
        Includes CLS, SEP, and all control tokens — used with torch.isin()
        for a single fused mask operation instead of a per-token loop.
        Called once after warmup (when we know we're in the GPU thread).
        """
        exclude_ids = list(self._exclude_token_ids)
        if self.tokenizer.cls_token_id is not None:
            exclude_ids.append(self.tokenizer.cls_token_id)
        if self.tokenizer.sep_token_id is not None:
            exclude_ids.append(self.tokenizer.sep_token_id)
        tensor = torch.tensor(exclude_ids, dtype=torch.long, device=self.device)
        logger.info(f"Built exclude_ids tensor on {self.device}: {len(exclude_ids)} token IDs")
        return tensor
        
    def start(self):
        """Start the GPU worker thread."""
        self._stop = False
        self._ready_event.clear()
        self._thread = threading.Thread(target=self.run, name="GPUWorker", daemon=True)
        self._thread.start()
        logger.info("GPU worker thread started")
        
    def wait_ready(self, timeout: float = 300.0) -> bool:
        """Block until the GPU worker has finished warmup. Returns True if ready."""
        return self._ready_event.wait(timeout=timeout)
        
    def stop(self):
        """Stop the GPU worker thread."""
        self._stop = True
        if self._thread:
            self._thread.join(timeout=5.0)
            logger.info("GPU worker thread stopped")
    
    def _warmup(self):
        """
        Warm up the model by running dummy forward passes.
        
        MUST run in the GPU worker thread so that CUDA graphs (used by
        torch.compile mode='reduce-overhead') are captured in the correct
        thread-local storage context.
        """
        # Always build exclude_ids tensor (needed for mean pooling)
        self._exclude_ids_tensor = self._build_exclude_ids_tensor()
        
        if not self._warmup_config:
            logger.info("No warmup config provided, skipping warmup")
            return
            
        bucket_thresholds = self._warmup_config.get("bucket_thresholds", [16, 32, 64, 128, 256, 512])
        batch_sizes = self._warmup_config.get("batch_sizes", [8, 16, 32, 64])
        
        logger.info(f"Warming up model in GPU worker thread (CUDA graph capture)...")
        logger.info(f"  Bucket thresholds: {bucket_thresholds}")
        logger.info(f"  Batch sizes: {batch_sizes}")
        warmup_start = time.time()
        
        with torch.inference_mode():
            for seq_len in bucket_thresholds:
                for bs in batch_sizes:
                    warmup_texts = ["[query] warmup text"] * bs
                    dummy_input = self.tokenizer(
                        warmup_texts,
                        padding="max_length",
                        truncation=True,
                        max_length=seq_len,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = self.model(**dummy_input)
                    
                    if hasattr(outputs, 'last_hidden_state'):
                        _ = mean_pooling(
                            last_hidden_state=outputs.last_hidden_state,
                            attention_mask=dummy_input['attention_mask'],
                            input_ids=dummy_input['input_ids'],
                            tokenizer=self.tokenizer,
                            exclude_tokens=CONTRASTIVE_CONTROL_TOKENS,
                            normalize=True
                        )
                    
                    torch.cuda.synchronize()
                    logger.info(f"  Warmed up seq_len={seq_len}, batch_size={bs}")
        
        warmup_time = time.time() - warmup_start
        logger.info(f"Model warmup complete ({warmup_time:.1f}s)")
            
    def run(self):
        """Main loop for GPU worker thread."""
        logger.info(f"GPU worker running on device: {self.device}")
        
        # Warmup MUST happen here (in the GPU worker thread) so that CUDA graphs
        # are captured in the correct thread-local storage context.
        try:
            self._warmup()
        except Exception as e:
            logger.error(f"[GPU Worker] Warmup failed: {e}", exc_info=True)
        finally:
            self._ready_event.set()
        
        while not self._stop:
            try:
                try:
                    batch_job: BatchJob = self.gpu_input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                self._process_batch(batch_job)
                
            except Exception as e:
                logger.error(f"[GPU Worker] Error: {e}", exc_info=True)
                
    def _process_batch(self, batch_job: BatchJob):
        """Process a single BatchJob on the GPU."""
        batch_start = time.time()
        batch_size = batch_job.total_items
        wait_time_ms = (batch_start - batch_job.enqueue_time) * 1000
        
        try:
            # Tensors are pre-concatenated — just move to GPU (no torch.stack!)
            # non_blocking=True overlaps CPU-GPU transfer with other work
            transfer_start = time.time()
            input_ids = batch_job.input_ids.to(self.device, non_blocking=True)
            attention_mask = batch_job.attention_mask.to(self.device, non_blocking=True)
            transfer_ms = (time.time() - transfer_start) * 1000
            
            # Run forward pass
            # inference_mode() is faster than no_grad(): also disables view tracking
            encode_start = time.time()
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                if hasattr(outputs, 'last_hidden_state'):
                    last_hidden_state = outputs.last_hidden_state
                else:
                    raise AttributeError(
                        f"[GPU Batch #{batch_job.batch_id}] Model output has no last_hidden_state. "
                        f"Type: {type(outputs)}"
                    )
                
                # Mean pooling excluding special tokens (fused mask via torch.isin)
                embeddings = self._mean_pooling_fast(
                    last_hidden_state, attention_mask, input_ids
                )
                
                embeddings_cpu = embeddings.cpu().numpy()
                
            encode_time_ms = (time.time() - encode_start) * 1000
            
            # Emit metrics
            if self.metrics:
                self.metrics.put_histogram(
                    "encoding_latency_ms",
                    encode_time_ms,
                    dimensions={
                        "bucket_size": str(batch_job.bucket_max_len),
                        "batch_size": str(batch_size)
                    }
                )
                self.metrics.put_histogram("queue_wait_ms", wait_time_ms)
                self.metrics.put_histogram(
                    "batch_size",
                    float(batch_size),
                    dimensions={"bucket_size": str(batch_job.bucket_max_len)}
                )
            
            # Emit single BatchResult (not N individual ItemResults)
            batch_result = BatchResult(
                chunks=batch_job.chunks,
                embeddings=embeddings_cpu
            )
            self.event_loop.call_soon_threadsafe(
                self.completion_queue.put_nowait,
                batch_result
            )
            
            total_time_ms = (time.time() - batch_start) * 1000
            
            logger.info(
                f"[GPU Batch #{batch_job.batch_id}] Processed | "
                f"Items: {batch_size} | "
                f"Bucket: {batch_job.bucket_max_len} | "
                f"Wait: {wait_time_ms:.2f}ms | "
                f"Transfer: {transfer_ms:.2f}ms | "
                f"Encoding: {encode_time_ms:.2f}ms | "
                f"Total: {total_time_ms:.2f}ms"
            )
            
        except Exception as e:
            logger.error(
                f"[GPU Batch #{batch_job.batch_id}] Error: {e}", exc_info=True
            )
            # Set exception on all pending InferRequests in this batch
            for chunk in batch_job.chunks:
                req = chunk.infer_request
                if not req.future.done():
                    self.event_loop.call_soon_threadsafe(
                        req.future.set_exception, e
                    )
    
    def _mean_pooling_fast(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling excluding special tokens — fused mask via torch.isin().
        
        Instead of cloning the attention mask and iterating over each exclude
        token ID (N separate tensor ops), we build a single boolean mask with
        torch.isin() against a pre-built GPU tensor of all exclude IDs.
        This reduces ~20 sequential tensor operations to 1.
        """
        # Single fused operation: mark all tokens to exclude (CLS, SEP, control tokens)
        # _exclude_ids_tensor is pre-built on GPU during warmup
        exclude_mask = torch.isin(input_ids, self._exclude_ids_tensor)
        
        # pooling_mask = attention_mask AND NOT excluded
        pooling_mask = attention_mask * (~exclude_mask).long()
        
        # Mean pooling
        input_mask_expanded = pooling_mask.unsqueeze(-1).type_as(last_hidden_state)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        mean_pooled = sum_embeddings / sum_mask
        
        # Normalize (convert to float32 first for precision)
        if mean_pooled.dtype == torch.bfloat16:
            mean_pooled = mean_pooled.float()
        mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        
        return mean_pooled
