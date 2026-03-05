#!/usr/bin/env python3
"""
gRPC server for the gpu_inference service.

This is the entry point. It:
- Loads the model and warms up torch.compile kernels
- Starts the batching pipeline (TokenBucketManager -> Batcher -> GPUWorker -> OutputNotifier)
- Serves gRPC InferenceService (Infer + HealthCheck)
- Serves a lightweight HTTP /health endpoint for monitoring
"""

import asyncio
import logging
import queue
import signal
import time
import warnings
from concurrent import futures
from threading import Thread
from typing import Optional

import grpc
import numpy as np
import torch

# Enable TF32 for Ampere+ GPUs (~8x faster matmuls with negligible precision loss)
torch.set_float32_matmul_precision('high')

# cuDNN auto-tuner: benchmarks kernels on first call, picks fastest for each shape
torch.backends.cudnn.benchmark = True

# Suppress transformers warnings
warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")

from .control_tokens import CONTRASTIVE_CONTROL_TOKENS, CONTRASTIVE_CONTROL_TOKEN_SET
from .metrics import create_metrics_client, MetricsClient
from .model_loader import load_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Suppress transformers noise
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.configuration_utils').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)

from .config import InferenceConfig
from .models import InferRequest, BatchChunk
from .batcher import TokenBucketManager, Batcher
from .output_notifier import OutputNotifier
from .gpu_worker import GPUWorker
from .backpressure import BackpressureManager, create_backpressure_manager

from . import inference_pb2
from . import inference_pb2_grpc


class InferenceServer:
    """Main gpu_inference server managing model and processing components."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
        # Metrics
        self.metrics: MetricsClient = create_metrics_client(
            enabled=config.metrics_enabled,
            namespace=config.metrics_namespace,
            region=config.metrics_region
        )
        
        # Processing components
        self.bucket_manager: Optional[TokenBucketManager] = None
        self.batcher: Optional[Batcher] = None
        self.gpu_worker: Optional[GPUWorker] = None
        self.output_notifier: Optional[OutputNotifier] = None
        self.backpressure_manager: Optional[BackpressureManager] = None
        
        # Queues
        self.gpu_input_queue: Optional[queue.Queue] = None
        self.completion_queue: Optional[asyncio.Queue] = None
        
        self.embedding_dimension: int = 0
        self.start_time: float = time.time()
        
    async def initialize(self):
        """Initialize model and all processing components."""
        logger.info("=" * 60)
        logger.info("gpu_inference Server Configuration")
        logger.info("=" * 60)
        for line in str(self.config).split('\n'):
            logger.info(line)
        logger.info("=" * 60)
        
        # Load model and tokenizer
        logger.info(f"Loading model from {self.config.local_model_path}")
        logger.info(f"Using device: {self.config.device}")

        from transformers import AutoTokenizer
        from .control_tokens import CONTRASTIVE_CONTROL_TOKENS

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.local_model_path)
        vocab_before = len(self.tokenizer)

        new_tokens = [t for t in CONTRASTIVE_CONTROL_TOKENS if t not in self.tokenizer.get_vocab()]
        if new_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
            logger.info(f"Added {len(new_tokens)} control tokens to tokenizer")

        self.model = load_model(self.config.local_model_path, force_dtype=torch.bfloat16)

        if len(self.tokenizer) != vocab_before:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized embeddings: {vocab_before} -> {len(self.tokenizer)}")

        self.model.to(self.config.device)
        self.model.eval()

        logger.info(f"Tokenizer vocabulary size: {len(self.tokenizer)}")
        logger.info(f"Model dtype: {self.model.dtype}")
        
        # Compile model (warmup happens later in the GPU worker thread to avoid
        # CUDA graph TLS issues — see gpu_worker.py for details)
        warmup_config = None
        if self.config.device == "cuda":
            logger.info("Compiling model with torch.compile(mode='reduce-overhead') for CUDA graph capture...")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled successfully")
                
                # Warmup will be done in the GPU worker thread (not here!) because
                # torch.compile(mode='reduce-overhead') captures CUDA graphs into
                # thread-local storage. Warmup captured in the main thread would
                # cause AssertionError in cudagraph_trees.py when the GPU worker
                # thread tries to use the compiled model.
                warmup_config = {
                    "bucket_thresholds": self.config.bucket_thresholds,
                    "batch_sizes": self.config.fixed_batch_sizes[:6],  # warmup first 6 sizes (up to 256)
                }
                logger.info("Warmup deferred to GPU worker thread (CUDA graph TLS requirement)")
                
            except RuntimeError as e:
                if "Dynamo is not supported" in str(e):
                    logger.error(f"torch.compile() failed: {e}")
                    raise
                raise
        
        self.embedding_dimension = self.model.config.hidden_size
        logger.info(f"Embedding dimension: {self.embedding_dimension}")
        
        # Initialize queues
        self.gpu_input_queue = queue.Queue(maxsize=self.config.gpu_queue_maxsize)
        self.completion_queue = asyncio.Queue(maxsize=1000)
        
        # Initialize bucket manager first (needed by backpressure manager)
        self.bucket_manager = TokenBucketManager(
            self.config.bucket_thresholds, metrics=self.metrics
        )
        
        # Initialize backpressure manager (receives bucket_manager reference)
        self.backpressure_manager = create_backpressure_manager(
            self.config.backpressure,
            bucket_manager=self.bucket_manager
        )
        
        self.batcher = Batcher(
            config=self.config,
            bucket_manager=self.bucket_manager,
            gpu_input_queue=self.gpu_input_queue,
            backpressure_manager=self.backpressure_manager
        )
        
        event_loop = asyncio.get_event_loop()
        self.gpu_worker = GPUWorker(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.config.device,
            gpu_input_queue=self.gpu_input_queue,
            completion_queue=self.completion_queue,
            event_loop=event_loop,
            metrics=self.metrics,
            warmup_config=warmup_config,
        )
        
        self.output_notifier = OutputNotifier(
            completion_queue=self.completion_queue
        )
        
        # Start pipeline — GPU worker does warmup in its own thread
        # (CUDA graphs must be captured in the thread that will use them)
        await self.batcher.start()
        await self.output_notifier.start()
        self.gpu_worker.start()
        
        # Wait for GPU worker warmup to complete before accepting requests
        if warmup_config:
            logger.info("Waiting for GPU worker warmup to complete...")
            ready = self.gpu_worker.wait_ready(timeout=300.0)
            if ready:
                logger.info("GPU worker warmup complete — ready to accept requests")
            else:
                logger.error("GPU worker warmup timed out after 300s!")
        
        logger.info("gpu_inference server initialized successfully")
    
    async def shutdown(self):
        """Cleanup resources."""
        logger.info("Shutting down gpu_inference server...")
        if self.gpu_worker:
            self.gpu_worker.stop()
        if self.batcher:
            await self.batcher.stop()
        if self.output_notifier:
            await self.output_notifier.stop()
        logger.info("gpu_inference server shut down")


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC servicer implementing Infer and HealthCheck RPCs."""
    
    def __init__(self, server: InferenceServer):
        self.server = server
    
    async def Infer(self, request: inference_pb2.InferRequest, context):
        """Handle an Infer RPC: admission control, bulk enqueue, wait for results."""
        request_start = time.time()
        request_id = request.request_id
        bucket_max_len = request.bucket_max_len
        max_wait_ms = request.max_wait_ms if request.max_wait_ms > 0 else 5000
        
        num_items = request.num_items
        
        if num_items == 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Empty input_ids")
            return inference_pb2.InferResponse(
                request_id=request_id,
                status=inference_pb2.ERROR,
                error_code="INVALID_ARGUMENT",
                error_message="Empty input_ids"
            )
        
        # --- Admission control ---
        accepted, reason = self.server.backpressure_manager.should_accept_and_update()
        if not accepted:
            logger.warning(
                f"[{request_id[:8]}] Rejected: {reason} | "
                f"Items: {num_items} | Bucket: {bucket_max_len}"
            )
            if self.server.metrics:
                self.server.metrics.put_counter(
                    "admission_rejections", 1.0,
                    dimensions={"reason": reason.split("=")[0] if "=" in reason else reason}
                )
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details(f"Server overloaded: {reason}")
            return inference_pb2.InferResponse(
                request_id=request_id,
                status=inference_pb2.ERROR,
                error_code="RESOURCE_EXHAUSTED",
                error_message=f"Server overloaded: {reason}"
            )
        
        # --- Bulk tensor conversion: bytes → numpy → torch ---
        seq_len = request.seq_len
        input_ids_np = np.frombuffer(
            request.input_ids_bytes, dtype=np.int32
        ).reshape(num_items, seq_len).copy()
        attention_mask_np = np.frombuffer(
            request.attention_mask_bytes, dtype=np.int32
        ).reshape(num_items, seq_len).copy()
        input_ids_tensor = torch.from_numpy(input_ids_np).long()
        attention_mask_tensor = torch.from_numpy(attention_mask_np).long()
        
        # --- Create InferRequest and enqueue as a single BatchChunk ---
        infer_req = InferRequest(
            request_id=request_id,
            num_items=num_items
        )
        
        chunk = BatchChunk(
            infer_request=infer_req,
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            num_items=num_items
        )
        await self.server.bucket_manager.enqueue_chunk(chunk, bucket_max_len)
        
        # --- Wait for all embeddings to complete ---
        try:
            timeout_s = max_wait_ms / 1000.0
            embeddings_list = await asyncio.wait_for(
                infer_req.future,
                timeout=timeout_s
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[{request_id[:8]}] Timeout after {max_wait_ms}ms | "
                f"Items: {num_items} | Bucket: {bucket_max_len}"
            )
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details(f"Timeout after {max_wait_ms}ms")
            return inference_pb2.InferResponse(
                request_id=request_id,
                status=inference_pb2.ERROR,
                error_code="DEADLINE_EXCEEDED",
                error_message=f"Timeout after {max_wait_ms}ms"
            )
        except Exception as e:
            logger.error(f"[{request_id[:8]}] Error: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.InferResponse(
                request_id=request_id,
                status=inference_pb2.ERROR,
                error_code="INTERNAL",
                error_message=str(e)
            )
        
        # --- Build response ---
        total_ms = (time.time() - request_start) * 1000
        queue_wait_ms = (infer_req.end_time - infer_req.start_time) * 1000 if infer_req.end_time else total_ms
        
        # Record latency for backpressure
        self.server.backpressure_manager.record_request_latency(total_ms)
        
        # Emit metrics
        if self.server.metrics:
            self.server.metrics.put_counter("infer_request_count", 1.0, dimensions={"status": "completed"})
            self.server.metrics.put_histogram("p95_server_side_latency_ms", total_ms)
        
        # Build response with bytes format only
        embeddings_np = np.stack(embeddings_list).astype(np.float32)
        
        timings = inference_pb2.Timings(
            queue_wait_ms=queue_wait_ms,
            encoding_ms=total_ms - queue_wait_ms,
            total_ms=total_ms
        )
        
        logger.info(
            f"[{request_id[:8]}] Completed | "
            f"Items: {num_items} | Bucket: {bucket_max_len} | "
            f"Latency: {total_ms:.2f}ms"
        )
        
        return inference_pb2.InferResponse(
            request_id=request_id,
            status=inference_pb2.COMPLETED,
            embeddings_bytes=embeddings_np.tobytes(),
            embedding_dim=embeddings_np.shape[1],
            timings=timings
        )
    
    async def HealthCheck(self, request: inference_pb2.HealthRequest, context):
        """Return health and load metrics."""
        bp_metrics = self.server.backpressure_manager.get_metrics()
        gpu_queue_depth = self.server.gpu_input_queue.qsize() if self.server.gpu_input_queue else 0
        uptime = int(time.time() - self.server.start_time)
        
        return inference_pb2.HealthResponse(
            status=bp_metrics["status"],
            accepting_requests=bp_metrics["accepting_requests"],
            queued_tokens=bp_metrics["queued_tokens"],
            service_rate_tokens_per_sec=bp_metrics["service_rate_tokens_per_sec"],
            estimated_drain_time_ms=bp_metrics["estimated_drain_time_ms"],
            p95_server_side_latency_ms=bp_metrics["p95_server_side_latency_ms"] or 0.0,
            gpu_queue_depth=gpu_queue_depth,
            uptime_seconds=uptime
        )


# ---------------------------------------------------------------------------
# Lightweight HTTP health endpoint (for ALB / monitoring tools)
# ---------------------------------------------------------------------------

def start_http_health_server(server: InferenceServer, port: int):
    """Run a minimal HTTP server for /health on a background thread."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                bp_metrics = server.backpressure_manager.get_metrics()
                gpu_queue_depth = server.gpu_input_queue.qsize() if server.gpu_input_queue else 0
                body = json.dumps({
                    **bp_metrics,
                    "gpu_queue_depth": gpu_queue_depth,
                    "embedding_dimension": server.embedding_dimension,
                    "uptime_seconds": int(time.time() - server.start_time),
                })
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            else:
                self.send_response(404)
                self.end_headers()
        
        def log_message(self, format, *args):
            pass  # suppress default access logs
    
    httpd = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = Thread(target=httpd.serve_forever, name="HTTPHealth", daemon=True)
    thread.start()
    logger.info(f"HTTP health endpoint listening on port {port}")
    return httpd


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

async def serve():
    """Start the gpu_inference gRPC server."""
    config = InferenceConfig()
    server = InferenceServer(config)
    
    # Initialize model and pipeline
    await server.initialize()
    
    # Start HTTP health endpoint
    start_http_health_server(server, config.http_health_port)
    
    # Start gRPC server
    grpc_server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=config.grpc_max_workers),
        options=[
            ('grpc.max_send_message_length', 64 * 1024 * 1024),   # 64MB
            ('grpc.max_receive_message_length', 64 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 10000),                     # ping every 10s
            ('grpc.keepalive_timeout_ms', 5000),                   # timeout 5s
            ('grpc.keepalive_permit_without_calls', True),
        ]
    )
    
    servicer = InferenceServicer(server)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, grpc_server)
    
    listen_addr = f"{config.grpc_host}:{config.grpc_port}"
    grpc_server.add_insecure_port(listen_addr)
    
    logger.info(f"gRPC server listening on {listen_addr}")
    await grpc_server.start()
    
    # Graceful shutdown on SIGTERM/SIGINT
    stop_event = asyncio.Event()
    
    def _signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)
    
    await stop_event.wait()
    
    logger.info("Shutting down gRPC server (30s grace)...")
    await grpc_server.stop(grace=30)
    await server.shutdown()
    logger.info("Server stopped.")


if __name__ == "__main__":
    asyncio.run(serve())

