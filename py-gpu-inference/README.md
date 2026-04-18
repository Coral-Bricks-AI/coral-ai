# py-gpu-inference

[![PyPI version](https://img.shields.io/pypi/v/coralbricks-gpu-inference.svg)](https://pypi.org/project/coralbricks-gpu-inference/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/coralbricks-gpu-inference.svg)](https://pypi.org/project/coralbricks-gpu-inference/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../LICENSE)

Production-grade GPU embedding inference in pure Python. No ONNX export, no TensorRT compilation, no C++ debugging — just `pip install` and serve.

Getting production throughput from a transformer model typically means leaving Python behind: export to ONNX, compile a TensorRT engine, configure a Triton model repository, and debug performance issues in C++. That's a steep tax for ML engineers who think in Python and PyTorch.

`py-gpu-inference` stays entirely in Python/PyTorch while delivering the infrastructure that actually matters for production: intelligent batching that doesn't waste compute on padding, admission control that doesn't crash under load, and `torch.compile` with CUDA graphs for near-TensorRT forward-pass speed. You keep your HuggingFace model, your familiar debugging tools, and your iteration speed.

```bash
pip install coralbricks-gpu-inference
python -m coralbricks.gpu_inference.grpc_server
```

That's it. Your model is now behind a gRPC server with token-bucket batching, backpressure, and GPU-optimized inference — no toolchain switch required.

## What's Different

### Token-Bucket Batching

Requests are routed into length-based buckets (16, 32, 64, 128, 256, 512 tokens) before batching. A 10-token query lands in the 16-token bucket and gets padded to 16 — not to the longest sequence in the batch. This eliminates 3-10x wasted FLOPs compared to naive batching.

The batcher drains buckets using a priority system: stale buckets first (deadline-based), then full buckets, then the largest bucket when the GPU is idle. Batch sizes are fixed (8, 16, 32, ..., 512) so `torch.compile` CUDA graphs are reused instead of recompiled.

### Dual Backpressure

Two admission control strategies, selectable at deploy time:

- **Throughput-based** (default): Estimates queue drain time from an EMA of the GPU's actual processing rate. Rejects requests when `queued_tokens / service_rate > threshold`. Predictive — it catches overload *before* latency spikes.
- **Latency-based**: Tracks P95 server-side latency in a sliding window. Soft threshold triggers 25% probabilistic shedding; hard threshold rejects 100%. Reactive — useful when latency SLAs are the primary constraint.

Both expose real-time metrics via HTTP `/health` and gRPC `HealthCheck` for load balancer integration.

### CUDA Graph-Aware Threading

`torch.compile(mode='reduce-overhead')` captures CUDA graphs into thread-local storage. Most implementations warm up in the main thread and run inference in a worker thread — silently missing the graph cache and falling back to eager execution.

This server runs warmup *in the GPU worker thread itself*, so captured graphs are available where inference actually happens. A `threading.Event` gate blocks request acceptance until warmup completes.

### Fused Mean Pooling

For models trained with structured control tokens (`[product]`, `[title]`, `[query]`, etc.), these tokens must be excluded from mean pooling to avoid diluting embeddings. The naive approach iterates over each exclude token and zeros its attention mask — ~20 sequential tensor operations.

This server pre-builds a single GPU tensor of all exclude IDs at startup and uses one `torch.isin()` call to produce the exclusion mask. One fused operation instead of 20.

### Zero-Copy Tensor Pipeline

- **gRPC transport**: Raw `bytes` fields instead of per-row protobuf encoding. `np.frombuffer` → `torch.from_numpy` with no intermediate copies.
- **Batching**: Tensors are `torch.stack`'d at the request boundary and `torch.cat`'d at the batch boundary. The GPU worker receives ready-to-go `[batch_size, seq_len]` tensors — no per-item operations.
- **Result routing**: One `BatchResult` per forward pass is routed back to individual request futures by `OutputNotifier`, avoiding per-item overhead.

### GPU Queue Depth Cap

The batcher self-throttles when the GPU input queue has 2+ pending batches, preventing memory pressure from over-queuing. Combined with `non_blocking=True` CPU→GPU transfers and `torch.inference_mode()` (faster than `no_grad` — also disables view tracking), this keeps GPU utilization high without resource contention.

## Architecture

```
Client Request (gRPC Infer)
    │
    ▼
InferenceServicer
    │  admission control (BackpressureManager)
    │  bytes → numpy → torch (zero-copy)
    ▼
TokenBucketManager
    │  routes to bucket by padded sequence length
    ▼
Batcher (async task)
    │  priority drain: stale → full → largest-when-idle
    │  torch.cat pre-stacked chunks
    │  fixed batch sizes for CUDA graph reuse
    ▼
GPUWorker (dedicated thread)
    │  .to(device, non_blocking=True)
    │  torch.compile + CUDA graph forward pass
    │  fused mean pooling (single torch.isin mask)
    ▼
OutputNotifier (async task)
    │  batch-level result → per-request future routing
    ▼
Client Response
```

## Quick Start

### Option 1: pip install from PyPI

```bash
pip install coralbricks-gpu-inference

# With optional extras
pip install "coralbricks-gpu-inference[s3,flash-attn]"
```

### Option 2: Clone and run

```bash
git clone https://github.com/Coral-Bricks-AI/coral-ai.git
cd coral-ai/py-gpu-inference

pip install -r requirements.txt        # core deps
pip install boto3                      # optional: S3 model loading + CloudWatch metrics
pip install flash-attn                 # optional: Flash Attention 2 (Ampere+ GPUs)

python -m coralbricks.gpu_inference.grpc_server
```

### Option 3: Install from source

```bash
# From a local clone
pip install -e .

# Directly from GitHub (no clone needed)
pip install "coralbricks-gpu-inference @ git+https://github.com/Coral-Bricks-AI/coral-ai.git#subdirectory=py-gpu-inference"
```

### Run

```bash
# Default: downloads answerdotai/ModernBERT-base from HuggingFace
python -m coralbricks.gpu_inference.grpc_server

# Local checkpoint
MODEL_PATH=/path/to/checkpoint python -m coralbricks.gpu_inference.grpc_server

# S3 checkpoint (requires boto3)
MODEL_PATH=s3://bucket/model/ python -m coralbricks.gpu_inference.grpc_server

# Any HuggingFace model
MODEL_PATH=BAAI/bge-base-en-v1.5 python -m coralbricks.gpu_inference.grpc_server
```

### Test

```bash
# With server running:
python -m coralbricks.gpu_inference.test_client
```

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `answerdotai/ModernBERT-base` | Local path, HuggingFace repo ID, or `s3://` URI |
| `MODEL_CACHE_DIR` | `/tmp/model_cache` | Cache directory for downloaded models |
| `DEVICE` | auto (`cuda` if available) | `cuda` or `cpu` |
| `GRPC_PORT` | `50051` | gRPC listen port |
| `HTTP_HEALTH_PORT` | `8001` | HTTP health endpoint port |
| `MAX_BATCH_SIZE` | `512` | Maximum items per GPU batch |
| `MAX_TOKENS_PER_BATCH` | `32768` | Maximum padded tokens per batch |
| `DEADLINE_WAIT_MS` | `50` | Max time before draining a stale bucket |
| `GPU_QUEUE_MAXSIZE` | `4` | GPU input queue depth cap |
| `BACKPRESSURE_THROUGHPUT_ENABLED` | `true` | Enable throughput-based admission control |
| `BACKPRESSURE_LATENCY_ENABLED` | `false` | Enable latency-based admission control |
| `METRICS_ENABLED` | `true` | Enable CloudWatch metrics |
| `METRICS_NAMESPACE` | `GPUInference` | CloudWatch namespace |

## Health & Monitoring

**HTTP** (for ALB / Kubernetes probes):
```bash
curl http://localhost:8001/health
```

```json
{
  "status": "healthy",
  "accepting_requests": true,
  "queued_tokens": 0,
  "service_rate_tokens_per_sec": 45200.0,
  "estimated_drain_time_ms": 0.0,
  "p95_server_side_latency_ms": null,
  "gpu_queue_depth": 0,
  "embedding_dimension": 768,
  "uptime_seconds": 3600
}
```

Status values: `healthy` → `degraded` (approaching threshold) → `overloaded` (rejecting requests).

**gRPC** `HealthCheck` returns the same fields via protobuf.

## When to Use This vs. Alternatives

| | py-gpu-inference | Triton + TensorRT | HuggingFace TEI | FastAPI wrapper |
|---|---|---|---|---|
| **Language** | Pure Python/PyTorch | C++, Python config | Rust | Python |
| **Setup** | `pip install` | ONNX export → TRT build → model repo config | Docker pull | ~50 lines |
| **Debug with** | pdb, print, PyTorch profiler | C++ traces, Nsight | Rust logs | pdb |
| **Batching** | Token-bucket (length-aware) | Dynamic batching (count-based) | Continuous batching | None |
| **Admission control** | Throughput + latency | None built-in | Queue limits | None |
| **CUDA graphs** | Yes (thread-aware) | Yes (via TRT) | Yes | No |
| **Control token handling** | Fused exclusion | Manual post-processing | No | Manual |
| **Best for** | Production perf without leaving Python | Maximum raw throughput (when you can invest in the toolchain) | Quick deployment | Prototyping |

## License

Apache 2.0 — see [LICENSE](../LICENSE) for details.
