# py-gpu-inference

Production-grade gRPC GPU embedding inference server. Built for high-throughput, low-latency embedding generation with transformer models.

## Features

- **Token-bucket batching**: Groups requests by padded sequence length (16, 32, 64, 128, 256, 512) to minimize padding waste
- **Backpressure / admission control**: Throughput-based (queue drain time) and latency-based (P95 tracking) rejection strategies
- **Single GPU worker thread**: Dedicated thread with `torch.compile` + CUDA graph capture for maximum GPU utilization
- **Fused mean pooling**: Single `torch.isin()` mask operation instead of per-token loops, excluding CLS/SEP/control tokens
- **Non-blocking CPU→GPU transfers**: `non_blocking=True` with bf16 precision, TF32 matmul, cuDNN auto-tuner
- **Bulk result routing**: `OutputNotifier` routes batch results back to individual requests without per-item overhead
- **Pluggable metrics**: Abstract `MetricsClient` with CloudWatch implementation and no-op fallback
- **Health endpoint**: HTTP `/health` + gRPC `HealthCheck` with queue depth, service rate, and P95 latency

## Architecture

```
Client Request (gRPC Infer)
    │
    ▼
InferenceServicer
    │  admission control (BackpressureManager)
    │  tensor conversion (bytes → torch)
    ▼
TokenBucketManager
    │  routes to bucket by padded length
    ▼
Batcher (async task)
    │  collects chunks, torch.cat, selects batch size
    ▼
GPUWorker (dedicated thread)
    │  .to(device, non_blocking=True)
    │  model forward pass
    │  fused mean pooling
    ▼
OutputNotifier (async task)
    │  routes BatchResult → InferRequest futures
    ▼
Client Response
```

## Quick Start

### Install

```bash
pip install -e .

# Optional: S3 model loading
pip install -e ".[s3]"

# Optional: Flash Attention 2 (Ampere+ GPUs)
pip install -e ".[flash-attn]"
```

### Run

```bash
# Use a HuggingFace model (default: answerdotai/ModernBERT-base)
python -m coral_gpu_inference.grpc_server

# Use a local checkpoint
MODEL_PATH=/path/to/checkpoint python -m coral_gpu_inference.grpc_server

# Use an S3 checkpoint (requires boto3)
MODEL_PATH=s3://bucket/model/checkpoint/ python -m coral_gpu_inference.grpc_server
```

### Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `answerdotai/ModernBERT-base` | Model path: local dir, HF repo ID, or S3 URI |
| `MODEL_CACHE_DIR` | `/tmp/model_cache` | Cache directory for downloaded models |
| `GRPC_HOST` | `0.0.0.0` | gRPC listen address |
| `GRPC_PORT` | `50051` | gRPC listen port |
| `HTTP_HEALTH_PORT` | `8001` | HTTP health endpoint port |
| `MAX_BATCH_SIZE` | `512` | Maximum items per batch |
| `MAX_TOKENS_PER_BATCH` | `32768` | Maximum padded tokens per batch |
| `DEADLINE_WAIT_MS` | `50` | Max wait before draining a stale bucket |
| `GPU_QUEUE_MAXSIZE` | `4` | GPU input queue depth cap |
| `METRICS_ENABLED` | `true` | Enable CloudWatch metrics |
| `METRICS_NAMESPACE` | `GPUInference` | CloudWatch namespace |
| `BACKPRESSURE_THROUGHPUT_ENABLED` | `true` | Enable throughput-based admission control |
| `BACKPRESSURE_LATENCY_ENABLED` | `false` | Enable latency-based admission control |

### Test Client

```bash
python -m coral_gpu_inference.test_client
```

## Control Tokens

This server supports models trained with structured control tokens (e.g., `[product]`, `[title]`, `[query]`). These tokens are automatically excluded from mean pooling to avoid diluting embeddings. If your model doesn't use control tokens, they'll be silently ignored.

## License

Apache 2.0
