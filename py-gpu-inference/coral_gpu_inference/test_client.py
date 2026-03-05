#!/usr/bin/env python3
"""
Lightweight test client for gpu_inference gRPC service.

Usage:
    python test_client.py
"""

import asyncio
import time

import grpc
import numpy as np

from . import inference_pb2
from . import inference_pb2_grpc


async def test_infer():
    """Test the Infer RPC with sample data."""
    
    # Connect to local gpu_inference server
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        
        print("🔌 Connected to gpu_inference server at localhost:50051")
        
        batch_size = 3
        seq_len = 32
        
        # Mock tokenized input (normally produced by a tokenizer)
        input_ids_data = np.random.randint(0, 30000, size=(batch_size, seq_len)).astype(np.int32)
        attention_mask_data = np.ones((batch_size, seq_len), dtype=np.int32)
        
        # Use the bulk bytes format (preferred over per-row TensorRow)
        request = inference_pb2.InferRequest(
            request_id="test-request-001",
            input_ids_bytes=input_ids_data.tobytes(),
            attention_mask_bytes=attention_mask_data.tobytes(),
            num_items=batch_size,
            seq_len=seq_len,
            bucket_max_len=seq_len,
            max_wait_ms=10000,
        )
        
        print(f"📤 Sending Infer request:")
        print(f"   - request_id: {request.request_id}")
        print(f"   - batch_size: {batch_size}")
        print(f"   - bucket_max_len: {seq_len}")
        
        start = time.time()
        try:
            response = await stub.Infer(request, timeout=10.0)
            elapsed_ms = (time.time() - start) * 1000
            
            if response.status == inference_pb2.COMPLETED:
                print(f"✅ Success! ({elapsed_ms:.1f}ms)")
                emb_dim = response.embedding_dim
                embeddings = np.frombuffer(
                    response.embeddings_bytes, dtype=np.float32
                ).reshape(-1, emb_dim)
                norms = np.linalg.norm(embeddings, axis=1)
                print(f"   - Received {embeddings.shape[0]} embeddings")
                print(f"   - Embedding dimension: {emb_dim}")
                print(f"   - L2 norms (should be ~1.0): {norms}")
                print(f"   - Server timings:")
                print(f"     queue_wait: {response.timings.queue_wait_ms:.1f}ms")
                print(f"     encoding: {response.timings.encoding_ms:.1f}ms")
                print(f"     total: {response.timings.total_ms:.1f}ms")
            else:
                print(f"❌ Error: {response.error_code} - {response.error_message}")
                
        except grpc.aio.AioRpcError as e:
            elapsed_ms = (time.time() - start) * 1000
            print(f"❌ gRPC error ({elapsed_ms:.1f}ms):")
            print(f"   - Code: {e.code()}")
            print(f"   - Details: {e.details()}")


async def test_health():
    """Test the HealthCheck RPC."""
    
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        
        print("\n🏥 Testing HealthCheck...")
        
        try:
            response = await stub.HealthCheck(inference_pb2.HealthRequest(), timeout=2.0)
            
            print(f"   - status: {response.status}")
            print(f"   - accepting_requests: {response.accepting_requests}")
            print(f"   - queued_tokens: {response.queued_tokens}")
            print(f"   - service_rate: {response.service_rate_tokens_per_sec:.1f} tokens/sec")
            print(f"   - estimated_drain_time: {response.estimated_drain_time_ms:.1f}ms")
            print(f"   - p95_latency: {response.p95_server_side_latency_ms:.1f}ms")
            print(f"   - gpu_queue_depth: {response.gpu_queue_depth}")
            print(f"   - uptime: {response.uptime_seconds}s")
            
        except grpc.aio.AioRpcError as e:
            print(f"❌ Health check failed: {e.code()} - {e.details()}")


async def main():
    print("=" * 60)
    print("gpu_inference Test Client")
    print("=" * 60)
    
    # Test health first
    await test_health()
    
    # Test inference
    print()
    await test_infer()
    
    print("\n" + "=" * 60)
    print("✨ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())

