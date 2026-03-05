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
        
        # Create sample tokenized data (simulating what gateway would send)
        # Let's create 3 items with bucket_max_len=32
        batch_size = 3
        seq_len = 32
        
        # Mock input_ids and attention_mask (normally from tokenizer)
        input_ids_data = np.random.randint(0, 30000, size=(batch_size, seq_len))
        attention_mask_data = np.ones((batch_size, seq_len), dtype=np.int32)
        
        # Convert to protobuf format
        input_ids_rows = []
        attention_mask_rows = []
        
        for i in range(batch_size):
            input_ids_rows.append(
                inference_pb2.TensorRow(values=input_ids_data[i].tolist())
            )
            attention_mask_rows.append(
                inference_pb2.TensorRow(values=attention_mask_data[i].tolist())
            )
        
        request = inference_pb2.InferRequest(
            request_id="test-request-001",
            input_ids=input_ids_rows,
            attention_mask=attention_mask_rows,
            bucket_max_len=seq_len,
            max_wait_ms=5000
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
                print(f"   - Received {len(response.embeddings)} embeddings")
                if response.embeddings:
                    emb_dim = len(response.embeddings[0].values)
                    print(f"   - Embedding dimension: {emb_dim}")
                print(f"   - Server timings:")
                print(f"     • queue_wait: {response.timings.queue_wait_ms:.1f}ms")
                print(f"     • encoding: {response.timings.encoding_ms:.1f}ms")
                print(f"     • total: {response.timings.total_ms:.1f}ms")
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

