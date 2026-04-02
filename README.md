# coral-ai

The memory layer for agentic AI. GPU-native embedding training, inference, and retrieval — built for agents that need to remember at scale.

## Components

| Package | Description | Status |
|---------|-------------|--------|
| [py-gpu-inference](py-gpu-inference/) | Production gRPC GPU embedding server with token-bucket batching and backpressure | Available |
| [coralbricks-crewai](integrations/crewai/) | CrewAI integration — memory store, search tool, and forget | Available |
| [coralbricks-langchain](integrations/langchain/) | LangChain integration — retriever, chat history, and agent tools | Available |
| [persistent-agent-memory](integrations/openclaw/) | OpenClaw skill — bash-based store / retrieve / forget | Available |
| contrastive-train | Multi-label contrastive training with safety guards | Coming soon |
| vector-explain | Explainable vector search with field-level similarity | Coming soon |

## Quick Start

```bash
cd py-gpu-inference
pip install -e .
python -m coral_gpu_inference.grpc_server
```

See each component's README for detailed documentation.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
