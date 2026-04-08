# coral-ai

Open source tools for building retrieval and memory systems for AI agents. Multi-source search, persistent memory, and GPU-accelerated inference.

## Components

| Package | Description | Status |
|---------|-------------|--------|
| [coral-retrieval](multi-source-retrieval/) | Multi-source retrieval — vector, graph, and SQL with rank fusion | Available |
| [coralbricks-crewai](integrations/crewai/) | CrewAI integration — memory store, search tool, and forget | Available |
| [coralbricks-langchain](integrations/langchain/) | LangChain integration — retriever, chat history, and agent tools | Available |
| [persistent-agent-memory](integrations/openclaw/) | OpenClaw skill — bash-based store / retrieve / forget | Available |
| [py-gpu-inference](py-gpu-inference/) | gRPC embedding server with token-bucket batching (Python/PyTorch) | Available |

## Quick Start

### Multi-source retrieval

Query across vector, graph, and SQL backends — results fused with Reciprocal Rank Fusion:

```python
from coral_retrieval import MultiSourceRetriever
from coral_retrieval.backends.in_memory_vector import InMemoryVectorBackend

retriever = MultiSourceRetriever(fusion="rrf")
retriever.add(InMemoryVectorBackend(embed_fn=your_embed_fn))
result = retriever.search("recent acquisitions in fintech", top_k=10)
```

### Persistent agent memory

```python
from coralbricks_crewai import CoralBricksMemory

memory = CoralBricksMemory(api_key="YOUR_KEY", project_id="my-project")
memory.save_memory("The user prefers concise answers with citations.")
results = memory.search_memory("What does the user prefer?")
```

See each package's README for full API details.

## License

Apache 2.0 — see [LICENSE](LICENSE).
