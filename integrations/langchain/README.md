# coralbricks-langchain

Use **CoralBricks** as the memory backend for your LangChain applications: retriever, chat history, agent memory tools, and RAG middleware — all backed by the CoralBricks Memory API.

- Drop-in **`CoralBricksRetriever`** for any LCEL chain or RAG pipeline.
- **`@dynamic_prompt` RAG middleware** — automatically injects retrieved context before every LLM call.
- Three **agent tools** (`store`, `search`, `delete`) for persistent memory across turns.
- **`CoralBricksChatMessageHistory`** — persistent chat history backed by the CoralBricks chat API.
- Share memory across agents via `project_id` / `session_id` scoping.

---

## Installation

```bash
pip install coralbricks-langchain
```

Requires Python 3.10+ and LangChain >= 1.0.

---

## API key and base URL

- **API key:** Get a CoralBricks API key from the [CoralBricks web app](https://coralbricks.ai).
- **Base URL:** `https://memory.coralbricks.ai`

```bash
export CORALBRICKS_API_KEY="your_coralbricks_api_key"
export CORAL_MEMORY_BASE_URL="https://memory.coralbricks.ai"
```

---

## Quick start

```python
from coralbricks_langchain import CoralBricksClient, CoralBricksMemory

client = CoralBricksClient(
    api_key="your_coralbricks_api_key",
    base_url="https://memory.coralbricks.ai",
)

memory = CoralBricksMemory(
    client=client,
    project_id="langchain:my-app",
    session_id="user-123",
)

# Save a memory (embed + store)
mem_id = memory.save_memory("Pro plan costs $199/month with unlimited ops.")

# Search by meaning
hits = memory.search_memory("What does the Pro plan cost?", top_k=3)
for h in hits:
    print(h.get("score"), h.get("text"))
```

---

## RAG with `@dynamic_prompt` middleware

The modern LangChain >= 1.0 RAG pattern — context is retrieved from CoralBricks and injected into the system prompt automatically before every model call:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_openai import ChatOpenAI

from coralbricks_langchain import CoralBricksClient, CoralBricksMemory, CoralBricksRetriever

client = CoralBricksClient(api_key="your_coralbricks_api_key")
memory = CoralBricksMemory(client=client, project_id="my-kb", session_id="docs-v1")
retriever = CoralBricksRetriever(memory=memory, top_k=5)

@dynamic_prompt
def rag_context(request: ModelRequest) -> str:
    query = request.messages[-1].content if request.messages else ""
    docs = retriever.invoke(query)
    context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
    return (
        "Answer the user's question using ONLY the context below.\n\n"
        f"Context:\n{context}"
    )

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_agent(model, middleware=[rag_context])

result = agent.invoke({"messages": [{"role": "user", "content": "What is the Pro plan price?"}]})
print(result["messages"][-1].content)
```

---

## Agent memory tools

Give your agent tools to store, search, and delete memories across turns:

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from coralbricks_langchain import (
    CoralBricksClient,
    CoralBricksMemory,
    get_tools,
    set_global_memory,
)

client = CoralBricksClient(api_key="your_coralbricks_api_key")
memory = CoralBricksMemory(client=client, project_id="my-app", session_id="user-123")

set_global_memory(memory)
tools = get_tools()  # [store_coralbricks_memory, search_coralbricks_memory, delete_coralbricks_memory]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_agent(
    model,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant with persistent memory (CoralBricks). "
        "Before answering, always search memory for relevant context. "
        "When you learn something important, store it."
    ),
)

result = agent.invoke({"messages": [{"role": "user", "content": "Remember: Alex is on the Enterprise plan."}]})
print(result["messages"][-1].content)
```

---

## Retriever (LCEL chains)

`CoralBricksRetriever` implements `BaseRetriever` and drops into any LCEL chain:

```python
from coralbricks_langchain import CoralBricksClient, CoralBricksMemory, CoralBricksRetriever

retriever = CoralBricksRetriever(
    memory=CoralBricksMemory(client=client, project_id="my-kb"),
    top_k=5,
)

docs = retriever.invoke("What is the cancellation policy?")
for doc in docs:
    print(doc.page_content)
```

---

## Chat message history

Persistent chat history backed by the CoralBricks chat API:

```python
from coralbricks_langchain import CoralBricksClient, CoralBricksChatMessageHistory

client = CoralBricksClient(api_key="your_coralbricks_api_key")
history = CoralBricksChatMessageHistory(client=client, conversation_id="conv-001")

history.add_user_message("Hello!")
history.add_ai_message("Hi, how can I help?")

for msg in history.messages:
    print(msg.type, msg.content)
```

---

## Public API

| Symbol | Description |
|--------|-------------|
| `CoralBricksClient` | Low-level HTTP client for the CoralBricks Memory API |
| `CoralBricksMemory` | High-level helper: `save_memory`, `search_memory`, `delete_memory` |
| `CoralBricksRetriever` | LangChain `BaseRetriever` for LCEL RAG pipelines |
| `CoralBricksChatMessageHistory` | LangChain `BaseChatMessageHistory` backed by CoralBricks chat storage |
| `store_coralbricks_memory` | Agent tool — embed and store a memory item |
| `search_coralbricks_memory` | Agent tool — semantic search over memories |
| `delete_coralbricks_memory` | Agent tool — delete memories by ID |
| `set_global_memory` | Configure the global `CoralBricksMemory` instance used by tools |
| `get_tools` | Factory returning all three memory tools |

---

## Conventions

| Field | Example | Purpose |
|-------|---------|---------|
| `project_id` | `langchain:my-app` | App/use-case namespace (shared across agents) |
| `session_id` | `user-123`, `conv-001` | Conversation or user scope |
| `metadata` | `{"source": "policy"}` | Optional metadata stored with each item |

---

## License

Apache-2.0.
