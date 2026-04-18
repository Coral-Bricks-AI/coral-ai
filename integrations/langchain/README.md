# coralbricks-langchain

Use **CoralBricks** as the memory backend for your LangChain applications: retriever, agent memory tools, and RAG middleware — all backed by the CoralBricks Memory API.

- Drop-in **`CoralBricksRetriever`** for any LCEL chain or RAG pipeline.
- Three **agent tools** (`store`, `search`, `forget`) for persistent memory across turns.
- **Memory stores** — each store gets a dedicated index. Share memory across agents via the same store name.

---

## Installation

```bash
pip install coralbricks-langchain
```

Requires Python 3.10+ and LangChain >= 1.0.

---

## API key

Get a CoralBricks API key from the [CoralBricks web app](https://coralbricks.ai).

---

## Quick start

```python
from coralbricks_langchain import CoralBricksMemory

memory = CoralBricksMemory(api_key="your_coralbricks_api_key")
memory.get_or_create_memory_store("langchain:my-app")
memory.set_session_id("user-123")

# Save a memory
mem_id = memory.save_memory("Pro plan costs $199/month with unlimited ops.")

# Search by meaning
hits = memory.search_memory("What does the Pro plan cost?", top_k=3)
for h in hits:
    print(h.get("score"), h.get("text"))

# Forget by meaning
memory.forget_memory("Pro plan pricing")
```

---

## Agent memory tools

Give your agent tools to store, search, and forget memories across turns.
Pass the `memory` instance directly to `get_tools()` — no global state required.

```python
from coralbricks_langchain import CoralBricksMemory, get_tools
from langchain_openai import ChatOpenAI

memory = CoralBricksMemory(api_key="your_coralbricks_api_key")
memory.get_or_create_memory_store("langchain:support-agent")
memory.set_session_id("user-123")

tools = get_tools(memory)

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
from coralbricks_langchain import CoralBricksMemory, CoralBricksRetriever

memory = CoralBricksMemory(api_key="your_coralbricks_api_key")
memory.get_or_create_memory_store("langchain:my-kb")

retriever = CoralBricksRetriever(memory=memory, top_k=5)

docs = retriever.invoke("What is the cancellation policy?")
for doc in docs:
    print(doc.page_content)
```

---

## API reference

### `CoralBricksMemory`

| Method | Description |
|--------|-------------|
| `CoralBricksMemory(api_key, base_url?)` | Create memory instance (client created internally) |
| `.get_or_create_memory_store(name)` | Attach to or create a dedicated memory store (idempotent) |
| `.set_project_id(id)` | Set project namespace |
| `.set_session_id(id)` | Set session/user namespace |
| `.save_memory(text, metadata?)` | Embed and store a memory item |
| `.search_memory(query, top_k=5)` | Semantic search over memories |
| `.forget_memory(query, top_k=5)` | Forget memories matching a semantic query |

### Other components

| Symbol | Description |
|--------|-------------|
| `CoralBricksRetriever` | LangChain `BaseRetriever` for LCEL RAG pipelines |
| `get_tools(memory)` | Factory returning `[store, search, forget]` tools bound to a memory instance |

---

## Conventions

| Field | Example | Purpose |
|-------|---------|---------|
| `store_name` | `langchain:support-agent` | Dedicated index for this app/use-case |
| `session_id` | `user-123`, `conv-001` | Conversation or user scope |
| `metadata` | `{"source": "policy"}` | Optional metadata stored with each item |

---

## License

Apache-2.0.
