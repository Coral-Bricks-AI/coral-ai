# coralbricks-crewai

Use **CoralBricks** as the memory backend for your CrewAI agents: store and search semantic memories over the Coralbricks Memory API.

- Keep **long‑lived knowledge** across runs and sessions (policies, FAQs, user preferences, summaries).
- Share memory across **multiple crews/agents** via `project_id`.
- Keep using your existing **LLM** (OpenAI, etc.) – Coralbricks only replaces the memory/KB layer.

---

## Installation

From PyPI:

```bash
pip install coralbricks-crewai
```

Requires Python 3.10+ and [CrewAI](https://github.com/crewai/crewai).

---

## API key and base URL

- **API key:** Get a Coralbricks API key from the [Coralbricks web app](https://coralbricks.ai). Use it for all requests to the Memory API.
- **Base URL:** Use the Coralbricks CrewAI Memory API: `https://cw.coralbricks.ai`

Environment variables (optional but convenient):

```bash
export CORALBRICKS_API_KEY="your_coralbricks_api_key"
export CORAL_MEMORY_BASE_URL="https://cw.coralbricks.ai"
```

---

## Quick start: client and memory

This is the simplest "just store and search nuggets" flow (no CrewAI yet):

```python
from coralbricks_crewai import CoralBricksClient, CoralBricksMemory

client = CoralBricksClient(
    api_key="your_coralbricks_api_key",
    base_url="https://cw.coralbricks.ai",
)

memory = CoralBricksMemory(
    client=client,
    project_id="crewai:my-app",    # logical app/use-case namespace
    session_id="conv-123",         # conversation or user id
)

# Save a memory (embed + store)
memory.save_memory(
    "Cancellations within 24h of check-in incur a $50 fee.",
    metadata={"source": "policy_pdf"},
)

# Search by meaning
hits = memory.search_memory("What is the cancellation fee?", top_k=3)
for h in hits:
    print(h.get("score"), h.get("text"))
```

Under the hood this calls the Coralbricks Memory API at `https://cw.coralbricks.ai` (`/v1/memory/save`, `/v1/memory/query`).

---

## Before vs after: what Coralbricks adds

**Without Coralbricks:** A CrewAI travel agent gives a **generic** 2-day Tokyo itinerary—same for every user.

**With Coralbricks:** You store a **memory** (e.g. “Team prefers staying near Shibuya station, loves ramen, hates long queues”). The agent **searches** that memory and returns a **personalized** itinerary. You’ll see phrases that only appear because of memory, for example:

- *“Location near **Shibuya** matches your preference.”*
- *“You can have **ramen** for breakfast.”*
- *“**Short queues**” or “avoid long waits.”*

So: **without Coral** → generic answer; **with Coral** → same agent, but it **recalls preferences** and weaves them into the plan. The only extra pieces are: **CoralBricks client + memory**, **`set_global_memory(memory)`**, and the **`search_coralbricks_memory()`** tool on the agent.

---

## CrewAI tool: search Coralbricks memory

Give your agents a tool that searches Coralbricks memory by natural language:

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

from coralbricks_crewai import (
    CoralBricksClient,
    CoralBricksMemory,
    search_coralbricks_memory,
    set_global_memory,
)

# 1. LLM (CrewAI still uses your LLM; Coralbricks only handles memory)
llm = ChatOpenAI(model="gpt-4o-mini")

# 2. Coralbricks client + memory
client = CoralBricksClient(
    api_key="your_coralbricks_api_key",
    base_url="https://cw.coralbricks.ai",
)
memory = CoralBricksMemory(
    client=client,
    project_id="crewai:my-app",
    session_id="conv-123",
)

# Make this memory available to the search tool
set_global_memory(memory)
memory_search_tool = search_coralbricks_memory()

# Optionally seed some knowledge
memory.save_memory("Support hours: Mon–Fri 9am–6pm EST. Emergency line 24/7.")
memory.save_memory("Refund policy: full refund within 30 days; then prorated.")

# 3. Agent that can use the Coralbricks search tool
agent = Agent(
    role="Support assistant",
    goal="Answer user questions using stored policies and FAQs.",
    backstory="You search Coralbricks memory for relevant nuggets before answering.",
    tools=[memory_search_tool],
    llm=llm,
)

task = Task(
    description=(
        "The user asks: 'What are your support hours and what is your refund policy?'\n"
        "1. Use the Coralbricks memory search tool to find support hours and refund policy.\n"
        "2. Answer in 2–3 short sentences, based only on what you found."
    ),
    expected_output="A short answer that cites support hours and refund policy from memory.",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)

if __name__ == "__main__":
    result = crew.kickoff()
    print(result)
```

**Important:** you must call `set_global_memory(memory)` once before running the crew so the `search_coralbricks_memory` tool knows which `CoralBricksMemory` instance to use.

---

## 2‑hop example (hotel → cancellation policy)

This pattern shows Coralbricks being used as **long‑term memory** across steps:

1. **Hop 1** – user: "I want to book a hotel in Tokyo for 2 nights."  
   Agent chooses a specific hotel and booking ref, then calls a **save tool** that does:
   ```python
   memory.save_memory("Selected hotel: Hotel Tokyo Plaza, ref XYZ123, check-in March 15.")
   ```

2. **Hop 2** – user: "What's your cancellation policy?"  
   Agent calls `search_coralbricks_memory("cancellation policy")` to get the policy nugget, and also searches for the saved booking.  
   Answer might be:

   > "For your booking at Hotel Tokyo Plaza (ref XYZ123, March 15), cancellations within 24 hours incur a $50 fee; earlier cancellations are fully refunded."

---

## Conventions

| Field         | Example                   | Purpose                                    |
|---------------|---------------------------|--------------------------------------------|
| `project_id`  | `crewai:hotel-support`    | App/use‑case namespace (often shared).      |
| `session_id`  | `conv-123`, `user-42`     | Conversation or user scope.                |
| `metadata`    | `{"source": "policy"}`    | Optional metadata stored with the item.    |

Multiple crews can share the same memory by using the same `project_id` (and possibly different `session_id`s).

---

## How this changes CrewAI behavior

Without Coralbricks:

- Agents are mostly **stateless** across runs, or rely on ad‑hoc local storage.
- There is no easy, shared, semantic memory across crews.

With Coralbricks:

- You get a **remote semantic memory**:
  - `save_memory` / ingest → store nuggets and doc chunks.
  - `search_memory` / `search_coralbricks_memory` tool → retrieve by meaning.
- Crews can share **one memory space** (same `project_id`).
- You still bring your own LLM (OpenAI, etc.); Coralbricks only handles the **memory/KB** side.

---

## License

Apache-2.0.
