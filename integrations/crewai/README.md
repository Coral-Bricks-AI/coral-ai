# coralbricks-crewai

Use **CoralBricks** as the memory backend for your CrewAI agents: store and search memories over the CoralBricks Memory API.

## Installation

```bash
pip install coralbricks-crewai
```

Requires Python 3.10+, [CrewAI](https://github.com/crewai/crewai), and [crewai-tools](https://pypi.org/project/crewai-tools/).

## API key and base URL

- **API key:** Get a CoralBricks API key from the [CoralBricks web app](https://coralbricks.ai). Use it for all requests to the Memory API.
- **Base URL:** The CoralBricks Memory API base URL (e.g. `http://54.90.249.165` or the URL of your deployed memory-api instance).

## Quick start: client and memory

```python
from coralbricks_crewai import CoralBricksClient, CoralBricksMemory

client = CoralBricksClient(
    api_key="your_coralbricks_api_key",
    base_url="http://54.90.249.165",
)

memory = CoralBricksMemory(
    client,
    project_id="crewai:my-app",   # optional, for namespacing
    session_id="conv-123",         # optional, e.g. conversation or user id
)

# Save a memory (embed + store)
memory.save_memory("Cancellations within 24h of check-in incur a $50 fee.", metadata={"source": "policy_pdf"})

# Search by meaning
hits = memory.search_memory("What is the cancellation fee?", top_k=3)
for h in hits:
    print(h.get("score"), h.get("text"))
```

## CrewAI tool: search CoralBricks memory

Give your agents a tool that searches CoralBricks memory by natural language:

```pythonaf
from crewai import Agent, Task, Crew, Process
from coralbricks_crewai import (
    CoralBricksClient,
    CoralBricksMemory,
    search_coralbricks_memory,
    set_global_memory,
)

client = CoralBricksClient(
    api_key="your_coralbricks_api_key",
    base_url="http://54.90.249.165",
)
memory = CoralBricksMemory(client, project_id="crewai:my-app", session_id="conv-123")
set_global_memory(memory)

agent = Agent(
    role="Researcher",
    goal="Discover insights about AI agents",
    backstory="Expert researcher",
    tools=[search_coralbricks_memory()],
)

task = Task(
    description="Explain how agents collaborate and how CoralBricks can store their long-term knowledge.",
    expected_output="A concise explanation that mentions persistent memory.",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
result = crew.kickoff()
print(result)
```

You must call `set_global_memory(memory)` once before running the crew so the tool can access your CoralBricks memory instance.

## Conventions

| Field        | Example                | Purpose                                  |
|-------------|------------------------|------------------------------------------|
| `project_id`| `crewai:hotel-support` | App or use case (one per crew/app).      |
| `session_id`| `conv-123`, `user-42`  | Conversation or user scope.              |
| `metadata`  | `{"source": "policy"}` | Optional metadata stored with the item.  |

## License

Apache-2.0.
