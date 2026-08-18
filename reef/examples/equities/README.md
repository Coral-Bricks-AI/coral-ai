# Equities — the Reef hello-world

The simplest end-to-end use of Reef: **one specialist, three skills, ~20 well-known tickers of data, ~50 lines of glue.** No planner, no synthesizer, no `SpecialistConfig`. Just `run_react()` wired to a persona prompt and two skill-dispatch tools, optionally scoped by a `HarnessConstraints` cutoff.

If you've read the [Reef write-up](https://coralbricks.ai/blog/write-a-winning-agent-harness), this is the worked code behind it. For the full-scale version of the same primitives, see [`alphacumen/`](../../../alphacumen) — 7 specialists, 69 skills, a planner, and runtime constraints.

> **Data is mock.** `data/companies.json` is ~20 well-known tickers with **fabricated point-in-time prices** — a monthly price grid from 2024-07-15 to 2026-07-15 plus a few filings each. The numbers are internally consistent and plausible, but not real market data. Don't use this example to make decisions — it exists to show the Reef wiring.

## Run it

```bash
git clone https://github.com/Coral-Bricks-AI/coral-ai.git
cd coral-ai
pip install -e .
export LLM_API_KEY=sk-...

python reef/examples/equities/ask.py "How has NVDA performed over the last year?"
```

Sample queries:

```bash
python reef/examples/equities/ask.py "Which money-center banks are in the corpus?"
python reef/examples/equities/ask.py "What does Moderna do and how has the stock done?"
python reef/examples/equities/ask.py "Compare AMD and INTC over the last 12 months."
python reef/examples/equities/ask.py "What did NVDA report most recently, and how had the stock done at that point?"
```

Any provider Reef supports works: pass `--model <provider>/<model>` (e.g., `--model anthropic/claude-sonnet-4-6`, `--model together/kimi-k2.6`, `--model aws/anthropic.claude-3-5-sonnet`) and set the matching env var (`ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, AWS creds, etc.).

### Scoping the run with `--asof`

Pass `--asof YYYY-MM-DD` to enable Reef's `HarnessConstraints` + `LocalEnforcer` for the run. The `@time_bounded` decorators on `list_recent_filings` (clamp) and `compute_total_return` (validate) then engage automatically:

```bash
# The filings skill will silently clamp its as_of_iso to 2025-12-31.
# If the model tries to compute a return past 2025-12-31, the enforcer
# raises AsofViolation and the model gets a tool-error envelope.
python reef/examples/equities/ask.py --asof 2025-12-31 \
    "What did NVDA report last, and how had the stock done at that point?"
```

## What's on disk

```
examples/equities/
├── ask.py                     # runner — calls run_react() (optionally under harness_context)
├── analyst.md                 # the system prompt (with {skill_index} placeholder)
├── data/companies.json        # the corpus (20 mock companies + monthly price_history + filings)
└── skills/
    ├── search_companies/
    │   ├── SKILL.md           # routing playbook the model reads
    │   └── impl.py            # @skill_fn-decorated Python the runtime calls
    ├── compute_total_return/
    │   ├── SKILL.md
    │   └── impl.py            # @time_bounded(mode="validate") + @skill_fn
    └── list_recent_filings/
        ├── SKILL.md
        └── impl.py            # @time_bounded(mode="clamp", filter_field=...) + @skill_fn
```

| File | Role |
|---|---|
| [`data/companies.json`](data/companies.json) | The corpus — 20 tickers with name, sector, description, a monthly `price_history` (2024-07 → 2026-07) and a `filings` array |
| [`skills/search_companies/`](skills/search_companies/) | BM25 search over ticker + name + sector + description |
| [`skills/compute_total_return/`](skills/compute_total_return/) | Trailing 1-year price return; **`@time_bounded(mode="validate")`** on `as_of_iso` |
| [`skills/list_recent_filings/`](skills/list_recent_filings/) | k most recent filings for a ticker up to a date; **`@time_bounded(mode="clamp", filter_field="filing_date")`** |
| [`analyst.md`](analyst.md) | The specialist's system prompt — renders the skill index inline, documents the temporal contract |
| [`ask.py`](ask.py) | Runner. Calls `reef.react.run_react()` directly with the analyst persona + two dispatch tools; wraps in `harness_context(...)` when `--asof` is set |

## One skill, end to end

Two files, sharing a slug. Markdown for the model, Python for the runtime.

[`skills/search_companies/SKILL.md`](skills/search_companies/SKILL.md):

```markdown
---
id: search_companies
when: Find companies by ticker, name, sector, or any free-text descriptor.
      Use FIRST when the user names a company or describes a sector.
applies_to: [equity_analyst]
---

Call `search_companies(query=<free text>, k=<int, default 5>)`.

Returns a ranked list of `{"ticker", "name", "sector", "score"}`.
After search, if the question is quantitative, follow up with
`compute_total_return` using the top result's `ticker`.
```

[`skills/search_companies/impl.py`](skills/search_companies/impl.py):

```python
from reef.skill_fn import skill_fn

@skill_fn(
    skill_id="search_companies",
    description="Rank companies by BM25 over ticker + name + sector + description.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
)
def search_companies(*, query: str, k: int = 5):
    ...  # BM25 over the corpus
    return {"query": query, "results": results}
```

The decorator registers the callable in a process-global registry at import time. The model dispatches by id — `invoke_skill_fn(skill_id="search_companies", fn="search_companies", args={...})` — and the runtime runs your Python.

## Skills load lazily

The model sees only a one-line *index* of every skill in its system prompt:

```
- search_companies     — Find companies by ticker, name, sector, or descriptor. Use FIRST...
- compute_total_return — Trailing 1-year price return for a ticker.
```

To use one, it calls `load_skill(skill_ids=["search_companies"])` and the body of `SKILL.md` plus the JSON Schema for `invoke_skill_fn` get spliced into the thread. Seventy skills indexed cost ~70 lines of context; only the loaded bodies pay tokens.

## Declarative time constraints (`@time_bounded` + `@skill_fn`)

The `list_recent_filings` and `compute_total_return` skills stack two decorators. The pattern is worth reading closely because it's how Reef stops LLMs from confidently reading past a domain freeze date:

```python
from reef.decorators import time_bounded
from reef.skill_fn import skill_fn

@time_bounded(asof_arg="as_of_iso", filter_field="filing_date", mode="clamp")
@skill_fn(
    skill_id="list_recent_filings",
    description="Return the k most recent SEC filings for a ticker, on or before as_of_iso...",
    parameters={
        "type": "object",
        "properties": {
            "ticker":    {"type": "string", "description": "..."},
            "as_of_iso": {"type": "string", "description": "..."},
            "k":         {"type": "integer", "default": 5},
        },
        "required": ["ticker"],
    },
)
def list_recent_filings(*, ticker, as_of_iso=None, k=5):
    ...  # unaware of the constraint — reads the whole corpus, returns rows.
```

**Two independent contracts, composed:**

- `@skill_fn` registers the callable so `invoke_skill_fn(skill_id=..., fn=..., args=...)` can dispatch it.
- `@time_bounded` stamps a `TimeBound(asof_arg, filter_field, mode)` onto the function so the runtime knows which arg carries the cutoff and which result-row field is the date.

**Three modes:**

| mode | behavior | used by |
|---|---|---|
| `inject` | Enforcer overwrites the model's arg with `constraints.asof`. Use when the model doesn't get to choose. | (not in this example) |
| `clamp` | Enforcer uses `min(model_value, constraints.asof)`. Use when the model may pick a narrower cutoff. | `list_recent_filings` |
| `validate` | Enforcer raises `AsofViolation` if the model passed a past-asof value. Use when quietly rewriting would hide the misuse. | `compute_total_return` |

**How it actually fires.** `ask.py` wraps the `run_react` call in `harness_context(HarnessConstraints(asof=..., tool_budget=...))` when `--asof` is passed. That binds the constraints + a `LocalEnforcer` for the run via `contextvars`. When the model dispatches through `invoke_skill_fn`, the framework reaches through to the inner `@skill_fn` callable, reads its `__time_bound__`, and applies the mode (inject / clamp / validate) plus the post-filter on `filter_field` — all before the skill body runs.

Sample failure envelope the model sees when `compute_total_return` is called with a future `as_of_iso`:

```json
{
  "error": "AsofViolation: compute_total_return.compute_total_return: arg 'as_of_iso'='2027-01-01' is past asof=2025-12-31. Use asof or an earlier date.",
  "skill_id": "compute_total_return",
  "fn": "compute_total_return",
  "arguments": {"ticker": "NVDA", "as_of_iso": "2027-01-01"},
  "constraint_violation": true
}
```

The model reads the message, corrects its `as_of_iso`, and retries. The skill body never ran; no fabricated future-price answer to walk back.

## What this example does NOT use

Deliberately. Once you scale past one specialist:

- **Planner / synthesizer / `swarm.run()`** — orchestrates multi-specialist runs, dispatches in parallel, prunes between rounds, writes the final structured envelope. See [`alphacumen/swarm.py`](../../../alphacumen/swarm.py).
- **`SpecialistConfig`** — wraps one specialist's persona + tool roster + per-call budget for the planner to dispatch to.
- **Full `HarnessConstraints` surface** — this example uses `asof` + `tool_budget`. Production adds `allowed_indices`, `token_budget`, `max_rounds`.
- **Real retrieval** — production AlphaCumen pulls from EDGAR + a half-dozen indexed corpora; this example reads one in-memory JSON.

When you have one specialist over a 20-row corpus, none of that buys you anything. When you have six specialists arguing across thousands of filings, all of it does.

## Where to go next

- [The Reef write-up](https://coralbricks.ai/blog/write-a-winning-agent-harness) — design rationale walked one section per primitive
- [`reef/`](../..) — the framework itself; read [`react.py`](../../react.py) and [`skill_fn.py`](../../skill_fn.py) to see how this hello-world hangs together
- [`alphacumen/`](../../../alphacumen) — the worked finance instance: 7 specialists, 69 skills, the planner + synthesizer scaffolding. Same primitives at a much larger scale.
