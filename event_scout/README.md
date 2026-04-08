# Event Scout

A small example agent that scrapes upcoming in-person tech / AI events from
Luma and Eventbrite using [TinyFish](https://tinyfish.io) browser automation,
and persists the results in [CoralBricks](https://coralbricks.ai) memory so
repeated runs dedup against everything you've already collected.

This is the open-source scraper + memory side of the CoralBricks events
dashboard. The dashboard UI is closed source, but the interesting bits — how
to point a web agent at a listing site, how to parse what comes back, and
how to persist it durably — are all in `event_scout.py`.

## What it does

1. Builds a list of source URLs (Luma + Eventbrite, configurable per city).
2. Fans out one TinyFish browser automation per URL with a natural-language
   goal that asks for in-person AI/agent events as a JSON array.
3. Parses each result tolerantly (LLM output isn't always clean JSON).
4. Filters out online/virtual events and anything outside the date window.
5. Loads previously-seen events from CoralBricks for dedup.
6. Writes new events back to CoralBricks under `project_id="events:scout"`.

## Quick start

```bash
pip install -r requirements.txt

export TINYFISH_API_KEY=...        # https://tinyfish.io
export CORALBRICKS_API_KEY=...     # https://coralbricks.ai

python event_scout.py                  # next 30 days, both cities
python event_scout.py --days 7         # narrower window
python event_scout.py --city sf        # one city
DEBUG=1 python event_scout.py          # raw TinyFish SSE events
```

## Environment variables

| name | required | default | purpose |
|---|---|---|---|
| `TINYFISH_API_KEY` | yes | — | TinyFish browser automation key |
| `CORALBRICKS_API_KEY` | no | — | If set, results persist to CoralBricks memory and dedup against prior runs. |
| `CORALBRICKS_BASE_URL` | no | `https://memory.coralbricks.ai` | Override only if you self-host. |
| `TINYFISH_MAX_CONCURRENCY` | no | `5` | How many sources to scrape in parallel. |
| `DEBUG` | no | — | Set to `1` to print raw TinyFish stream events. |

## Adapting it

- **Different cities or sources**: edit `build_sources()` — it's just a list
  of `{city, label, url, goal}` dicts.
- **Different topics**: edit `build_goal()` — that's the natural-language
  instruction TinyFish executes against each page.
- **Different storage**: replace `save_events_to_coralbricks` with your own
  sink. The dedup key is `name|date|city`.

## How CoralBricks fits in

Without persistence, every run is a cold start: you scrape the same events
over and over. With CoralBricks the script becomes useful as a daily cron —
each run only writes what's actually new, so the downstream consumer (a
dashboard, an email digest, an LLM that picks events to attend) only sees
diffs.

The dedup key is intentionally lossy (`name|date|city`) — it tolerates the
small variations between Luma's "Mar 15" and Eventbrite's "March 15, 2026"
once you normalize through `parse_event_date`. Tighten or relax it for your
own use case.
