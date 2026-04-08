"""
Event Scout
-----------
A small example agent that scrapes upcoming in-person tech / AI events
from Luma + Eventbrite using TinyFish browser automation, and persists
the results in CoralBricks memory for dedup and re-use across runs.

Usage:
    python event_scout.py                   # next 30 days, both cities
    python event_scout.py --days 7          # next 7 days
    python event_scout.py --city sf         # one city
    DEBUG=1 python event_scout.py           # print raw TinyFish SSE events

Env vars (.env or shell):
    TINYFISH_API_KEY        required — https://tinyfish.io
    CORALBRICKS_API_KEY     required to persist events
    CORALBRICKS_BASE_URL    optional, defaults to https://memory.coralbricks.ai
    TINYFISH_MAX_CONCURRENCY  optional, default 5

This is the open-source companion to the CoralBricks events dashboard.
The dashboard UI lives in a separate (closed) repo; this script is the
scraper + memory side of the same system, and you can adapt it to your
own event sources or topic filters.
"""

import os
import ast
import json
import argparse
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")

TINYFISH_API_KEY = os.getenv("TINYFISH_API_KEY", "")
TINYFISH_URL = "https://agent.tinyfish.ai/v1/automation/run-sse"
TINYFISH_MAX_CONCURRENCY = int(os.getenv("TINYFISH_MAX_CONCURRENCY", "5"))

CORALBRICKS_API_KEY = os.getenv("CORALBRICKS_API_KEY", "")
CORALBRICKS_BASE_URL = os.getenv("CORALBRICKS_BASE_URL", "https://memory.coralbricks.ai")
CORALBRICKS_PROJECT_ID = "events:scout"

# Drop "online" events — this scout is built for in-person discovery.
ONLINE_KEYWORDS = ["online", "virtual", "zoom", "webinar", "remote", "livestream", "live stream"]


# ── Goal template ─────────────────────────────────────────────────────────────

def build_goal(date_from: str, date_to: str) -> str:
    """The natural-language instruction we send to TinyFish for each page."""
    return (
        f"Today is {date_from}. Find upcoming in-person events on this page "
        f"happening between {date_from} and {date_to} that are specifically about: "
        "AI agents, agentic AI, LLM applications, autonomous agents, multi-agent systems, "
        "open-source AI, or AI infrastructure. "
        "Only include: meetups, conferences, hackathons, demo nights, and networking events "
        "focused on the topics above. "
        "Exclude: generic tech/startup events not focused on AI, "
        "online/virtual/webinar events, training courses, workshops, bootcamps, "
        "and career fairs. "
        "For each event return: name, date (DD Month YYYY), time (e.g. 6:00 PM or null if not shown), "
        "location (venue name and city), organizer (if visible), a 1-sentence description, and the event URL. "
        "Return as a JSON array with keys: name, date, time, location, organizer, description, url."
    )


def build_sources(date_from: str, date_to: str, city_filter: str) -> list[dict]:
    """Build the source list with date-aware Eventbrite URLs.
    Edit this function to point at your own cities or platforms."""
    eb_date = f"start_date.range_start={date_from}&start_date.range_end={date_to}"
    goal = build_goal(date_from, date_to)

    all_sources = [
        {"city": "sf", "label": "SF Bay Area",
         "url": "https://luma.com/sf?q=ai+agents", "goal": goal},
        {"city": "seattle", "label": "Seattle",
         "url": "https://luma.com/seattle?q=ai+agents", "goal": goal},
        {"city": "sf", "label": "SF Bay Area",
         "url": f"https://www.eventbrite.com/d/ca--san-francisco/ai-agents--events/?{eb_date}",
         "goal": goal},
        {"city": "seattle", "label": "Seattle",
         "url": f"https://www.eventbrite.com/d/wa--seattle/ai-agents--events/?{eb_date}",
         "goal": goal},
    ]
    return [s for s in all_sources if city_filter == "all" or s["city"] == city_filter]


# ── Date parsing + filtering ──────────────────────────────────────────────────

DATE_FORMATS = [
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%d/%m/%Y", "%Y-%m-%d",
]


def parse_event_date(date_str: str) -> datetime | None:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def filter_by_date(events: list[dict], date_from: datetime, date_to: datetime) -> list[dict]:
    out = []
    for ev in events:
        parsed = parse_event_date(ev.get("date", ""))
        if parsed is None:
            continue
        if date_from <= parsed <= date_to:
            out.append(ev)
    return out


def is_online_event(ev: dict) -> bool:
    location = (ev.get("location") or "").lower()
    name = (ev.get("name") or "").lower()
    return any(kw in location or kw in name for kw in ONLINE_KEYWORDS)


def filter_events(events: list[dict]) -> list[dict]:
    return [ev for ev in events if not is_online_event(ev)]


# ── TinyFish scraper ──────────────────────────────────────────────────────────

def tinyfish_scrape(url: str, goal: str, timeout: float = 90.0) -> str:
    """Stream a TinyFish browser automation run for one URL.
    Returns the last content chunk from the SSE stream."""
    print(f"  [tinyfish] {url[:80]}...", flush=True)
    last_content = ""
    with httpx.stream(
        "POST",
        TINYFISH_URL,
        headers={
            "X-API-Key": TINYFISH_API_KEY,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        json={"url": url, "goal": goal},
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            line = line.strip()
            if DEBUG:
                print(f"    [raw] {line[:200]}", flush=True)
            if not line:
                continue
            raw = line[5:].strip() if line.startswith("data:") else line
            if raw in ("[DONE]", ""):
                break
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                if line and not line.startswith("event:") and not line.startswith(":"):
                    last_content = line
                continue
            etype = event.get("type", "")
            result = event.get("result")
            if etype == "COMPLETE" and isinstance(result, (dict, list)):
                last_content = json.dumps(result)
            else:
                content = (
                    event.get("content") or event.get("text") or event.get("output")
                    or result or event.get("message") or event.get("answer") or ""
                )
                if content:
                    last_content = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
    return last_content


def extract_json_array(text: str) -> list[dict]:
    """Best-effort parser for the LLM/TinyFish output. Tolerates wrapped
    objects, single quotes, and stray prose around the JSON."""
    text = text.strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            for v in obj.values():
                if isinstance(v, list):
                    return v
        except json.JSONDecodeError:
            pass
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return v
    except (ValueError, SyntaxError):
        pass
    for sc, ec in (("[", "]"), ("{", "}")):
        s, e = text.find(sc), text.rfind(ec)
        if s != -1 and e > s:
            try:
                obj = ast.literal_eval(text[s : e + 1])
                if isinstance(obj, list):
                    return obj
                if isinstance(obj, dict):
                    for v in obj.values():
                        if isinstance(v, list):
                            return v
            except (ValueError, SyntaxError):
                pass
    return []


# ── CoralBricks memory layer ──────────────────────────────────────────────────

def _event_key(name: str, date: str, city: str) -> str:
    return f"{name.strip().lower()}|{date.strip()}|{city.strip().lower()}"


def load_existing_event_keys() -> set[str]:
    """Fetch the events we've already saved in prior runs so we can dedup
    before sending writes. Returns a set of name|date|city keys."""
    if not CORALBRICKS_API_KEY:
        return set()
    try:
        resp = httpx.get(
            f"{CORALBRICKS_BASE_URL}/v1/memory/list",
            params={"project_id": CORALBRICKS_PROJECT_ID, "limit": "1000"},
            headers={"x-api-key": CORALBRICKS_API_KEY},
            timeout=30,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        keys: set[str] = set()
        for item in items:
            d = item.get("data") or {}
            keys.add(_event_key(str(d.get("name", "")), str(d.get("date", "")), str(d.get("city", ""))))
        print(f"[coralbricks] loaded {len(keys)} existing events for dedup")
        return keys
    except Exception as e:
        print(f"[coralbricks] could not load existing events: {e}")
        return set()


def save_events_to_coralbricks(city_label: str, events: list[dict], existing_keys: set[str]) -> int:
    """Save new events to CoralBricks memory. Mutates existing_keys to
    record what we just wrote so the rest of the run won't duplicate."""
    if not CORALBRICKS_API_KEY:
        return 0
    saved = 0
    for ev in events:
        key = _event_key(ev.get("name", ""), ev.get("date", ""), city_label)
        if key in existing_keys:
            continue
        try:
            data = {
                "name": ev.get("name", ""),
                "date": ev.get("date", ""),
                "time": ev.get("time") or "",
                "city": city_label,
                "location": ev.get("location") or "",
                "organizer": ev.get("organizer") or "",
                "description": ev.get("description") or "",
                "url": ev.get("url") or "",
                "source": "scraped",
                "timestamp": datetime.now().isoformat(),
            }
            resp = httpx.post(
                f"{CORALBRICKS_BASE_URL}/v1/memory/save",
                headers={"x-api-key": CORALBRICKS_API_KEY},
                json={
                    "project_id": CORALBRICKS_PROJECT_ID,
                    "session_id": city_label,
                    "items": [{"data": data}],
                },
                timeout=30,
            )
            resp.raise_for_status()
            existing_keys.add(key)
            saved += 1
        except Exception as e:
            print(f"  [coralbricks] error saving '{ev.get('name')}': {e}")
    return saved


# ── Display ───────────────────────────────────────────────────────────────────

BOLD = "\033[1m"; CYAN = "\033[96m"; YELLOW = "\033[93m"; DIM = "\033[2m"; RESET = "\033[0m"


def print_events(city_label: str, events: list[dict]) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  {city_label}  ({len(events)} events){RESET}")
    print(f"{BOLD}{CYAN}{'─' * 70}{RESET}")
    if not events:
        print(f"  {DIM}No events found.{RESET}")
        return
    for i, ev in enumerate(events, 1):
        print(f"\n  {BOLD}{i}. {ev.get('name', 'Unknown')}{RESET}")
        print(f"     {YELLOW}{ev.get('date', 'TBD')}{RESET}  {DIM}|  {ev.get('location', '')}{RESET}")
        if ev.get("description"):
            print(f"     {ev['description']}")
        if ev.get("url"):
            print(f"     {DIM}{ev['url']}{RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape upcoming in-person tech / AI events")
    parser.add_argument("--city", choices=["seattle", "sf", "all"], default="all")
    parser.add_argument("--days", type=int, default=30, help="Days ahead to search (default: 30)")
    args = parser.parse_args()

    if not TINYFISH_API_KEY:
        print("error: TINYFISH_API_KEY is not set. Get one at https://tinyfish.io")
        return

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = today + timedelta(days=args.days)
    date_from = today.strftime("%Y-%m-%d")
    date_to = end_date.strftime("%Y-%m-%d")

    sources = build_sources(date_from, date_to, args.city)

    print(f"\n{BOLD}Event Scout — {today.strftime('%B %d, %Y')}{RESET}")
    print(f"Window: next {args.days} days ({date_from} → {date_to})")
    print(f"Sources: {len(sources)}  |  CoralBricks: {'on' if CORALBRICKS_API_KEY else 'off (results not saved)'}\n")

    city_events: dict[str, list[dict]] = {}
    cb_existing_keys = load_existing_event_keys()
    total_cb_saved = 0

    def fetch_source(source: dict) -> tuple[str, list[dict]]:
        label = source["label"]
        print(f"scraping {label}: {source['url'][:70]}...")
        raw = tinyfish_scrape(source["url"], source["goal"])
        events = extract_json_array(raw)
        events = filter_by_date(events, today, end_date)
        events = filter_events(events)
        return label, events

    with ThreadPoolExecutor(max_workers=min(TINYFISH_MAX_CONCURRENCY, len(sources))) as executor:
        futures = {executor.submit(fetch_source, s): s for s in sources}
        for future in as_completed(futures):
            label, events = future.result()
            if label not in city_events:
                city_events[label] = []
            existing = {e.get("name", "").lower() for e in city_events[label]}
            new_events = []
            for ev in events:
                if ev.get("name", "").lower() not in existing:
                    city_events[label].append(ev)
                    existing.add(ev.get("name", "").lower())
                    new_events.append(ev)
            if new_events and CORALBRICKS_API_KEY:
                n = save_events_to_coralbricks(label, new_events, cb_existing_keys)
                total_cb_saved += n
                print(f"  [coralbricks] +{n} new events saved ({label})")

    if CORALBRICKS_API_KEY:
        print(f"\n[coralbricks] done — {total_cb_saved} new events saved total.")

    for label in city_events:
        city_events[label].sort(key=lambda e: parse_event_date(e.get("date", "")) or datetime.max)

    total = sum(len(evs) for evs in city_events.values())
    for city_label, events in city_events.items():
        print_events(city_label, events)

    print(f"\n{DIM}total: {total} events in the next {args.days} days{RESET}")


if __name__ == "__main__":
    main()
