"""Per-model generation speed: output tok/s and estimated TTFT distributions.

Log mechanics this leans on: Claude Code appends one JSONL entry per content
block, stamped when that block finishes streaming. So for each API turn we know
t0 = the entry before the turn (request sent), t1 = first block complete,
tN = last block complete, and the turn's `usage.output_tokens`.
- tok/s = output_tokens / (tN - t0). End-to-end: includes TTFT and thinking.
- TTFT isn't logged. Estimate = (t1 - t0) - first_block_tokens / decode_rate,
  with decode_rate measured on the rest of the turn (post-first-block tokens
  over tN - t1) and block tokens approximated as chars/3.8. Single-block turns
  carry no decode-rate sample and are skipped for TTFT.
Turns under 50 output tokens are skipped (tool-ack noise), as are gaps over
600s (user idle, not the model).

Run: python3 speed.py [days]   (default 30; no dependencies)
Nothing leaves your machine; it only reads local files.
"""
import json, glob, os, sys, collections
from datetime import datetime, timedelta, timezone

DAYS = float(sys.argv[1]) if len(sys.argv) > 1 else 30
CUTOFF = (datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp()
CHARS_PER_TOK = 3.8

def ts(o):
    t = o.get("timestamp")
    if not t: return None
    try: return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
    except: return None

def block_toks(m):
    n = 0
    for b in m.get("content") or []:
        if isinstance(b, dict):
            n += len(b.get("thinking") or "") + len(b.get("text") or "")
            if b.get("type") == "tool_use": n += len(json.dumps(b.get("input") or {}))
    return n / CHARS_PER_TOK

tps = collections.defaultdict(list); ttfts = collections.defaultdict(list)
ins = collections.defaultdict(list); outs = collections.defaultdict(list)
tot = collections.defaultdict(lambda: [0, 0.0])
files = glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")) + \
        glob.glob(os.path.expanduser("~/.claude/projects/*/*/subagents/*.jsonl"))
scanned = 0
for f in files:
    if os.path.getmtime(f) < CUTOFF: continue
    scanned += 1
    evs = []
    for line in open(f):
        try: o = json.loads(line)
        except: continue
        if o.get("type") in ("user", "assistant") and ts(o) is not None: evs.append(o)
    turns = collections.defaultdict(list)
    for i, o in enumerate(evs):
        if o["type"] == "assistant":
            mid = (o.get("message") or {}).get("id")
            if mid: turns[mid].append(i)
    for idxs in turns.values():
        first, last = idxs[0], idxs[-1]
        m = evs[last]["message"]
        t_end = ts(evs[last])
        if t_end < CUTOFF or first == 0: continue
        u = m.get("usage") or {}
        out = u.get("output_tokens", 0) or 0
        t0, t1 = ts(evs[first - 1]), ts(evs[first])
        total = t_end - t0
        if out < 50 or total <= 1 or total > 600: continue
        model = m.get("model", "?")
        tps[model].append(out / total); tot[model][0] += out; tot[model][1] += total
        ins[model].append(sum(u.get(k, 0) or 0 for k in
            ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")))
        outs[model].append(out)
        if len(idxs) > 1 and t_end - t1 > 0.5:
            ftoks = block_toks(evs[first]["message"])
            rate = max(out - ftoks, 1) / (t_end - t1)
            ttfts[model].append(max(t1 - t0 - ftoks / rate, 0.05))

def pct(v, p): return v[min(len(v) - 1, round(p / 100 * (len(v) - 1)))]
print(f"{scanned} session files scanned (last {DAYS:g} days)\n")
print("── p50 / p99 per turn (in/out = tokens, ttft = seconds, tok/s = end-to-end incl. TTFT + thinking) ──")
hdr = f"{'model':<28}{'msgs':>6}" + "".join(f"{c:>11}" for c in
      ("in p50", "in p99", "out p50", "out p99", "ttft p50", "ttft p99", "tok/s p1", "tok/s p50", "tok/s p99"))
print(hdr); print("-" * len(hdr))
for mo in sorted(tps, key=lambda mo: -len(tps[mo])):
    row = f"{mo:<28}{len(tps[mo]):>6}"
    for v in (sorted(ins[mo]), sorted(outs[mo])):
        row += f"{pct(v,50):>11,}{pct(v,99):>11,}"
    t = sorted(ttfts[mo])
    row += f"{pct(t,50):>11.2f}{pct(t,99):>11.2f}" if t else f"{'—':>11}{'—':>11}"
    v = sorted(tps[mo])
    row += f"{pct(v,1):>11.1f}{pct(v,50):>11.1f}{pct(v,99):>11.1f}"
    print(row)
print()
