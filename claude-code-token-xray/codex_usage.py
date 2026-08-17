#!/usr/bin/env python3
"""Analyze token usage in local Codex session rollouts.

Codex writes detailed JSONL session events below ``~/.codex/sessions``.  This
script sums the native per-model-call usage records, so its totals do not depend
on a third-party tokenizer.  It never sends session data anywhere.

Run:
    python3 codex_usage.py --days 7
    python3 codex_usage.py --start 2026-08-11 --end 2026-08-18
    python3 codex_usage.py --days 7 --format json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    def add(self, values: dict[str, Any]) -> None:
        self.calls += 1
        for field in TOKEN_FIELDS:
            value = values.get(field, 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                setattr(self, field, getattr(self, field) + int(value))

def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_boundary(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date/time {value!r}; use YYYY-MM-DD or ISO 8601"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed.astimezone(timezone.utc)


def _find_role(value: Any) -> str | None:
    """Best-effort role lookup across current and older thread_source shapes."""
    if not isinstance(value, dict):
        return None
    for key in ("agent_role", "agent_type", "role", "name", "nickname"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate and candidate not in {"user", "assistant"}:
            return candidate
    for child in value.values():
        role = _find_role(child)
        if role:
            return role
    return None


def classify_agent(meta: dict[str, Any]) -> str:
    """Return a useful primary/subagent label from session metadata."""
    source = str(meta.get("source") or "unknown")
    thread_source = meta.get("thread_source")
    explicit_role = meta.get("agent_role") or meta.get("agent_type")

    if thread_source in (None, "user"):
        if isinstance(explicit_role, str) and explicit_role:
            return f"primary:{explicit_role}"
        if source == "cli":
            return "primary:interactive"
        return f"primary:{source}"

    role = explicit_role if isinstance(explicit_role, str) else _find_role(thread_source)
    if role:
        return f"subagent:{role}"
    if isinstance(thread_source, str):
        normalized = thread_source.replace("sub_agent", "subagent").replace("-", "_")
        return normalized if normalized != "user" else "primary"
    return "subagent"


def _usage_delta(current: dict[str, Any], previous: dict[str, int]) -> dict[str, int]:
    """Fallback for old events that have totals but no last_token_usage."""
    delta: dict[str, int] = {}
    for field in TOKEN_FIELDS:
        raw = current.get(field, 0)
        value = int(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else 0
        old = previous.get(field, 0)
        delta[field] = value - old if value >= old else value
        previous[field] = value
    return delta


def iter_rollouts(sessions_dir: Path) -> Iterable[Path]:
    if not sessions_dir.exists():
        return []
    return sorted(sessions_dir.rglob("rollout-*.jsonl"))


def analyze(sessions_dir: Path, start: datetime, end: datetime) -> dict[str, Any]:
    totals = Usage()
    by_model: dict[str, Usage] = defaultdict(Usage)
    by_agent: dict[str, Usage] = defaultdict(Usage)
    by_model_agent: dict[tuple[str, str], Usage] = defaultdict(Usage)
    by_day: dict[str, Usage] = defaultdict(Usage)
    files_scanned = files_with_usage = malformed_lines = fallback_records = 0
    earliest: datetime | None = None
    latest: datetime | None = None

    for path in iter_rollouts(sessions_dir):
        files_scanned += 1
        meta: dict[str, Any] = {}
        agent_type = "primary:unknown"
        current_model = "unknown"
        previous_total: dict[str, int] = {}
        used_file = False

        try:
            stream = path.open(encoding="utf-8")
        except OSError:
            continue
        with stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    malformed_lines += 1
                    continue

                event_type = event.get("type")
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    payload = {}

                if event_type == "session_meta":
                    meta = payload
                    agent_type = classify_agent(meta)
                    current_model = str(meta.get("model") or current_model)
                    continue
                if event_type == "turn_context":
                    current_model = str(payload.get("model") or current_model)
                    continue
                if event_type != "event_msg" or payload.get("type") != "token_count":
                    continue

                timestamp = parse_timestamp(event.get("timestamp"))
                if timestamp is None:
                    continue
                info = payload.get("info")
                if not isinstance(info, dict):
                    continue
                total_usage = info.get("total_token_usage")
                if timestamp < start:
                    # Preserve the cumulative baseline for legacy events whose
                    # first in-window record lacks last_token_usage.
                    if isinstance(total_usage, dict):
                        for field in TOKEN_FIELDS:
                            raw = total_usage.get(field, 0)
                            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                                previous_total[field] = int(raw)
                    continue
                if timestamp >= end:
                    continue
                usage = info.get("last_token_usage")
                if not isinstance(usage, dict):
                    if not isinstance(total_usage, dict):
                        continue
                    usage = _usage_delta(total_usage, previous_total)
                    fallback_records += 1
                else:
                    if isinstance(total_usage, dict):
                        for field in TOKEN_FIELDS:
                            raw = total_usage.get(field, 0)
                            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                                previous_total[field] = int(raw)

                day = timestamp.date().isoformat()
                totals.add(usage)
                by_model[current_model].add(usage)
                by_agent[agent_type].add(usage)
                by_model_agent[(current_model, agent_type)].add(usage)
                by_day[day].add(usage)
                earliest = timestamp if earliest is None or timestamp < earliest else earliest
                latest = timestamp if latest is None or timestamp > latest else latest
                used_file = True

        if used_file:
            files_with_usage += 1

    def usage_map(values: dict[Any, Usage]) -> dict[str, dict[str, int]]:
        ordered = sorted(values.items(), key=lambda item: item[1].total_tokens, reverse=True)
        return {str(key): asdict(usage) for key, usage in ordered}

    cross = {
        f"{model} | {agent}": asdict(usage)
        for (model, agent), usage in sorted(
            by_model_agent.items(), key=lambda item: item[1].total_tokens, reverse=True
        )
    }
    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "observed": {
            "first_usage": earliest.isoformat() if earliest else None,
            "last_usage": latest.isoformat() if latest else None,
        },
        "sessions_dir": str(sessions_dir),
        "files_scanned": files_scanned,
        "files_with_usage": files_with_usage,
        "malformed_lines": malformed_lines,
        "fallback_records": fallback_records,
        "totals": asdict(totals),
        "by_model": usage_map(by_model),
        "by_agent_type": usage_map(by_agent),
        "by_model_and_agent_type": cross,
        "by_day": {day: asdict(by_day[day]) for day in sorted(by_day)},
    }


def _human(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _table(title: str, rows: dict[str, dict[str, int]]) -> str:
    headers = ("name", "calls", "input", "cache read", "cache write", "cache%", "output", "reasoning", "total")
    rendered: list[list[str]] = []
    for name, usage in rows.items():
        input_tokens = usage["input_tokens"]
        cached = usage["cached_input_tokens"]
        cache_pct = 100 * cached / input_tokens if input_tokens else 0
        rendered.append(
            [
                name,
                str(usage["calls"]),
                _human(input_tokens),
                _human(cached),
                _human(usage["cache_write_input_tokens"]),
                f"{cache_pct:.1f}%",
                _human(usage["output_tokens"]),
                _human(usage["reasoning_output_tokens"]),
                _human(usage["total_tokens"]),
            ]
        )
    if not rendered:
        return f"{title}\n  (no usage)"
    widths = [len(header) for header in headers]
    for row in rendered:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [title]
    lines.append("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    lines.append("  ".join("-" * width for width in widths))
    for row in rendered:
        lines.append(
            "  ".join(
                cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
                for i, cell in enumerate(row)
            )
        )
    return "\n".join(lines)


def format_text(report: dict[str, Any]) -> str:
    totals = report["totals"]
    window = report["window"]
    observed = report["observed"]
    lines = [
        "CODEX TOKEN XRAY",
        f"window: {window['start']} to {window['end']} (end exclusive)",
        f"observed usage: {observed['first_usage'] or 'none'} to {observed['last_usage'] or 'none'}",
        f"files: {report['files_with_usage']} with usage / {report['files_scanned']} scanned",
        f"model calls: {totals['calls']:,}",
        "",
        _table("BY MODEL", report["by_model"]),
        "",
        _table("BY AGENT TYPE", report["by_agent_type"]),
        "",
        _table("BY MODEL x AGENT TYPE", report["by_model_and_agent_type"]),
        "",
        _table("BY DAY (UTC)", report["by_day"]),
    ]
    if report["malformed_lines"] or report["fallback_records"]:
        lines.extend(
            [
                "",
                f"parser notes: malformed lines={report['malformed_lines']}, "
                f"cumulative-usage fallbacks={report['fallback_records']}",
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=30, help="rolling window in days (default: 30)")
    parser.add_argument(
        "--start", type=parse_boundary, help="inclusive local YYYY-MM-DD or ISO timestamp"
    )
    parser.add_argument(
        "--end", type=parse_boundary, help="exclusive local YYYY-MM-DD or ISO timestamp"
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path(os.path.expanduser("~/.codex/sessions")),
        help="Codex sessions directory (default: ~/.codex/sessions)",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    end = args.end or datetime.now(timezone.utc)
    start = args.start or end - timedelta(days=args.days)
    if start >= end:
        raise SystemExit("--start must be earlier than --end")
    report = analyze(args.sessions_dir.expanduser(), start, end)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
