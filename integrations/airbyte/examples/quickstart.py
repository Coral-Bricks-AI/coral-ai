"""Tiny demo: point at a directory of Airbyte JSONL and print records.

Run from this package root::

    pip install -e .
    python examples/quickstart.py

By default this reads the checked-in HackerNews-style fixtures so the
demo runs with zero external setup. Replace ``AIRBYTE_PATH`` below with
the directory your own Airbyte destination writes to (Local JSON default
root: ``/tmp/airbyte_local/<path>``).
"""

from __future__ import annotations

from pathlib import Path

from coralbricks.airbyte import read_airbyte_output

AIRBYTE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> None:
    records = read_airbyte_output(
        AIRBYTE_PATH,
        stream="stories",
        text_field="title",
        id_field=lambda d: f"hn-{d['id']}",
    )
    print(f"read {len(records)} records from {AIRBYTE_PATH}")
    for r in records[:5]:
        snippet = r["text"][:70] + ("…" if len(r["text"]) > 70 else "")
        print(f"  {r['id']:<12}  source={r['source']:<10}  {snippet!r}")


if __name__ == "__main__":
    main()
