"""Tests for coralbricks.context_prep.cleaners (trafilatura)."""

from __future__ import annotations

import pytest

from coralbricks.context_prep.cleaners import clean_documents, clean_html

trafilatura = pytest.importorskip("trafilatura")


SAMPLE_HTML = """
<!doctype html>
<html>
  <head><title>The Great Article</title></head>
  <body>
    <nav>nav junk</nav>
    <main>
      <article>
        <h1>The Great Article</h1>
        <p>Coral Bricks is a memory layer for agentic AI. It is fast and friendly.</p>
        <p>This second paragraph also contains useful body content for trafilatura to extract.</p>
      </article>
    </main>
    <footer>copyright 2026</footer>
  </body>
</html>
"""


def test_clean_html_extracts_main_text() -> None:
    result = clean_html(SAMPLE_HTML)
    assert "Coral Bricks" in result["text"]
    assert "nav junk" not in result["text"]
    assert result["title"] == "The Great Article"


def test_clean_documents_drops_empty() -> None:
    out = clean_documents(["", "<html></html>"], drop_empty=True)
    assert out == []


def test_clean_documents_returns_records() -> None:
    out = clean_documents([SAMPLE_HTML, SAMPLE_HTML])
    assert len(out) == 2
    assert all("Coral Bricks" in d["text"] for d in out)
    assert all(d["metadata"]["cleaned_by"] == "trafilatura" for d in out)
    assert out[0]["metadata"].get("title") == "The Great Article"
