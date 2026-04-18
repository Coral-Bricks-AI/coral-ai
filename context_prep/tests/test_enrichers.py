"""Tests for coralbricks.context_prep.enrichers."""

from __future__ import annotations

from coralbricks.context_prep.enrichers import (
    REGISTRY,
    DateExtractor,
    EmailExtractor,
    MoneyExtractor,
    TickerExtractor,
    URLExtractor,
    enrich_documents,
)


def test_url_extractor() -> None:
    text = "see https://coralbricks.com and http://example.org/path?x=1 too"
    hits = URLExtractor().extract(text)
    assert {h.value for h in hits} == {
        "https://coralbricks.com",
        "http://example.org/path?x=1",
    }
    for h in hits:
        assert h.label == "URL"
        assert text[h.start : h.end] == h.value


def test_email_extractor() -> None:
    hits = EmailExtractor().extract("contact us at founders@coralbricks.com please")
    assert [h.value for h in hits] == ["founders@coralbricks.com"]


def test_date_extractor() -> None:
    text = "Filed on 2026-04-18, updated 04/19/2026 and again on April 20, 2026"
    hits = DateExtractor().extract(text)
    values = {h.value.lower() for h in hits}
    assert "2026-04-18" in values
    assert "04/19/2026" in values
    assert any("april 20" in v for v in values)


def test_money_extractor() -> None:
    text = "Revenue of $1.2 billion and EUR 500m, plus USD 10,000"
    hits = MoneyExtractor().extract(text)
    matched = {h.value for h in hits}
    assert any("$1.2 billion" in m for m in matched)
    assert any("EUR 500m" in m for m in matched)
    assert any("USD 10,000" in m for m in matched)


def test_ticker_extractor_cashtags_only_by_default() -> None:
    hits = TickerExtractor().extract("$AAPL and AAPL and $MSFT")
    assert {h.value for h in hits} == {"AAPL", "MSFT"}


def test_ticker_extractor_with_vocab() -> None:
    hits = TickerExtractor(vocab={"AAPL", "TSLA"}).extract("AAPL TSLA NVDA $AAPL")
    cashtags = [h for h in hits if h.metadata.get("cashtag")]
    bare = [h for h in hits if not h.metadata.get("cashtag")]
    assert {h.value for h in cashtags} == {"AAPL"}
    assert {h.value for h in bare} == {"AAPL", "TSLA"}


def test_registry_contains_expected_keys() -> None:
    assert {"tickers", "urls", "emails", "dates", "money"} <= set(REGISTRY)


def test_enrich_documents_populates_metadata() -> None:
    docs = enrich_documents(
        [{"id": "x", "text": "$AAPL on https://example.com 2026-04-18"}],
        ["tickers", "urls", "dates"],
    )
    assert len(docs) == 1
    extractions = docs[0]["metadata"]["extractions"]
    assert extractions["tickers"][0]["value"] == "AAPL"
    assert extractions["urls"][0]["value"] == "https://example.com"
    assert extractions["dates"][0]["value"] == "2026-04-18"
