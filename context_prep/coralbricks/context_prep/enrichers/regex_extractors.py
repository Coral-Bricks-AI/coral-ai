"""Regex-based extractors. Zero deps, deterministic, fast."""

from __future__ import annotations

import re
from typing import Iterable

from .base import BaseExtractor, ExtractionResult


def _matches(pattern: re.Pattern, text: str, label: str | None = None) -> list[ExtractionResult]:
    return [
        ExtractionResult(value=m.group(0), start=m.start(), end=m.end(), label=label)
        for m in pattern.finditer(text)
    ]


# Note on tickers: dollar-prefixed cashtags ($AAPL) are unambiguous. Bare
# upper-case tokens like "AAPL" are too noisy without a vocabulary, so we
# require either a $ prefix or an explicit ticker_vocab.
_CASHTAG_RE = re.compile(r"\$[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b")
_URL_RE = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?"
    r"(?:\(\d{2,4}\)|\d{2,4})"
    r"[\s.-]?\d{3,4}[\s.-]?\d{3,4}"
)
_HASHTAG_RE = re.compile(r"(?<![A-Za-z0-9_])#[A-Za-z0-9_]{2,}")
_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]{2,}")
_MONEY_RE = re.compile(
    r"(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY|INR)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|trillion|m|bn|tn|k))?"
    r"|[$€£¥]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|trillion|m|bn|tn|k))?",
    re.IGNORECASE,
)

# Date patterns: ISO-8601-ish, US numeric, and "Month Day, Year".
_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_RE = re.compile(
    rf"\b(?:\d{{4}}-\d{{2}}-\d{{2}}"
    rf"|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}"
    rf"|\d{{1,2}}-\d{{1,2}}-\d{{2,4}}"
    rf"|(?:{_MONTH_NAMES})\s+\d{{1,2}},?\s+\d{{4}}"
    rf"|\d{{1,2}}\s+(?:{_MONTH_NAMES})\s+\d{{4}})\b",
    re.IGNORECASE,
)


class URLExtractor(BaseExtractor):
    name = "urls"

    def extract(self, text: str) -> list[ExtractionResult]:
        return _matches(_URL_RE, text, label="URL")


class EmailExtractor(BaseExtractor):
    name = "emails"

    def extract(self, text: str) -> list[ExtractionResult]:
        return _matches(_EMAIL_RE, text, label="EMAIL")


class PhoneExtractor(BaseExtractor):
    name = "phones"

    def extract(self, text: str) -> list[ExtractionResult]:
        return _matches(_PHONE_RE, text, label="PHONE")


class HashtagExtractor(BaseExtractor):
    name = "hashtags"

    def extract(self, text: str) -> list[ExtractionResult]:
        return _matches(_HASHTAG_RE, text, label="HASHTAG")


class MentionExtractor(BaseExtractor):
    name = "mentions"

    def extract(self, text: str) -> list[ExtractionResult]:
        return _matches(_MENTION_RE, text, label="MENTION")


class MoneyExtractor(BaseExtractor):
    name = "money"

    def extract(self, text: str) -> list[ExtractionResult]:
        return _matches(_MONEY_RE, text, label="MONEY")


class DateExtractor(BaseExtractor):
    name = "dates"

    def extract(self, text: str) -> list[ExtractionResult]:
        return _matches(_DATE_RE, text, label="DATE")


class TickerExtractor(BaseExtractor):
    """Extract stock tickers.

    Always finds ``$AAPL``-style cashtags. If a ``vocab`` is provided
    (set of valid uppercase symbols), bare uppercase tokens that match
    are also returned.
    """

    name = "tickers"

    def __init__(self, vocab: Iterable[str] | None = None):
        self._vocab = {v.upper() for v in vocab} if vocab else None
        self._bare_re = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b") if self._vocab else None

    def extract(self, text: str) -> list[ExtractionResult]:
        results = [
            ExtractionResult(
                value=m.group(0)[1:],
                start=m.start(),
                end=m.end(),
                label="TICKER",
                metadata={"cashtag": True},
            )
            for m in _CASHTAG_RE.finditer(text)
        ]
        if self._bare_re and self._vocab:
            for m in self._bare_re.finditer(text):
                if m.group(0) in self._vocab:
                    results.append(
                        ExtractionResult(
                            value=m.group(0),
                            start=m.start(),
                            end=m.end(),
                            label="TICKER",
                            metadata={"cashtag": False},
                        )
                    )
        results.sort(key=lambda r: r.start)
        return results
