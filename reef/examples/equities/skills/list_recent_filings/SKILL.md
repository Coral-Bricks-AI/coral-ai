---
id: list_recent_filings
when: List a company's most recent SEC filings (10-K, 10-Q, 8-K). Use for questions about "recent filings", "latest 10-Q", "what did they report last quarter", or any event that would appear in a filing headline.
applies_to: [equity_analyst]
---

**Dedicated tool: `list_recent_filings`. Call AFTER `search_companies` returns the ticker.**

```
list_recent_filings(
    ticker=<TICKER from search_companies>,
    as_of_iso=<optional YYYY-MM-DD>,
    k=<optional int, default 5>,
)
```

Returns up to `k` filings, most recent first. Each row is
`{"filing_date", "form_type", "headline"}`.

Quote the returned `answer_summary_block` verbatim when replying.
