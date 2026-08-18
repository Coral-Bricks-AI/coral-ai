---
id: compute_total_return
when: Compute trailing 1-year price return for a ticker.
applies_to: [equity_analyst]
---

**Dedicated tool: `compute_total_return`. Call AFTER `search_companies` returns the ticker.**

```
compute_total_return(
    ticker=<TICKER from search_companies>,
    as_of_iso=<optional YYYY-MM-DD>,
)
```

Price return only (no dividends). Returns `pct_return_1y`, the anchoring
prices/dates, and an `answer_summary_block` — quote that verbatim.

If the ticker is unknown or the corpus lacks a snapshot for the target
date, the tool returns an `error` envelope. Surface it to the user.
