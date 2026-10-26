# Cross-Asset Momentum and Trend Signals: Research and Backtest

A research pipeline over 24 liquid ETFs (equity sectors, broad and international
equity, Treasuries, credit, TIPS, REITs, commodities) plus Treasury yields from
FRED, from 2007 to 2026-09.

**Status:** data ingestion, data quality, and the curated panel are done.
Features, signals, and backtest come next.

## Run it

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python -m xasset.pipeline        # first run downloads (~30s); reruns use the cache
python -m xasset.pipeline --refresh   # force a re-download
pytest                           # injected-defect tests for every DQ rule
```

No API keys are needed. `.env` is git-ignored anyway.

## Pipeline

```
yfinance / FRED --(retry + backoff)--> data/raw/*.parquet + manifest.json   (cache, immutable)
        --> DuckDB raw.prices, raw.macro                                     (as received)
        --> clean.prices      ticker x NYSE-session grid, 1 boolean per DQ rule
            clean.macro_daily macro aligned to sessions (lag 1, ffill <= 5)
        --> analysis.panel    VIEW: clean minus HARD_RULES, joined to macro
        --> data/curated/panel.parquet
```

Row counts are reconciled at every arrow, and the run fails on any mismatch.

## Outputs

| Path | What |
|---|---|
| `reports/data_quality.md` | Counts per rule per ticker, reconciliation, every extreme return, stale runs, calendar alignment, macro checks |
| `data/warehouse.duckdb` | Layers `raw`, `clean`, `analysis`, `meta` |
| `data/curated/panel.parquet` | `date, ticker, asset_class, adj_close, daily_return, volume, dgs10, dgs2, t10y2y, macro_ffilled, dq_soft_flags` |
| `docs/decisions.md` | Why each choice was made |

## Layout

```
xasset/config.py     every threshold and rule list
xasset/ingest.py     cache, retries, manifest
xasset/calendar.py   NYSE sessions (exchange_calendars)
xasset/quality.py    DQ rules -> flag columns
xasset/curate.py     macro alignment and checks
xasset/warehouse.py  DuckDB layers and analysis view
xasset/report.py     markdown DQ report
xasset/pipeline.py   orchestration and reconciliation
tests/               synthetic-defect tests
```
