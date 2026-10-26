# Design Decisions (and how to defend them)

Each entry covers what was decided, why, and what was traded away. The numbers
come from `reports/data_quality.md`.

## Ingestion

**One cached parquet file per ticker, written atomically, plus a manifest.**
The cache is the boundary between "talking to an API" and "doing research".
After the first run, every rerun is offline, fast, and deterministic. Writing to
`*.tmp` and then renaming means a crash can never leave a half-written file that
looks complete. Resumability comes for free: a rerun downloads only what is
missing.

**Retries with exponential backoff and jitter.** Free APIs rate-limit and fail
intermittently. Waiting longer after each failure (2s, 4s, 8s...) gives the
service time to recover. Jitter (a small random addition) keeps retries from
arriving all at once. yfinance often returns an *empty frame* instead of raising
an error, so empty results are treated as failures and retried as well.

**The raw cache is never modified.** Cleaning happens downstream, so a rule can
change without re-downloading, and anyone can audit exactly what the vendor
sent.

**A pinned end date (2026-09-30).** "Download up to today" makes results change
every day. A fixed window plus the cache makes the study reproducible.

**The manifest records `downloaded_at`.** Yahoo recomputes `Adj Close`
backwards every time a new dividend is paid, so the same date can have a
different adjusted price in next month's download. Returns (ratios) are
unaffected; price *levels* are not. Recording the download time keeps that
behavior visible instead of surprising someone later.

## Adjusted prices: what yfinance actually returns

- `Adj Close`: adjusted for splits **and** dividends. All returns use this, so
  a return is a total return, the P&L a holder actually earned.
- `Close`: adjusted for splits but **not** dividends (this surprises many
  people). The report compares returns from the two series. A large gap marks a
  big distribution (for example, DBC on 2024-12-23: Close -5.0%, Adj Close
  +0.1%). USO's 1:8 reverse split in April 2020 leaves no trace in either
  series, which confirms the split adjustment worked.

## Universe

HYG starts 2007-04-11; every other ticker has full 2007 coverage. Rule: keep a
ticker if its first price is on or before 2007-12-31, and tag a start after
2007-01-01 as `included_late_start`. The reasoning: a 12-month momentum signal
can't exist before 2008 anyway, so HYG costs about three months of signal
history. Dropping it would remove half the credit sleeve. The rule is in
`config.py`, so a stricter cutoff is a one-line change.

## Data quality: flag, don't delete

- **One boolean column per rule.** Each rule can then be counted, audited, and
  switched off independently. A single `is_bad` column would hide which rule
  fired.
- **Hard vs. soft rules** (`config.HARD_RULES`). Hard rules exclude a row from
  the analysis view: duplicates (copies after the first), off-calendar rows,
  NULL prices, non-positive prices. These rows would give *wrong numbers*. Soft
  rules keep the row and record the rule in `dq_soft_flags`. They mark things
  that are *suspicious but possibly real*.
- **Extreme returns are soft.** 191 of the 247 flags fall in the 2008–09 and
  2020 crises. The 56 outside those windows were each checked against the
  calendar, and each matches a known event: the Aug 2011 US downgrade, the Apr
  2013 gold crash, OPEC in Nov 2014, Feb 2018 "Volmageddon", the 2022 rate
  shocks (SHY), SVB (Mar 2023), the Nov 2024 election, the Apr 2025 tariff
  crash and rebound, and the Jan 2026 metals crash (SLV -28.5%). The
  diagnostic: flags *cluster across many tickers on the same date*, which is
  what a market event looks like. A single-ticker spike with no peers would
  look like a data error. Deleting extreme returns removes exactly the days a
  risk model most needs and makes a strategy look safer than it is.
- **Extreme means |r| > 15% OR more than 8 robust sigmas.** A fixed threshold
  alone misses a 2% move in a short-Treasury ETF, and a volatility-scaled rule
  alone over-flags in calm periods. The robust sigma uses the median absolute
  deviation over the prior 252 days, so the crash itself does not inflate the
  yardstick. It uses past data only, which a test enforces.
- **Stale prices are soft.** The three SHY runs (Dec 2012, Apr 2013, Aug 2021)
  have normal volume. A 1–3 year Treasury fund at zero rates often doesn't move
  by a cent for a week. A dead feed would usually show zero volume, which is a
  separate rule.
- **Missing sessions become explicit rows.** A missing day has no row to flag,
  so the clean table is a full ticker × NYSE-session grid. A missing day is a
  row with NULL price and `flag_missing_session = True`.
- **The NYSE calendar comes from `exchange_calendars`.** "Weekdays minus
  holidays" misses unscheduled closures (Hurricane Sandy, the presidential
  funerals).
- **A gap-spanning return keeps its real value.** After a missing day, the next
  return is the true two-day return, flagged `flag_return_spans_gap`. It is not
  a fake 0% followed by a catch-up move.

The Yahoo data turned out calendar-clean: 0 missing sessions, 0 duplicates.
That does **not** show the checks work, so `tests/test_quality.py` injects each
defect into synthetic data and asserts that the right rule (and only that rule)
fires.

## Reconciliation

Row counts are checked at every handoff: manifest → raw tables → clean
(raw + inserted rows) → analysis view (clean − hard exclusions) → parquet
export. They are also checked per ticker. The exclusion logic exists twice, in
pandas and in the SQL view filter, and the pipeline asserts both give the same
count, so the two can't silently drift apart. Any mismatch fails the run.

## Storage layers (DuckDB)

`raw` (as received) → `clean` (grid + flags, nothing deleted) → `analysis.panel`
(a **view**, not a copy). Because the analysis layer is a view, moving a rule
between hard and soft changes it on the next run with no data rewrite. DuckDB
was chosen because it is a single embedded file with fast columnar SQL over
dataframes and needs no server.

## Forward-fill policy

- **Prices: never filled.** A filled price creates a fake 0% return and makes
  a gap invisible. It also biases volatility estimates downward.
- **Macro: filled, but lagged and capped.**
  - *Lag 1 session*: FRED's DGS10 comes from the Fed's H.15 release, published
    around 4:15pm ET the next business day, after the equity close. Using the
    value dated *t* at the close of *t* would be look-ahead. (A trader can see
    yields in real time, so one could argue for no lag. The conservative choice
    costs almost nothing for a slow-moving signal.)
  - *Fill*: the bond market closes on some days the NYSE is open (Columbus Day,
    Veterans Day). On those days the last published yield *is* what a trader
    knew. 36 sessions get filled.
  - *Cap at 5 sessions*: if the feed died, values become NULL instead of
    freezing a yield for months.
- **The spread check** (T10Y2Y vs DGS10 − DGS2) matches exactly, because FRED
  builds T10Y2Y from the same inputs. It is a consistency check, not
  independent verification, and should be described that way.

## Known limitations

- **Survivorship and selection bias.** The universe is today's well-known,
  liquid ETFs, chosen with hindsight. Results say nothing about funds that
  closed.
- **ETFs are not futures.** USO and DBC hold rolling futures and carry roll
  costs (contango) that a direct futures position would manage differently.
  GLD and SLV hold physical metal.
- **One vendor.** yfinance is unofficial and unaudited. A production pipeline
  would cross-check a second source (e.g. compare closes with another vendor
  and flag disagreements).
- **Daily close only.** The data can't show whether the close was tradable or
  what the bid-ask spread was.
