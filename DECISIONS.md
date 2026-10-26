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

**Rule: every research return uses Adj Close.** For income-paying funds, the
gap is most of the return. Full period, annualized, first to last price:

| | Adj Close (total return) | Close (split-only) | Gap |
|---|---|---|---|
| TLT (2007-01 to 2026-09) | +2.47%/yr | -0.68%/yr | 3.15 pp/yr |
| HYG (2007-04 to 2026-09) | +4.76%/yr | -1.54%/yr | 6.29 pp/yr |

On Close, both bond funds *lost* money; on a total-return basis, both made
money. A backtest on Close would badly understate bonds and credit relative to
equities and gold (GLD pays nothing), which biases any cross-asset comparison.

How it is enforced (`tests/test_return_basis.py`):
- `analysis.panel`, the only input for research code, has no `close` column.
- A static (AST) scan fails if any module outside ingest/quality/report
  references `close`, or if code inside them computes a return from `close`
  under any name other than the labelled diagnostic `split_only_return`.
- A synthetic ex-dividend day checks that `daily_return` is 0% when Close falls
  by the dividend.
- The real panel is checked: `daily_return == adj_close.pct_change()`.
- Verified by mutation: switching `daily_return` to `close` fails two tests.

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

---

# Research methodology

Results are in the README, which is generated from `results/`. This section
explains *why* each choice was made; it avoids quoting results so it can't go
stale.

## Timing and look-ahead

**Signal at the close, trade one session later.** Signals use the close of the
last session of the month. The trade executes at the next session's close, so
the first return earned is two sessions after the signal. Trading at the same
close the signal was computed from assumes you can see the closing price before
the close, which you can't. At a monthly horizon, the extra day costs almost
nothing and removes the objection.

**Labels match execution.** The "next-month return" used for IC and ML
training is the return the backtest actually earns, from the trade close to the
next trade close, not a calendar-month return. Otherwise the IC would measure
something the strategy can't capture.

**How look-ahead is tested, not just argued.** The test replaces all data after
a cutoff with random noise and recomputes everything. If any feature, target
weight or daily P&L value at or before the cutoff changes, information leaked
from the future. A deliberately leaky feature (tomorrow's price) is included as
a positive control: the check must catch it, or the test proves nothing.

## The hold-out

**The last 3 years are untouched.** Every parameter is chosen by in-sample
Sharpe. The selection function receives backtests with all out-of-sample rows
cut off, so it *cannot* see the hold-out; a test makes the OOS returns
spectacular or terrible and confirms the choice doesn't move.

**Purging.** One monthly signal's return window straddles the boundary (it
starts in-sample and ends out-of-sample). It is dropped from both periods, so
neither period's IC uses information from the other.

**All grid points are reported, including out-of-sample.** Showing the full
grid proves the chosen parameters weren't cherry-picked, and reporting a number
doesn't influence the choice, which was made in-sample. A small, pre-declared
grid also limits how much overfitting is possible.

## Strategy construction

**Cross-sectional momentum: terciles, equal weight, dollar-neutral.** This
follows the spec. Long top third, short bottom third. It ranks *raw* returns
across very different assets, so high-volatility assets (oil, silver, emerging
markets) tend to sit in the extreme terciles. That is a known weakness. A
volatility-adjusted ranking is the natural fix, but adding it *after* seeing
the out-of-sample result would be data snooping, so it is listed as future work
rather than quietly swapped in.

**Trend: volatility-scaled.** Without scaling, a trend portfolio is mostly a
bet on its most volatile assets. Scaling each asset to the same target
volatility (using trailing 60-day vol) makes each asset contribute roughly
equal risk. Per-asset leverage is capped because SHY's tiny volatility would
otherwise demand extreme leverage. Gross leverage is capped too. These caps
were set once and not tuned.

**"Flat" vs "short" below the MA was a grid choice**, made on in-sample data
like every other parameter.

**Two benchmarks.** The spec says equal-weight buy-and-hold. A pure
buy-and-hold drifts toward whatever performed best, so a monthly-rebalanced
1/N is also shown, which is the fairer comparison for strategies that
rebalance monthly. Both are reported.

**Costs on traded notional.** Cost = rate × Σ|Δweight|, where Δweight is
measured against the *drifted* weights (what you actually hold before
rebalancing), not the previous targets. This matters: even an equal-weight
portfolio trades every month because prices move.

## Metrics

- **Annualized return** is geometric (CAGR). **Volatility and Sharpe** use
  daily returns × √252. Sharpe uses a 0% risk-free rate because there's no
  cash series in the data; this flatters long-only strategies a bit and is
  listed as a limitation.
- **Max drawdown is measured within each period**, so a drawdown that spans
  the IS/OOS boundary is split. The drawdown chart shows the continuous
  series.
- **IC** is Spearman rank correlation, because ranks ignore magnitudes and are
  robust to one huge return (e.g. USO in April 2020). **t-stat** =
  mean / (std / √n). Monthly forward windows don't overlap, so treating the
  ICs as roughly independent is reasonable.
- **"Significant" means |t| ≥ 2**, roughly the 5% level. The README verdicts
  are computed from this rule, not written after looking at the numbers.

## ML extension

- **Target: cross-sectional rank of next-month return**, the same task as the
  momentum signal (ordering assets), so the two ICs are directly comparable.
- **Inputs are cross-sectional ranks**, which are stable over time; raw
  momentum in 2008 and 2017 live on very different scales.
- **Walk-forward, expanding window, retrained every January.** Each fit may
  only use rows whose label had fully realized by the fit date (asserted in
  code). The training log in `results/ml_training_log.csv` shows each fit date
  next to the newest label it used.
- **Hyperparameters fixed in advance** (shallow trees, slow learning rate,
  large leaves, i.e. deliberately conservative) and never tuned. Tuning them
  would need its own nested validation, and on ~24 assets × ~15 years, the
  room for overfitting is large.
- **The comparison is paired**: ML and momentum IC on the *same* months, with a
  t-test on the difference. Comparing two means from different months would be
  misleading.

## Engineering choices

- **The README is generated** (`analysis/findings.py`), and a test fails if the
  committed README differs from what the script produces. No number can be
  typed by hand or go stale.
- **The dashboard reads only small pre-aggregated files** in `results/`, so
  it deploys on a free tier without the data pipeline, DuckDB or scikit-learn.
- **Bugs this project's own tests found** (useful interview stories):
  - Default arguments like `end=config.END_DATE` are evaluated once, when
    Python imports the module. Changing the config later (as tests do) was
    silently ignored. Reconciliation still passed, because the counts were
    internally consistent while the grid covered the wrong window. Fixed by
    resolving defaults inside the function.
  - The first trade's turnover was dated the day before its cost, so turnover
    metrics dropped it. The hand-checked toy backtest caught it.
