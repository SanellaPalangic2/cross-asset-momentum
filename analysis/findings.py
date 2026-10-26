"""Generate README.md from results/. Every number in the README comes from here.

    python analysis/findings.py           # write README.md
    python analysis/findings.py --check   # exit 1 if README.md is stale

Don't edit README.md by hand. Change this script or the research, then rerun.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from xasset import config

RES = config.RESULTS_DIR
README = config.ROOT / "README.md"
NAMES = {"xs_mom": "XS momentum (L/S terciles)", "ts_trend": "TS trend (vol-scaled)",
         "ew_rebal": "Equal weight, monthly rebalanced", "ew_hold": "Equal weight, buy and hold",
         "ml_xs": "ML rank L/S (walk-forward)"}


T_SIGNIFICANT = 2.0   # |t| >= 2, roughly 5% two-sided
P_SIGNIFICANT = 0.05


def pct(x, d=1):
    return "n/a" if pd.isna(x) else f"{x:.{d}%}"


def num(x, d=2):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def load():
    r = {
        "summary": json.loads((RES / "summary.json").read_text()),
        "dq": json.loads((RES / "dq_summary.json").read_text()),
        "perf": pd.read_csv(RES / "performance.csv"),
        "grid": pd.read_csv(RES / "param_grid.csv"),
        "ic": pd.read_csv(RES / "ic_summary.csv"),
        "ml": pd.read_csv(RES / "ml_vs_momentum.csv"),
        "cost": pd.read_csv(RES / "cost_sensitivity.csv"),
        "sub": pd.read_csv(RES / "subperiods.csv"),
        "regime": pd.read_csv(RES / "regime.csv"),
        "corr": pd.read_csv(RES / "strategy_correlation.csv", index_col=0),
        "train": pd.read_csv(RES / "ml_training_log.csv"),
    }
    return r


def perf_lookup(perf):
    return {(row.strategy, row.period): row for row in perf.itertuples()}


def performance_table(perf):
    rows = []
    for k in NAMES:
        for period in ("in-sample", "out-of-sample"):
            p = perf[(perf["strategy"] == k) & (perf["period"] == period)].iloc[0]
            rows.append({"Strategy": NAMES[k], "Period": f"{period} ({p['start'][:7]} to {p['end'][:7]})",
                         "Ann. return": pct(p["ann_return"]), "Ann. vol": pct(p["ann_vol"]),
                         "Sharpe": num(p["sharpe"]), "Max DD": pct(p["max_drawdown"], 0),
                         "Turnover/yr": num(p["turnover_per_year"], 1), "Hit rate (months)": pct(p["hit_rate_monthly"], 0)})
    return md(pd.DataFrame(rows))


def ml_table(ml):
    return md(pd.DataFrame({
        "Period": ml["period"], "Months": ml["months"],
        "ML mean IC": ml["ml_ic_mean"].map(lambda x: num(x, 3)),
        "12-1 momentum mean IC": ml["mom_ic_mean"].map(lambda x: num(x, 3)),
        "Difference": ml["diff_mean"].map(lambda x: num(x, 3)),
        "Paired t": ml["diff_t"].map(num), "p-value": ml["diff_p"].map(num),
    }))


def ic_table(ic):
    rows = []
    for sig in ic["signal"].unique():
        row = {"Signal": sig}
        for period in ("in-sample", "out-of-sample"):
            x = ic[(ic["signal"] == sig) & (ic["period"] == period)].iloc[0]
            tag = "IS" if period == "in-sample" else "OOS"
            row[f"{tag} mean IC"] = num(x["ic_mean"], 3)
            row[f"{tag} std"] = num(x["ic_std"], 3)
            row[f"{tag} t-stat"] = num(x["ic_t"])
            row[f"{tag} months"] = int(x["n_months"])
        rows.append(row)
    return md(pd.DataFrame(rows))


def grid_table(grid, family):
    g = grid[grid["family"] == family]
    return md(pd.DataFrame({
        "Parameters": g["params"],
        "Selected (IS only)": g["selected"].map({True: "**yes**", False: ""}),
        "Spec default": g["spec_default"].map({True: "yes", False: ""}),
        "IS Sharpe": g["is_sharpe"].map(num), "IS ann. return": g["is_ann_return"].map(pct),
        "OOS Sharpe": g["oos_sharpe"].map(num), "OOS ann. return": g["oos_ann_return"].map(pct),
    }))


def cost_table(cost):
    rows = []
    for k in ("xs_mom", "ts_trend", "ew_rebal"):
        for period in ("in-sample", "out-of-sample"):
            d = cost[(cost["strategy"] == k) & (cost["period"] == period)].set_index("cost_bps")
            row = {"Strategy": NAMES[k], "Period": period}
            for bps, v in d["sharpe"].items():
                row[f"Sharpe @ {bps:g} bp"] = num(v)
            row["Turnover/yr"] = num(d["turnover_per_year"].iloc[0], 1)
            rows.append(row)
    return md(pd.DataFrame(rows))


def sub_table(sub):
    rows = []
    for name in sub["subperiod"].unique():
        d = sub[sub["subperiod"] == name].set_index("strategy")
        row = {"Window": f"{name} ({d['start'].iloc[0]} to {d['end'].iloc[0]})"}
        for k in ("xs_mom", "ts_trend", "ew_rebal"):
            row[NAMES[k]] = f"{pct(d.loc[k, 'total_return'])} (DD {pct(d.loc[k, 'max_drawdown'], 0)})"
        rows.append(row)
    return md(pd.DataFrame(rows))


def regime_table(reg):
    return md(pd.DataFrame({
        "Strategy": reg["strategy"].map(NAMES), "Curve": reg["regime"],
        "Months": reg["months"], "of which OOS": reg["oos_months"],
        "Mean monthly return": reg["mean_monthly"].map(lambda x: pct(x, 2)),
        "Ann. Sharpe": reg["ann_sharpe"].map(num), "Hit rate": reg["hit_rate"].map(lambda x: pct(x, 0)),
    }))


def dq_section(dq):
    rules = pd.DataFrame([{"Rule": k.removeprefix("flag_"), "Type": "hard (excluded)" if k in dq["hard_rules"]
                           else "soft (kept, flagged)", "Rows flagged": v} for k, v in dq["rule_totals"].items()])
    late = [u for u in dq["universe"] if u["status"] != "included"]
    late_txt = "; ".join(f"{u['ticker']} starts {u['first_date'][:10]} ({u['status']})" for u in late) or "none"
    return (f"- {dq['reconciliation_passed']} of {dq['reconciliation_checks']} row-count reconciliation checks pass "
            f"(cache → raw → clean → analysis view → parquet, also per ticker).\n"
            f"- Clean rows: {dq['clean_rows']:,}; analysis-panel rows: {dq['panel_rows']:,}.\n"
            f"- Universe exceptions: {late_txt}.\n\n" + md(rules) +
            "\n\nFull detail, including every flagged extreme return: [`reports/data_quality.md`](reports/data_quality.md).")


def verdicts(r) -> dict:
    P = perf_lookup(r["perf"])
    ic = r["ic"].set_index(["signal", "period"])
    ml = r["ml"].set_index("period")
    xs_o, ts_o, ew_o = P[("xs_mom", "out-of-sample")], P[("ts_trend", "out-of-sample")], P[("ew_rebal", "out-of-sample")]
    xs_i, ts_i, ew_i = P[("xs_mom", "in-sample")], P[("ts_trend", "in-sample")], P[("ew_rebal", "in-sample")]
    mom_oos = ic.loc[("mom_12_1", "out-of-sample")]
    v = {}
    significant = mom_oos.ic_t >= T_SIGNIFICANT
    v["xs"] = (f"Cross-sectional momentum **{'showed' if significant else 'showed no'} statistically significant "
               f"predictive power out of sample**: 12-1 momentum IC {num(mom_oos.ic_mean, 3)} with t-stat "
               f"{num(mom_oos.ic_t)} (threshold {T_SIGNIFICANT:g}). The strategy's OOS Sharpe was {num(xs_o.sharpe)} "
               f"(in-sample {num(xs_i.sharpe)}), and in-sample it suffered a {pct(xs_i.max_drawdown, 0)} drawdown.")
    beat = ts_o.sharpe > ew_o.sharpe
    v["ts"] = (f"Time-series trend had an OOS Sharpe of {num(ts_o.sharpe)} (in-sample {num(ts_i.sharpe)}), "
               f"{'above' if beat else 'below'} the equal-weight benchmark's {num(ew_o.sharpe)} out of sample "
               f"(in-sample: {num(ts_i.sharpe)} vs {num(ew_i.sharpe)}). In-sample max drawdown was "
               f"{pct(ts_i.max_drawdown, 0)} vs the benchmark's {pct(ew_i.max_drawdown, 0)}"
               + (", so its clearest benefit is drawdown control." if ts_i.max_drawdown > ew_i.max_drawdown / 2 else "."))
    o, i = ml.loc["out-of-sample"], ml.loc["in-sample"]
    sig = (o.diff_p < P_SIGNIFICANT) and (o.diff_mean > 0)
    v["ml"] = (f"The ML model **{'beat' if sig else 'did not significantly beat'}** simple 12-1 momentum. On identical "
               f"months, its mean IC was {num(o.ml_ic_mean, 3)} vs {num(o.mom_ic_mean, 3)} out of sample "
               f"(paired difference t = {num(o.diff_t)}, p = {num(o.diff_p)}, {int(o.months)} months) and "
               f"{num(i.ml_ic_mean, 3)} vs {num(i.mom_ic_mean, 3)} before the hold-out (t = {num(i.diff_t)}, "
               f"p = {num(i.diff_p)}, {int(i.months)} months)." +
               ("" if sig else " A higher point estimate on a few dozen noisy monthly ICs is not evidence of a better signal."))
    return v


def render(r) -> str:
    s, dq = r["summary"], r["dq"]
    v = verdicts(r)
    P = perf_lookup(r["perf"])
    sel = s["selected"]
    corr = r["corr"]
    reg = r["regime"]
    inv_months = int(reg.loc[(reg["strategy"] == "ts_trend") & (reg["regime"] == "inverted"), "months"].iloc[0])
    ts_eq_corr = corr.loc["TS trend", "trend: equity"]
    n_eq = sum(1 for c in config.UNIVERSE.values() if c.startswith("equity"))
    cost_note = r["cost"].set_index(["strategy", "period", "cost_bps"])["sharpe"]
    hi = max(config.COST_SENSITIVITY_BPS)
    oos_n_months = P[("ts_trend", "out-of-sample")].months

    return f"""<!-- GENERATED by analysis/findings.py from results/. Do not edit by hand. -->
# Cross-Asset Momentum and Trend Signals: Research and Backtest

**Question.** Do simple momentum and trend signals, applied across {s['n_assets']} liquid ETFs spanning
equities, Treasuries, credit, TIPS, REITs and commodities, predict returns well enough to survive
realistic costs, and do they hold up on {s['holdout_years']} years of data that was never used to make any choice?

## Key results

The last {s['holdout_years']} years ({s['oos_start']} to {s['data_end']}) are a true hold-out. All parameters were chosen on
earlier data only. Costs are {s['cost_bps']:g} bp per side. Positions trade at the close one session after the signal.

- {v['xs']}
- {v['ts']}
- {v['ml']}

![Cumulative returns](results/figures/cumulative_returns.png)

## Data

| Source | Series | Notes |
|---|---|---|
| yfinance | {s['n_assets']} ETFs, daily `Adj Close` and volume | `Adj Close` is split- **and** dividend-adjusted; every return uses it (enforced by `tests/test_return_basis.py`) |
| FRED (pandas-datareader, no key) | DGS10, DGS2, T10Y2Y | lagged {config.MACRO_PUBLICATION_LAG_SESSIONS} session for publication; forward-filled at most {config.MACRO_FFILL_LIMIT_SESSIONS} sessions |

Window: {s['data_start']} to {s['data_end']}. First signal {s['first_signal']} (after the longest look-back warms up).

### Pipeline

```
yfinance / FRED --(retry + backoff)--> data/raw/*.parquet + manifest   (immutable cache, resumable)
   --> DuckDB raw.*            as received
   --> DuckDB clean.*          ticker x NYSE-session grid, one boolean column per DQ rule
   --> analysis.panel (VIEW)   clean minus hard-rule rows, joined to lagged macro
   --> features --> signals --> backtest / IC / ML --> results/ --> README, charts, dashboard
```

### Data quality

Rows are flagged, never deleted. Hard rules exclude a row from the panel; soft rules keep it and tag it.
Prices are never forward-filled.

{dq_section(dq)}

## Signals (all known at the close of the signal date)

| Feature | Definition |
|---|---|
| Momentum {config.MOM_LOOKBACK_MONTHS}-{config.MOM_SKIP_MONTHS} | return from t-{config.MOM_LOOKBACK_MONTHS}m to t-{config.MOM_SKIP_MONTHS}m ({config.TRADING_DAYS_PER_MONTH} sessions = 1 month), skipping the latest month |
| Short-term reversal | minus the trailing {config.REVERSAL_MONTHS}-month return |
| Trend | price / {config.TREND_MA_DAYS}-day moving average - 1 |
| Realized vol | {config.VOL_WINDOW_DAYS}-day std of daily returns, annualized; used to scale trend positions |
| Curve regime | inverted if 10Y-2Y < 0 (publication-lagged) |

**Timing.** Signals are computed at the close of the last session of each month. Trades execute at the close
of the next session, so a new position first earns the return {config.EXECUTION_LAG_SESSIONS + 1} sessions after the signal. The
no-lookahead tests (`tests/test_features.py`) replace all data after a cutoff with noise and assert that every
feature, every target weight and every daily P&L value at or before the cutoff is unchanged. A deliberately
leaky feature is included to prove the check can fail.

**Strategies.**
(a) *Cross-sectional momentum*: long the top third and short the bottom third by momentum, equal weight
(the long and short books are each equal to capital). Selected look-back: {sel['xs_mom']['lookback_months']}-{sel['xs_mom']['skip_months']} months.
(b) *Time-series trend*: long above the moving average, {sel['ts_trend']['below']} below, each asset scaled to
{pct(config.VOL_TARGET_PER_ASSET, 0)} vol / N (per-asset leverage cap {config.MAX_ASSET_LEVERAGE:g}x, gross cap {config.MAX_GROSS_LEVERAGE:g}x). Selected MA: {sel['ts_trend']['ma_days']} days.
(c) *Benchmarks*: equal weight, monthly rebalanced, and equal-weight buy-and-hold.
(d) *ML*: gradient boosting, long/short terciles of predicted rank.

## Results: in-sample vs out-of-sample (net of {s['cost_bps']:g} bp)

{performance_table(r['perf'])}

Sharpe ratios use a 0% risk-free rate (see Limitations). Max drawdown is measured within each period, so a
drawdown that spans the boundary is split; the chart below shows the continuous series. The ML strategy starts in {s['ml_first_test_year']}, after its first
walk-forward training window.

![Drawdowns](results/figures/drawdowns.png)

### Information coefficient (Spearman, signal vs next-month return)

The t-stat is mean / (std / sqrt(months)). The one signal month whose return window straddles the hold-out
boundary is excluded from both periods.

{ic_table(r['ic'])}

![Rolling IC](results/figures/rolling_ic.png)

### ML extension

`HistGradientBoostingRegressor` predicts each asset's next-month cross-sectional return rank from ranked
momentum, 1-month return, trend, volatility and the curve flag. It is retrained every January on an expanding
window, using only rows whose label had fully realized by the fit date. Hyperparameters were fixed in advance
and never tuned. Retraining continues through the hold-out, as it would in live trading; no choice was made
after seeing hold-out results.

{v['ml']}

{ml_table(r['ml'])}

## Robustness

### Parameter sensitivity (all grid points shown; selection used in-sample Sharpe only)

Cross-sectional momentum:

{grid_table(r['grid'], 'xs_mom')}

Time-series trend:

{grid_table(r['grid'], 'ts_trend')}

### Transaction costs

{cost_table(r['cost'])}

At {hi:g} bp, XS momentum's in-sample Sharpe is {num(cost_note[('xs_mom', 'in-sample', hi)])}; its turnover makes it the
most cost-sensitive strategy.

![Cost sensitivity](results/figures/cost_sensitivity.png)

### Stress windows (total return, max drawdown; all inside the in-sample period)

{sub_table(r['sub'])}

### Yield curve inverted vs not (regime known at the start of each month)

{regime_table(r['regime'])}

Only {inv_months} months had an inverted curve, so these differences are not statistically meaningful.

### Correlation across strategies and asset-class trend sleeves

![Correlation](results/figures/strategy_correlation.png)

## Limitations

- **ETFs are not futures.** USO and DBC hold rolling futures and carry roll costs; GLD and SLV hold metal;
  bond ETFs differ from bond futures. Real trend-following CTAs trade futures with far more markets.
- **Small universe.** Only {s['n_assets']} assets, {n_eq} of them equity funds. The trend portfolio's monthly correlation with its
  equity sleeve alone is {num(ts_eq_corr)}, so "cross-asset" diversification is weaker than the asset list suggests.
  Cross-sectional ranks over {s['n_assets']} names are noisy.
- **The period may be unusual.** Falling rates for most of the sample, then the 2022 shock. The hold-out is
  only {oos_n_months} months, during a strong bull market.
- **Transaction costs are estimated**, as a flat per-side rate. There is no bid-ask modelling, market impact or
  borrow cost for shorts.
- **Sharpe uses a 0% risk-free rate**, and idle cash earns 0%. This overstates long-only Sharpe relative to an
  excess-return Sharpe; long/short books would also earn interest on collateral.
- **Selection bias.** The ETF list was chosen with hindsight; all of these funds survived.
- **Single, unofficial data vendor** (yfinance), with no second source to cross-check prices.
- **Past performance does not predict future results.**

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m xasset.pipeline          # download (cached), DQ, curated panel
python -m xasset.research          # signals, backtests, robustness, ML -> results/
python analysis/charts.py          # figures
python analysis/findings.py        # regenerate this README
pytest                             # no-lookahead, DQ, backtest math, reconciliation, README freshness
streamlit run app/app.py           # dashboard (reads results/ only)
```

## Layout

```
src/xasset/    pipeline (ingest, quality, curate, warehouse, report), features, backtest, ml, research
analysis/      charts.py, findings.py (this README)
app/           Streamlit dashboard (pre-aggregated data only; render.yaml for Render)
tests/         pytest suite
results/       small generated outputs and figures (committed)
reports/       data_quality.md
DECISIONS.md   plain-language log of every methodological choice
data/          raw cache, DuckDB, curated panel (git-ignored, regenerated)
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if README.md is out of date")
    args = ap.parse_args()
    text = render(load())
    if args.check:
        current = README.read_text(encoding="utf-8") if README.exists() else ""
        if current != text:
            sys.exit("README.md is stale: run python analysis/findings.py")
        print("README.md is up to date")
        return
    README.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", README)


if __name__ == "__main__":
    main()
