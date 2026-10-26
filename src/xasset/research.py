"""Research run: strategies, IS-only parameter selection, OOS evaluation, robustness, ML.

    python -m xasset.research          (after python -m xasset.pipeline)

Writes small, pre-aggregated files to results/. The README (analysis/findings.py),
the charts and the dashboard all read only those files.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .backtest import (BacktestResult, equal_weight_targets, ic_summary, information_coefficient,
                       metrics, monthly_returns, run_backtest, ts_trend_targets, xs_momentum_targets)
from .features import (build_features, forward_returns, load_market, momentum, signal_dates, trend)
from .ml import build_dataset, walk_forward

log = logging.getLogger("xasset.research")

OOS = pd.Timestamp(config.OOS_START)
STRATEGY_LABELS = {
    "xs_mom": "XS momentum (L/S terciles)",
    "ts_trend": "TS trend (vol-scaled)",
    "ew_rebal": "Equal weight, monthly rebalanced",
    "ew_hold": "Equal weight, buy and hold",
    "ml_xs": "ML rank (L/S terciles, walk-forward)",
}


def check_holdout_definition() -> None:
    expected = pd.Timestamp(config.END_DATE) - pd.DateOffset(years=config.HOLDOUT_YEARS) + pd.Timedelta(days=1)
    assert OOS == expected, f"OOS_START {OOS.date()} != END_DATE - {config.HOLDOUT_YEARS}y + 1d ({expected.date()})"


def in_sample_only(res: BacktestResult) -> BacktestResult:
    """A copy of a backtest with everything on or after OOS_START removed."""
    def cut(s):
        return s[s.index < OOS]
    return replace(res, returns=cut(res.returns), gross=cut(res.gross), costs=cut(res.costs),
                   turnover=cut(res.turnover), weights=res.weights[res.weights.index < OOS])


def select_on_is(results: dict) -> object:
    """Grid key with the best IN-SAMPLE Sharpe. Out-of-sample data is cut off before scoring."""
    scores = {k: metrics(in_sample_only(v))["sharpe"] for k, v in results.items()}
    return max(scores, key=scores.get)


def first_common_signal(dates: pd.DatetimeIndex, feats: list[pd.DataFrame], min_frac=0.9) -> pd.Timestamp:
    """First signal date on which every warm-up feature exists for >= 90% of tickers."""
    ok = pd.Series(True, index=dates)
    for f in feats:
        ok &= f.loc[dates].notna().mean(axis=1) >= min_frac
    return ok[ok].index[0]


def split_signals(windows: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """IS signals: label ended before OOS. OOS signals: label started on/after OOS.
    The one signal whose label straddles the boundary is purged from both."""
    is_ = windows.index[windows["label_end"] < OOS]
    oos = windows.index[windows["label_start"] >= OOS]
    return is_, oos


def run() -> dict:
    check_holdout_definition()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mkt = load_market()
    P, R = mkt.prices, mkt.returns
    feats = build_features(mkt)
    all_dates = signal_dates(P.index)
    warm = [momentum(P, max(l for l, _ in config.XS_MOM_GRID), 0), trend(P, max(config.TREND_MA_GRID)),
            feats["vol_60"]]
    first_sig = first_common_signal(all_dates, warm)
    dates = all_dates[all_dates >= first_sig]
    fwd, windows = forward_returns(P, dates)
    log.info("signals %s .. %s (%d); OOS from %s", dates[0].date(), dates[-1].date(), len(dates), OOS.date())

    # ---------------------------------------------------------- parameter grids
    xs_grid = {(lb, sk): run_backtest(xs_momentum_targets(momentum(P, lb, sk), dates), R)
               for lb, sk in config.XS_MOM_GRID}
    ts_grid = {(ma, below): run_backtest(ts_trend_targets(trend(P, ma), feats["vol_60"], dates, below=below), R)
               for ma in config.TREND_MA_GRID for below in config.TREND_BELOW_GRID}
    xs_key, ts_key = select_on_is(xs_grid), select_on_is(ts_grid)
    log.info("selected on IS: xs_mom=%s ts_trend=%s", xs_key, ts_key)

    grid_rows = []
    for fam, grid, sel, fmt in [("xs_mom", xs_grid, xs_key, lambda k: f"{k[0]}-{k[1]} months"),
                                ("ts_trend", ts_grid, ts_key, lambda k: f"MA {k[0]}d, {k[1]} below")]:
        for k, res in grid.items():
            mi, mo = metrics(res, end=OOS), metrics(res, start=OOS)
            grid_rows.append({"family": fam, "params": fmt(k), "selected": k == sel,
                              "spec_default": k in [(config.MOM_LOOKBACK_MONTHS, config.MOM_SKIP_MONTHS)]
                              or (fam == "ts_trend" and k[0] == config.TREND_MA_DAYS),
                              "is_sharpe": mi["sharpe"], "is_ann_return": mi["ann_return"],
                              "oos_sharpe": mo["sharpe"], "oos_ann_return": mo["ann_return"]})
    pd.DataFrame(grid_rows).to_csv(config.RESULTS_DIR / "param_grid.csv", index=False)

    # ------------------------------------------------------ main strategies
    mom_sel = momentum(P, *xs_key)
    trend_sel = trend(P, ts_key[0])
    targets = {
        "xs_mom": xs_momentum_targets(mom_sel, dates),
        "ts_trend": ts_trend_targets(trend_sel, feats["vol_60"], dates, below=ts_key[1]),
        "ew_rebal": equal_weight_targets(P, dates),
        "ew_hold": equal_weight_targets(P, dates[:1]),
    }

    # ML walk-forward predictions -> its own L/S strategy
    ds = build_dataset(feats, fwd, windows, dates)
    pred, train_log = walk_forward(ds)
    train_log.to_csv(config.RESULTS_DIR / "ml_training_log.csv", index=False)
    targets["ml_xs"] = xs_momentum_targets(pred, pred.index)

    results = {k: run_backtest(t, R) for k, t in targets.items()}

    rows = []
    for k, res in results.items():
        for period, kw in [("in-sample", {"end": OOS}), ("out-of-sample", {"start": OOS}), ("full", {})]:
            rows.append({"strategy": k, "label": STRATEGY_LABELS[k], "period": period, **metrics(res, **kw)})
    perf = pd.DataFrame(rows)
    perf.to_csv(config.RESULTS_DIR / "performance.csv", index=False)

    daily = pd.DataFrame({k: r.returns for k, r in results.items()})
    daily.to_parquet(config.RESULTS_DIR / "daily_returns.parquet")

    # ------------------------------------------------------------------- IC
    signals = {
        "mom_12_1": feats["mom_12_1"],
        f"mom_selected ({xs_key[0]}-{xs_key[1]})": mom_sel,
        "reversal (-ret_1m)": -feats["ret_1m"],
        "trend_200": feats["trend_200"],
        "ml_pred": pred,
    }
    ic = pd.DataFrame({k: information_coefficient(v, fwd) for k, v in signals.items()})
    ic.index.name = "date"
    ic.to_csv(config.RESULTS_DIR / "ic_monthly.csv")
    is_sig, oos_sig = split_signals(windows)
    ic_rows = []
    for name in ic:
        for period, idx in [("in-sample", is_sig), ("out-of-sample", oos_sig)]:
            ic_rows.append({"signal": name, "period": period, **ic_summary(ic.loc[ic.index.intersection(idx), name])})
    pd.DataFrame(ic_rows).to_csv(config.RESULTS_DIR / "ic_summary.csv", index=False)

    # ML vs momentum on exactly the same months, paired.
    ml_cmp = []
    for period, idx in [("in-sample", is_sig), ("out-of-sample", oos_sig)]:
        both = ic.loc[ic.index.intersection(idx), ["ml_pred", "mom_12_1"]].dropna()
        diff = both["ml_pred"] - both["mom_12_1"]
        t, p = stats.ttest_1samp(diff, 0.0) if len(diff) > 2 else (np.nan, np.nan)
        ml_cmp.append({"period": period, "months": len(both), "ml_ic_mean": both["ml_pred"].mean(),
                       "mom_ic_mean": both["mom_12_1"].mean(), "diff_mean": diff.mean(),
                       "diff_t": float(t), "diff_p": float(p)})
    pd.DataFrame(ml_cmp).to_csv(config.RESULTS_DIR / "ml_vs_momentum.csv", index=False)

    # ------------------------------------------------------------ robustness
    cost_rows = []
    for k in ("xs_mom", "ts_trend", "ew_rebal"):
        for bps in config.COST_SENSITIVITY_BPS:
            res = run_backtest(targets[k], R, cost_bps=bps)
            for period, kw in [("in-sample", {"end": OOS}), ("out-of-sample", {"start": OOS})]:
                m = metrics(res, **kw)
                cost_rows.append({"strategy": k, "cost_bps": bps, "period": period, "sharpe": m["sharpe"],
                                  "ann_return": m["ann_return"], "turnover_per_year": m["turnover_per_year"]})
    pd.DataFrame(cost_rows).to_csv(config.RESULTS_DIR / "cost_sensitivity.csv", index=False)

    sub_rows = []
    for name, (a, b) in config.SUBPERIODS.items():
        for k in ("xs_mom", "ts_trend", "ew_rebal"):
            r = daily[k].loc[a:b].dropna()
            nav = (1 + r).cumprod()
            sub_rows.append({"subperiod": name, "start": a, "end": b, "strategy": k,
                             "total_return": float(nav.iloc[-1] - 1),
                             "ann_vol": float(r.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)),
                             "max_drawdown": float((nav / nav.cummax() - 1).min())})
    pd.DataFrame(sub_rows).to_csv(config.RESULTS_DIR / "subperiods.csv", index=False)

    # Regime: a month is "inverted" if the curve was inverted at the previous
    # month-end signal date, i.e. known before the month's returns.
    inv = feats["curve_inverted"].reindex(all_dates)
    reg_rows = []
    for k in ("xs_mom", "ts_trend", "ew_rebal"):
        m = monthly_returns(daily[k].dropna())
        prev_inv = inv.groupby(inv.index.to_period("M")).last().shift(1).reindex(m.index)
        for state, mask in [("inverted", prev_inv == True), ("not inverted", prev_inv == False)]:  # noqa: E712
            x = m[mask]
            reg_rows.append({"strategy": k, "regime": state, "months": int(len(x)),
                             "oos_months": int((x.index.to_timestamp() >= OOS).sum()),
                             "mean_monthly": float(x.mean()),
                             "ann_sharpe": float(x.mean() / x.std() * np.sqrt(12)) if len(x) > 2 else np.nan,
                             "hit_rate": float((x > 0).mean())})
    pd.DataFrame(reg_rows).to_csv(config.RESULTS_DIR / "regime.csv", index=False)

    # Asset-class sleeves of the trend strategy, for the correlation heatmap.
    sleeve = {}
    for name, classes in config.SLEEVES.items():
        cols = mkt.asset_class.index[mkt.asset_class.isin(classes)]
        t = ts_trend_targets(trend_sel[cols], feats["vol_60"][cols], dates, below=ts_key[1])
        sleeve[f"trend: {name}"] = run_backtest(t, R[cols]).returns
    corr_in = pd.concat([daily[["xs_mom", "ts_trend", "ew_rebal"]].rename(columns={
        "xs_mom": "XS momentum", "ts_trend": "TS trend", "ew_rebal": "Equal weight"}),
        pd.DataFrame(sleeve)], axis=1).dropna()
    monthly = corr_in.apply(monthly_returns)
    monthly.corr().to_csv(config.RESULTS_DIR / "strategy_correlation.csv")

    # -------------------------------------------------------------- summary
    summary = {
        "data_start": str(P.index[0].date()), "data_end": str(P.index[-1].date()),
        "n_assets": int(P.shape[1]), "first_signal": str(dates[0].date()),
        "last_signal": str(dates[-1].date()), "oos_start": str(OOS.date()),
        "holdout_years": config.HOLDOUT_YEARS, "cost_bps": config.COST_BPS_PER_SIDE,
        "execution_lag": config.EXECUTION_LAG_SESSIONS,
        "selected": {"xs_mom": {"lookback_months": xs_key[0], "skip_months": xs_key[1]},
                     "ts_trend": {"ma_days": ts_key[0], "below": ts_key[1]}},
        "ml_first_test_year": int(train_log["test_year"].min()),
        "n_is_signals": int(len(is_sig)), "n_oos_signals": int(len(oos_sig)),
    }
    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("results written to %s", config.RESULTS_DIR)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()


if __name__ == "__main__":
    main()
