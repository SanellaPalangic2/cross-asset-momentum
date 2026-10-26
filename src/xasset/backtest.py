"""Backtest engine, strategy rules, metrics and information coefficient.

Engine conventions (each is tested on a hand-checked toy in tests/test_backtest.py):
- Targets are decided at the close of a signal date s using data up to s.
- The trade executes at the close of s + lag (config.EXECUTION_LAG_SESSIONS),
  so new weights first earn the return of session s + lag + 1.
- Between trades, weights drift with prices: w_i <- w_i (1 + r_i) / (1 + r_p).
- Unallocated capital (1 - sum w) earns the risk-free rate; if sum w > 1, the
  excess is financed at that rate. A dollar-neutral long/short book therefore
  earns rf on its collateral, and its excess return is just the spread.
- Sharpe is computed on EXCESS returns (portfolio return - rf), which equals
  sum_i w_i (r_i - rf), the same definition for long-only, long/short and
  levered books.
- Cost on a trade day = bps * sum_i |target_i - drifted_i|, i.e. a per-side
  rate on traded notional, charged against that day's return.
- Turnover is reported as sum |delta w| (buys + sells) per year.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config

DPY = config.TRADING_DAYS_PER_YEAR


@dataclass
class BacktestResult:
    returns: pd.Series        # daily net returns
    gross: pd.Series          # daily returns before costs
    costs: pd.Series          # daily cost drag
    turnover: pd.Series       # sum |dw| on each trade date
    weights: pd.DataFrame     # post-trade weights on each trade date
    rf: pd.Series | None = None  # daily risk-free return the cash earned (None = 0%)


def run_backtest(targets: pd.DataFrame, returns: pd.DataFrame,
                 cost_bps: float = config.COST_BPS_PER_SIDE,
                 lag: int = config.EXECUTION_LAG_SESSIONS,
                 rf: pd.Series | None = None) -> BacktestResult:
    """Simulate holding `targets` (signal date x ticker) over daily `returns`.

    rf: daily risk-free return earned by cash. None means 0% (the old convention).
    """
    targets = targets.reindex(columns=returns.columns).fillna(0.0)
    idx = returns.index
    trade_pos = idx.get_indexer(targets.index) + lag
    if (idx.get_indexer(targets.index) < 0).any():
        raise ValueError("signal dates must be sessions in the returns index")
    keep = trade_pos < len(idx)
    trades = dict(zip(trade_pos[keep], targets.to_numpy()[keep]))
    first = int(trade_pos[keep].min())

    r = returns.to_numpy(dtype=float)
    r = np.where(np.isnan(r), 0.0, r)  # a held asset with no price that day contributes 0
    rf_arr = (np.zeros(len(idx)) if rf is None
              else rf.reindex(idx).to_numpy(dtype=float))
    c = cost_bps / 1e4
    n = len(idx)
    w = np.zeros(r.shape[1])
    gross, cost = np.zeros(n), np.zeros(n)
    to_dates, to_vals, w_rows = [], [], []
    for i in range(first, n):
        if np.isnan(rf_arr[i]):
            raise ValueError(f"risk-free rate missing on {idx[i].date()}")
        rp = w @ r[i] + (1 - w.sum()) * rf_arr[i]   # assets + cash (or financing) for day i
        gross[i] = rp
        if rp > -1:
            w = w * (1 + r[i]) / (1 + rp)   # drift to end-of-day weights (cash absorbs the rest)
        if i in trades:                     # trade at today's close
            tgt = trades[i]
            to = np.abs(tgt - w).sum()
            cost[i] = c * to
            w = tgt.copy()
            to_dates.append(idx[i]); to_vals.append(to); w_rows.append(tgt)
    sl = slice(first + 1, n)                # first day with a position
    gross_s = pd.Series(gross[sl], index=idx[sl])
    # The first trade's cost is charged on the trade date itself, which is the
    # day before the series starts. Fold it into the first return so no cost is lost.
    cost_s = pd.Series(cost[sl], index=idx[sl])
    cost_s.iloc[0] += cost[first]
    # Book the initial trade's turnover on the same day as its cost.
    to_index = pd.DatetimeIndex([idx[first + 1] if d == idx[first] else d for d in to_dates])
    return BacktestResult(returns=gross_s - cost_s, gross=gross_s, costs=cost_s,
                          turnover=pd.Series(to_vals, index=to_index),
                          weights=pd.DataFrame(w_rows, index=pd.DatetimeIndex(to_dates), columns=returns.columns),
                          rf=pd.Series(rf_arr[sl], index=idx[sl]))


# ---------------------------------------------------------------- strategies

def xs_momentum_targets(score: pd.DataFrame, dates: pd.DatetimeIndex,
                        quantile: float = config.XS_QUANTILE) -> pd.DataFrame:
    """Long the top `quantile` by score, short the bottom, equal weight (+1 / -1 legs)."""
    rows = {}
    for d in dates:
        s = score.loc[d].dropna()
        k = int(np.floor(len(s) * quantile))
        w = pd.Series(0.0, index=score.columns)
        if k >= 1:
            ranked = s.sort_values(kind="mergesort")  # deterministic tie-break
            w[ranked.index[-k:]] = 1.0 / k
            w[ranked.index[:k]] = -1.0 / k
        rows[d] = w
    return pd.DataFrame(rows).T


def ts_trend_targets(trend_score: pd.DataFrame, vol: pd.DataFrame, dates: pd.DatetimeIndex,
                     below: str = "flat", vol_target: float = config.VOL_TARGET_PER_ASSET,
                     max_asset_lev: float = config.MAX_ASSET_LEVERAGE,
                     max_gross: float = config.MAX_GROSS_LEVERAGE) -> pd.DataFrame:
    """Long assets above their MA; short (or flat) below; each scaled to vol_target / N."""
    if below not in ("flat", "short"):
        raise ValueError(below)
    rows = {}
    for d in dates:
        t, v = trend_score.loc[d], vol.loc[d]
        ok = t.notna() & v.notna() & (v > 0)
        n = int(ok.sum())
        w = pd.Series(0.0, index=trend_score.columns)
        if n:
            sign = np.where(t[ok] > 0, 1.0, -1.0 if below == "short" else 0.0)
            lev = np.minimum(vol_target / v[ok], max_asset_lev)
            w[ok] = sign * lev / n
            gross = w.abs().sum()
            if gross > max_gross:
                w *= max_gross / gross
        rows[d] = w
    return pd.DataFrame(rows).T


def equal_weight_targets(prices: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """1/N over assets with a price on the signal date."""
    avail = prices.loc[dates].notna()
    return avail.div(avail.sum(axis=1), axis=0).astype(float)


# ------------------------------------------------------------------- metrics

def max_drawdown(returns: pd.Series) -> float:
    nav = (1 + returns).cumprod()
    return float((nav / nav.cummax() - 1).min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    nav = (1 + returns).cumprod()
    return nav / nav.cummax() - 1


def monthly_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).groupby(returns.index.to_period("M")).prod() - 1


def metrics(res: BacktestResult, start=None, end=None) -> dict:
    """Performance over [start, end). Sharpe is on excess returns over res.rf (0% if None)."""
    r = res.returns
    if start is not None:
        r = r[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r[r.index < pd.Timestamp(end)]
    if len(r) < 2:
        return {}
    years = len(r) / DPY
    to = res.turnover[(res.turnover.index >= r.index[0]) & (res.turnover.index <= r.index[-1])]
    m = monthly_returns(r)
    rf = res.rf.reindex(r.index).fillna(0.0) if res.rf is not None else pd.Series(0.0, index=r.index)
    ex = r - rf
    sd, sd_ex = r.std(), ex.std()
    return {
        "start": r.index[0].date().isoformat(), "end": r.index[-1].date().isoformat(),
        "ann_return": float((1 + r).prod() ** (1 / years) - 1),
        "ann_vol": float(sd * np.sqrt(DPY)),
        "sharpe": float(ex.mean() / sd_ex * np.sqrt(DPY)) if sd_ex > 0 else np.nan,
        "rf_ann": float((1 + rf).prod() ** (1 / years) - 1),
        "max_drawdown": max_drawdown(r),
        "turnover_per_year": float(to.sum() / years),
        "hit_rate_monthly": float((m > 0).mean()),
        "months": int(len(m)),
        "cost_drag_per_year": float(res.costs.reindex(r.index).sum() / years),
    }


def information_coefficient(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Per signal date: Spearman rank correlation between signal and forward return."""
    out = {}
    for d in fwd.index:
        if d not in signal.index:
            continue
        pair = pd.concat([signal.loc[d], fwd.loc[d]], axis=1).dropna()
        if len(pair) >= 5 and pair.iloc[:, 0].nunique() > 1:
            out[d] = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic
    return pd.Series(out, dtype=float)


def ic_summary(ic: pd.Series) -> dict:
    """Mean, std and t-stat (mean / (std / sqrt(n))) of monthly ICs.

    Monthly ICs come from non-overlapping forward windows, so treating them as
    roughly independent for the t-stat is reasonable.
    """
    ic = ic.dropna()
    n = len(ic)
    sd = ic.std()
    return {"ic_mean": float(ic.mean()), "ic_std": float(sd), "ic_t": float(ic.mean() / sd * np.sqrt(n)) if n > 1 and sd > 0 else np.nan,
            "ic_hit": float((ic > 0).mean()), "n_months": int(n)}
