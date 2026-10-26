"""Backtest math on toy examples with hand-checked answers."""
import numpy as np
import pandas as pd
import pytest

from xasset.backtest import (equal_weight_targets, information_coefficient, ic_summary, max_drawdown,
                             metrics, run_backtest, ts_trend_targets, xs_momentum_targets)

D = pd.bdate_range("2020-01-01", periods=6)
RETS = pd.DataFrame({"A": [0.0, 0.10, 0.10, -0.05, 0.02, 0.0],
                     "B": [0.0, 0.00, 0.05, 0.00, 0.00, 0.10]}, index=D)


def test_rebalancing_drift_and_costs_by_hand():
    # Signal d0 -> 50/50, traded at the d1 close (lag 1). Signal d2 -> 100% A, traded at the d3 close.
    targets = pd.DataFrame({"A": [0.5, 1.0], "B": [0.5, 0.0]}, index=[D[0], D[2]])
    res = run_backtest(targets, RETS, cost_bps=10, lag=1)

    # d2: first day holding 50/50. Gross = .5*.10 + .5*.05 = .075.
    # Cost of the initial trade: turnover 1.0 * 10bp = .001, charged here.
    assert res.returns.index[0] == D[2]
    assert res.gross[D[2]] == pytest.approx(0.075)
    assert res.returns[D[2]] == pytest.approx(0.074)

    # Drift after d2: wA = .5*1.10/1.075, wB = .5*1.05/1.075.
    wA, wB = 0.55 / 1.075, 0.525 / 1.075
    # d3: gross = wA * -.05. Then drift again and trade to (1, 0) at the close.
    g3 = wA * -0.05
    assert res.gross[D[3]] == pytest.approx(g3)
    wA3, wB3 = wA * 0.95 / (1 + g3), wB / (1 + g3)
    turnover = abs(1 - wA3) + abs(0 - wB3)
    assert res.turnover[D[3]] == pytest.approx(turnover)
    assert res.returns[D[3]] == pytest.approx(g3 - 0.001 * turnover)

    # d4, d5: 100% A.
    assert res.returns[D[4]] == pytest.approx(0.02)
    assert res.returns[D[5]] == pytest.approx(0.0)

    # Costs are fully accounted for: gross - net = 10bp * total turnover.
    assert (res.gross - res.returns).sum() == pytest.approx(0.001 * res.turnover.sum())


def test_zero_cost_and_lag_zero():
    targets = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[D[0]])
    res = run_backtest(targets, RETS, cost_bps=0, lag=0)
    # Traded at the d0 close, so the d1 return is earned.
    assert res.returns.index[0] == D[1]
    assert (1 + res.returns).prod() - 1 == pytest.approx(1.1 * 1.1 * 0.95 * 1.02 * 1.0 - 1)


def test_long_short_drift_by_hand():
    # +1 A / -1 B, dollar neutral. d2: gross = .10 - .05 = .05.
    targets = pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=[D[0]])
    res = run_backtest(targets, RETS, cost_bps=0, lag=1)
    assert res.returns[D[2]] == pytest.approx(0.05)
    # Drift: wA = 1.1/1.05, wB = -1.05/1.05 = -1. d3: gross = (1.1/1.05)*(-.05).
    assert res.returns[D[3]] == pytest.approx((1.1 / 1.05) * -0.05)


def test_metrics_by_hand():
    idx = pd.bdate_range("2021-01-01", periods=3)
    assert max_drawdown(pd.Series([0.10, -0.50, 0.20], index=idx)) == pytest.approx(-0.5)

    r = pd.Series([0.01, -0.01] * 126, index=pd.bdate_range("2021-01-01", periods=252))
    res = run_backtest(pd.DataFrame({"X": [1.0]}, index=[r.index[0]]), r.to_frame("X"), cost_bps=0, lag=0)
    m = metrics(res)
    expected_sharpe = res.returns.mean() / res.returns.std() * np.sqrt(252)
    assert m["sharpe"] == pytest.approx(expected_sharpe)
    assert m["ann_vol"] == pytest.approx(res.returns.std() * np.sqrt(252))
    assert m["ann_return"] == pytest.approx((1 + res.returns).prod() ** (252 / len(res.returns)) - 1)
    assert m["turnover_per_year"] == pytest.approx(1.0 * 252 / len(res.returns))


def test_xs_momentum_picks_top_and_bottom_third():
    d = pd.DatetimeIndex(["2020-01-31"])
    score = pd.DataFrame([[6, 5, 4, 3, 2, 1, np.nan]], index=d, columns=list("ABCDEFG"))
    w = xs_momentum_targets(score, d).iloc[0]
    # 6 valid assets -> k = 2. Long A, B; short E, F; G has no score.
    assert w.to_dict() == {"A": 0.5, "B": 0.5, "C": 0, "D": 0, "E": -0.5, "F": -0.5, "G": 0}


def test_ts_trend_scaling_and_caps():
    d = pd.DatetimeIndex(["2020-01-31"])
    tr = pd.DataFrame([[0.05, -0.02, 0.01]], index=d, columns=list("ABC"))
    vol = pd.DataFrame([[0.20, 0.10, 0.01]], index=d, columns=list("ABC"))
    flat = ts_trend_targets(tr, vol, d, below="flat", vol_target=0.20, max_asset_lev=5, max_gross=10).iloc[0]
    # A: +1 * min(.2/.2, 5)/3; B below the MA -> 0; C: +1 * min(.2/.01=20, 5)/3.
    assert flat.to_dict() == pytest.approx({"A": 1 / 3, "B": 0.0, "C": 5 / 3})
    short = ts_trend_targets(tr, vol, d, below="short", vol_target=0.20, max_asset_lev=5, max_gross=10).iloc[0]
    assert short["B"] == pytest.approx(-2 / 3)
    capped = ts_trend_targets(tr, vol, d, below="short", vol_target=0.20, max_asset_lev=5, max_gross=1.0).iloc[0]
    assert capped.abs().sum() == pytest.approx(1.0)
    assert capped["C"] / capped["A"] == pytest.approx(5.0)  # the cap preserves relative weights


def test_equal_weight_ignores_missing():
    d = pd.DatetimeIndex(["2020-01-31"])
    px = pd.DataFrame([[1.0, 2.0, np.nan, 4.0]], index=d, columns=list("ABCD"))
    assert equal_weight_targets(px, d).iloc[0].tolist() == pytest.approx([1 / 3, 1 / 3, 0, 1 / 3])


def test_information_coefficient():
    d = pd.DatetimeIndex(["2020-01-31", "2020-02-28"])
    sig = pd.DataFrame([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], index=d, columns=list("ABCDE"), dtype=float)
    fwd = pd.DataFrame([[.01, .02, .03, .04, .05], [.05, .04, .03, .02, .01]], index=d, columns=list("ABCDE"))
    ic = information_coefficient(sig, fwd)
    assert ic.tolist() == pytest.approx([1.0, -1.0])
    s = ic_summary(pd.Series([0.1, 0.2, 0.3]))
    assert s["ic_t"] == pytest.approx(0.2 / 0.1 * np.sqrt(3))
