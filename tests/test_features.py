"""No-lookahead: changing any data AFTER date t must not change anything AT OR BEFORE t.

Checked for every feature, the strategy weights, and the backtest P&L. A
deliberately leaky feature is included as a positive control, to prove the
check can fail.
"""
import numpy as np
import pandas as pd
import pytest

from xasset import config
from xasset.backtest import run_backtest, ts_trend_targets, xs_momentum_targets
from xasset.calendar import nyse_sessions
from xasset.features import (build_features, curve_inverted, forward_returns, market_from_panel,
                             momentum, realized_vol, short_term_return, signal_dates, trend)

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
END = "2009-12-31"


def synthetic_panel(seed=0, scramble_after=None, base=None):
    """Random-walk panel. If scramble_after is set, every value after that date is replaced."""
    rng = np.random.default_rng(seed)
    dates = nyse_sessions(config.START_DATE, END)
    if base is None:
        rets = rng.normal(0.0003, 0.012, (len(dates), len(TICKERS)))
        spread = np.cumsum(rng.normal(0, 0.05, len(dates)))
    else:
        rets, spread = base
        rets, spread = rets.copy(), spread.copy()
    if scramble_after is not None:
        after = dates > pd.Timestamp(scramble_after)
        rng2 = np.random.default_rng(seed + 99)
        rets[after] = rng2.normal(0.0, 0.05, (after.sum(), len(TICKERS)))
        spread[after] = rng2.normal(0, 2, after.sum())
    prices = 100 * np.cumprod(1 + rets, axis=0)
    daily = np.vstack([np.full(len(TICKERS), np.nan), prices[1:] / prices[:-1] - 1])
    panel = pd.DataFrame({
        "date": np.repeat(dates, len(TICKERS)),
        "ticker": np.tile(TICKERS, len(dates)),
        "adj_close": prices.ravel(), "daily_return": daily.ravel(),
        "t10y2y": np.repeat(spread, len(TICKERS)),
        "asset_class": "equity_broad",
    })
    return panel, (rets, spread)


@pytest.fixture(autouse=True)
def universe(monkeypatch):
    monkeypatch.setattr(config, "UNIVERSE", {t: "equity_broad" for t in TICKERS})


CUTOFFS = ["2008-03-31", "2008-11-14", "2009-06-30"]

FEATURES = {
    "mom_12_1": lambda m: momentum(m.prices, 12, 1),
    "mom_6_1": lambda m: momentum(m.prices, 6, 1),
    "ret_1m": lambda m: short_term_return(m.prices),
    "trend_200": lambda m: trend(m.prices, 200),
    "trend_50": lambda m: trend(m.prices, 50),
    "vol_60": lambda m: realized_vol(m.returns),
    "curve_inverted": lambda m: curve_inverted(m.t10y2y),
}


def _assert_unchanged_through(a, b, cutoff):
    a, b = a.loc[:cutoff], b.loc[:cutoff]
    assert a.notna().to_numpy().any(), "feature is all-NaN before cutoff; test would be vacuous"
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b))


def _markets(cutoff):
    panel, base = synthetic_panel()
    scrambled, _ = synthetic_panel(base=base, scramble_after=cutoff)
    return market_from_panel(panel), market_from_panel(scrambled)


@pytest.mark.parametrize("cutoff", CUTOFFS)
@pytest.mark.parametrize("name", list(FEATURES))
def test_feature_ignores_future_data(name, cutoff):
    clean, scrambled = _markets(cutoff)
    _assert_unchanged_through(FEATURES[name](clean), FEATURES[name](scrambled), cutoff)


def test_check_detects_a_leaky_feature():
    """Positive control: a feature that peeks one day ahead must be caught."""
    clean, scrambled = _markets(CUTOFFS[1])
    leaky = lambda m: m.prices.shift(-1) / m.prices - 1  # noqa: E731
    with pytest.raises(AssertionError):
        _assert_unchanged_through(leaky(clean), leaky(scrambled), CUTOFFS[1])


def test_build_features_defaults_ignore_future_data():
    clean, scrambled = _markets(CUTOFFS[2])
    fa, fb = build_features(clean), build_features(scrambled)
    for k in fa:
        _assert_unchanged_through(fa[k], fb[k], CUTOFFS[2])


@pytest.mark.parametrize("cutoff", CUTOFFS[1:])
def test_strategy_weights_and_pnl_ignore_future_data(cutoff):
    clean, scrambled = _markets(cutoff)
    out = []
    for m in (clean, scrambled):
        dates = signal_dates(m.prices.index)
        f = build_features(m)
        xs = xs_momentum_targets(f["mom_12_1"], dates)
        ts = ts_trend_targets(f["trend_200"], f["vol_60"], dates, below="short")
        out.append((xs, ts, run_backtest(xs, m.returns).returns, run_backtest(ts, m.returns).returns))
    for a, b in zip(*out):
        _assert_unchanged_through(a, b, cutoff)


def test_positions_take_effect_after_the_signal_date():
    """A target decided at the close of s must not earn any return on or before s + lag."""
    clean, _ = _markets(CUTOFFS[0])
    dates = signal_dates(clean.prices.index)
    s = dates[15]
    tgt = pd.DataFrame([[1.0] + [0.0] * 5], index=[s], columns=TICKERS)
    res = run_backtest(tgt, clean.returns, cost_bps=0, lag=config.EXECUTION_LAG_SESSIONS)
    first_earning = clean.prices.index[clean.prices.index.get_loc(s) + config.EXECUTION_LAG_SESSIONS + 1]
    assert res.returns.index[0] == first_earning
    assert res.returns.iloc[0] == pytest.approx(clean.returns.loc[first_earning, "AAA"])


def test_forward_label_window_matches_execution():
    clean, _ = _markets(CUTOFFS[0])
    idx = clean.prices.index
    dates = signal_dates(idx)
    fwd, win = forward_returns(clean.prices, dates)
    s, s_next = dates[10], dates[11]
    lag = config.EXECUTION_LAG_SESSIONS
    start, end = idx[idx.get_loc(s) + lag], idx[idx.get_loc(s_next) + lag]
    assert (win.loc[s, "label_start"], win.loc[s, "label_end"]) == (start, end)
    assert fwd.loc[s, "AAA"] == pytest.approx(clean.prices.loc[end, "AAA"] / clean.prices.loc[start, "AAA"] - 1)
    # The label of signal s always ends after s: it is forward-looking by design and must
    # never be used as a feature.
    assert (win["label_start"] > win.index).all()
