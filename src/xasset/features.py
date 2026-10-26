"""Features and labels. No look-ahead.

Every feature value on date t uses prices up to and including the close of t,
and nothing later. All windows are backward-looking (shift/rolling on the
session index). tests/test_features.py enforces this by scrambling all data
after t and checking that nothing at or before t changes.

Labels (forward returns) are the only forward-looking quantities. They live in
a separate function so they cannot leak into a feature by accident.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

M = config.TRADING_DAYS_PER_MONTH


@dataclass
class Market:
    """Wide (session x ticker) matrices built from analysis.panel."""
    prices: pd.DataFrame       # adj_close (total-return price)
    returns: pd.DataFrame      # daily_return from adj_close
    t10y2y: pd.Series          # already lagged one session for publication in the pipeline
    asset_class: pd.Series     # ticker -> asset class


def load_market(path=None) -> Market:
    panel = pd.read_parquet(path or config.CURATED_DIR / "panel.parquet")
    return market_from_panel(panel)


def market_from_panel(panel: pd.DataFrame) -> Market:
    prices = panel.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    returns = panel.pivot(index="date", columns="ticker", values="daily_return").sort_index()
    t10y2y = panel.groupby("date")["t10y2y"].first().sort_index()
    asset_class = panel.groupby("ticker")["asset_class"].first()
    cols = [t for t in config.UNIVERSE if t in prices.columns]
    return Market(prices[cols], returns[cols], t10y2y, asset_class[cols])


# ------------------------------------------------------------------ features

def momentum(prices: pd.DataFrame, lookback_months: int = config.MOM_LOOKBACK_MONTHS,
             skip_months: int = config.MOM_SKIP_MONTHS) -> pd.DataFrame:
    """Return from t-lookback to t-skip months. 12-1 skips the latest month,
    where short-term reversal works against momentum."""
    return prices.shift(skip_months * M) / prices.shift(lookback_months * M) - 1


def short_term_return(prices: pd.DataFrame, months: int = config.REVERSAL_MONTHS) -> pd.DataFrame:
    """Trailing 1-month return. The reversal signal is its negative."""
    return prices / prices.shift(months * M) - 1


def trend(prices: pd.DataFrame, window: int = config.TREND_MA_DAYS) -> pd.DataFrame:
    """Price relative to its trailing moving average (> 0 means above the MA)."""
    return prices / prices.rolling(window, min_periods=window).mean() - 1


def realized_vol(returns: pd.DataFrame, window: int = config.VOL_WINDOW_DAYS) -> pd.DataFrame:
    """Annualized trailing standard deviation of daily returns."""
    return returns.rolling(window, min_periods=window).std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)


def curve_inverted(t10y2y: pd.Series) -> pd.Series:
    """True when 10Y-2Y < 0. Unknown (NULL) spread counts as not inverted."""
    return t10y2y.lt(0) & t10y2y.notna()


def build_features(mkt: Market) -> dict[str, pd.DataFrame | pd.Series]:
    """The spec's default features, all as of each session's close."""
    return {
        "mom_12_1": momentum(mkt.prices),
        "ret_1m": short_term_return(mkt.prices),
        "trend_200": trend(mkt.prices),
        "vol_60": realized_vol(mkt.returns),
        "curve_inverted": curve_inverted(mkt.t10y2y),
    }


# ------------------------------------------------------------- dates, labels

def signal_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last session of each calendar month."""
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("M")).max().to_numpy(), name="date")


def forward_returns(prices: pd.DataFrame, dates: pd.DatetimeIndex,
                    lag: int = config.EXECUTION_LAG_SESSIONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LABEL: the return a position formed at signal date s actually earns.

    The trade executes at the close of s+lag and is held until the next
    rebalance executes at the close of s_next+lag. Also returns each label's
    window (label_start, label_end), so training code can prove a label was
    already known. Signals without a complete forward window are dropped.
    """
    pos = prices.index.get_indexer(dates)
    start, end = pos[:-1] + lag, pos[1:] + lag
    ok = end < len(prices.index)
    start, end, sig = start[ok], end[ok], dates[:-1][ok]
    fwd = pd.DataFrame(prices.iloc[end].to_numpy() / prices.iloc[start].to_numpy() - 1,
                       index=sig, columns=prices.columns)
    windows = pd.DataFrame({"label_start": prices.index[start], "label_end": prices.index[end]}, index=sig)
    return fwd, windows
