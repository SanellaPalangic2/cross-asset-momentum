"""Data-quality rules: raw prices -> clean prices with one boolean column per rule.

Principles:
- Never delete. A rule only sets its own flag column; whether a flag excludes
  a row is decided later by config.HARD_RULES, so every decision is countable
  and reversible.
- Missing sessions can't be flagged on a row that doesn't exist, so the clean
  table is built on a full (ticker x NYSE session) grid. A missing day becomes
  an explicit row with NULL prices and flag_missing_session = True.
- Prices are never filled. A missing price stays NULL.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .calendar import nyse_sessions

FLAG_COLUMNS = config.HARD_RULES + config.SOFT_RULES


def screen_universe(raw: pd.DataFrame, start: str = config.START_DATE,
                    cutoff: str = config.LATE_START_CUTOFF) -> pd.DataFrame:
    """One row per ticker: first/last date and an include/flag/drop decision."""
    valid = raw.dropna(subset=["adj_close"])
    first_session = nyse_sessions(start, config.END_DATE)[0]
    u = (valid.groupby("ticker")["date"].agg(first_date="min", last_date="max", raw_rows="size")
         .reindex(list(config.UNIVERSE)).rename_axis("ticker").reset_index())
    u["asset_class"] = u["ticker"].map(config.UNIVERSE)
    u["status"] = np.select(
        [u["first_date"].isna(),
         u["first_date"] > pd.Timestamp(cutoff),
         u["first_date"] > first_session],
        ["dropped_no_data", "dropped_late_start", "included_late_start"],
        default="included")
    return u


def _build_grid(raw: pd.DataFrame, universe: pd.DataFrame, end: str) -> pd.DataFrame:
    """Attach raw rows to a full ticker x session grid over each ticker's active range."""
    keep = universe.loc[universe["status"].str.startswith("included"), ["ticker", "first_date"]]
    raw = raw[raw["ticker"].isin(keep["ticker"])].copy()
    raw["row_origin"] = "raw"

    sessions = nyse_sessions(config.START_DATE, end)
    grid = pd.concat([pd.DataFrame({"ticker": t, "date": sessions[sessions >= first]})
                      for t, first in keep.itertuples(index=False)], ignore_index=True)
    # Sessions with no raw row at all become inserted rows. Prices stay NULL.
    present = raw[["ticker", "date"]].drop_duplicates()
    missing = grid.merge(present, on=["ticker", "date"], how="left", indicator=True)
    missing = missing.loc[missing["_merge"] == "left_only", ["ticker", "date"]]
    missing["row_origin"] = "inserted_missing_session"

    out = pd.concat([raw, missing], ignore_index=True)
    out["is_session"] = out["date"].isin(sessions)
    return out.sort_values(["ticker", "date", "raw_row_id"], na_position="last").reset_index(drop=True)


def _robust_z(r: pd.Series) -> pd.Series:
    """Return in units of trailing robust sigma (1.4826 * MAD), using only past data."""
    past = r.shift(1)
    roll = past.rolling(config.ROBUST_Z_WINDOW, min_periods=config.ROBUST_Z_MIN_PERIODS)
    med = roll.median()
    mad = (past - med).abs().rolling(config.ROBUST_Z_WINDOW, min_periods=config.ROBUST_Z_MIN_PERIODS).median()
    return (r - med) / (1.4826 * mad.replace(0, np.nan))


def apply_rules(raw: pd.DataFrame, end: str = config.END_DATE) -> dict[str, pd.DataFrame]:
    """Run every rule. Returns {"clean", "universe", "coverage"}."""
    raw = raw.copy()
    if "raw_row_id" not in raw:
        raw["raw_row_id"] = np.arange(len(raw))
    universe = screen_universe(raw)
    df = _build_grid(raw, universe, end)
    sessions = nyse_sessions(config.START_DATE, end)
    is_raw = df["row_origin"].eq("raw")

    # Structural rules
    df["dup_rank"] = df.groupby(["ticker", "date"]).cumcount() + 1
    df["flag_duplicate_date"] = is_raw & df.groupby(["ticker", "date"])["date"].transform("size").gt(1)
    df["flag_off_calendar"] = is_raw & ~df["is_session"]
    df["flag_missing_session"] = df["row_origin"].eq("inserted_missing_session")
    df["flag_missing_price"] = is_raw & (df["adj_close"].isna() | df["close"].isna())
    df["flag_nonpositive_price"] = is_raw & ((df["adj_close"] <= 0) | (df["close"] <= 0))

    # A row is usable for returns if no hard rule excludes it.
    usable = (is_raw & df["dup_rank"].eq(1) & ~df["flag_off_calendar"]
              & ~df["flag_missing_price"] & ~df["flag_nonpositive_price"])
    df["is_usable"] = usable

    # Return-based rules, computed on the usable series only.
    u = df.loc[usable, ["ticker", "date", "adj_close", "close", "volume"]].copy()
    g = u.groupby("ticker")
    u["daily_return"] = g["adj_close"].pct_change(fill_method=None)
    u["split_only_return"] = g["close"].pct_change(fill_method=None)   # split- but not dividend-adjusted
    u["prev_date"] = g["date"].shift(1)

    # Gap: the previous usable price is not from the immediately preceding session,
    # so this "daily" return actually covers several days.
    session_pos = pd.Series(np.arange(len(sessions)), index=sessions)
    u["flag_return_spans_gap"] = (u["prev_date"].notna()
                                  & (u["date"].map(session_pos) - u["prev_date"].map(session_pos)).gt(1))

    u["robust_z"] = g["daily_return"].transform(_robust_z)
    u["flag_extreme_return"] = (u["daily_return"].abs().gt(config.EXTREME_ABS_RETURN)
                                | u["robust_z"].abs().gt(config.EXTREME_ROBUST_Z))

    # A run is a maximal stretch of consecutive usable rows with the same close.
    run_id = (~g["close"].diff().eq(0)).groupby(u["ticker"]).cumsum()
    runs = u.groupby([u["ticker"], run_id])["date"]
    u["stale_run_length"] = runs.transform("size")
    u["stale_run_start"] = runs.transform("min")
    u["flag_stale_price"] = u["stale_run_length"].ge(config.STALE_RUN_SESSIONS)
    u["flag_zero_volume"] = u["volume"].eq(0)

    cols = ["daily_return", "split_only_return", "robust_z", "stale_run_length", "stale_run_start",
            "flag_return_spans_gap", "flag_extreme_return", "flag_stale_price", "flag_zero_volume"]
    df = df.join(u[cols])
    for c in cols:
        if c.startswith("flag_"):
            df[c] = df[c].fillna(False).astype(bool)

    # Cross-asset alignment: on a session, did every active ticker print a usable price?
    on_session = df[df["is_session"] & df["dup_rank"].eq(1)]
    coverage = (on_session.groupby("date")
                .agg(active_tickers=("ticker", "size"), priced_tickers=("is_usable", "sum"))
                .reset_index())
    coverage["missing_tickers"] = coverage["active_tickers"] - coverage["priced_tickers"]
    partial = set(coverage.loc[coverage["missing_tickers"] > 0, "date"])
    df["flag_partial_coverage_date"] = df["is_session"] & df["date"].isin(partial)

    for c in FLAG_COLUMNS:
        df[c] = df[c].astype(bool)
    df["asset_class"] = df["ticker"].map(config.UNIVERSE)
    return {"clean": df, "universe": universe, "coverage": coverage}


def excluded_mask(df: pd.DataFrame) -> pd.Series:
    """Rows dropped from the analysis view under the current HARD_RULES."""
    mask = pd.Series(False, index=df.index)
    for rule in config.HARD_RULES:
        hit = df[rule] & df["dup_rank"].gt(1) if rule == "flag_duplicate_date" else df[rule]
        mask |= hit
    return mask
