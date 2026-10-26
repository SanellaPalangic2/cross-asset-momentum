"""Macro alignment and macro data-quality checks.

Forward-fill rule (documented in docs/decisions.md):
- Macro series MAY be forward-filled onto NYSE sessions. A yield is a slowly
  moving state variable, and the last published value is what a trader knew.
  The fill is capped at MACRO_FFILL_LIMIT_SESSIONS.
- Prices are NEVER filled. A filled price creates a fake 0% return and hides a
  data problem.
- Macro is lagged by MACRO_PUBLICATION_LAG_SESSIONS so a value appears on a
  session only once it would have been published.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .calendar import nyse_sessions


def align_macro(raw_macro: pd.DataFrame, end: str | None = None,
                lag: int | None = None, limit: int | None = None) -> pd.DataFrame:
    """raw_macro (series_id, date, value) -> one row per NYSE session.

    For session t, each series takes the latest observation dated on or before
    session t-lag. If that observation is more than `limit` sessions old, the
    value is NULL.
    """
    end = end or config.END_DATE  # defaults resolved at call time, not import time
    lag = config.MACRO_PUBLICATION_LAG_SESSIONS if lag is None else lag
    limit = config.MACRO_FFILL_LIMIT_SESSIONS if limit is None else limit
    sessions = nyse_sessions(config.START_DATE, end)
    out = pd.DataFrame({"date": sessions})
    out["as_of"] = out["date"].shift(lag)  # NaT for the first `lag` sessions
    for series_id, col in config.FRED_SERIES.items():
        obs = (raw_macro.loc[raw_macro["series_id"].eq(series_id) & raw_macro["value"].notna(),
                             ["date", "value"]]
               .rename(columns={"date": "obs_date", "value": col}).sort_values("obs_date"))
        m = pd.merge_asof(out[["as_of"]].dropna().sort_values("as_of"), obs,
                          left_on="as_of", right_on="obs_date", direction="backward")
        m = out[["as_of"]].merge(m, on="as_of", how="left")
        # Staleness = sessions that passed between the observation and the as-of date.
        # An observation on a non-session (e.g. Good Friday) counts from the next session.
        pos_asof = sessions.searchsorted(m["as_of"].fillna(sessions[0]), side="right")
        pos_obs = sessions.searchsorted(m["obs_date"].fillna(sessions[0]), side="right")
        stale = pd.Series(pos_asof - pos_obs, dtype="float").where(m["obs_date"].notna())
        too_old = stale.gt(limit)
        out[col] = m[col].where(~too_old).to_numpy()
        out[f"{col}_obs_date"] = m["obs_date"].where(~too_old).to_numpy()
        out[f"{col}_ffilled"] = stale.gt(0).to_numpy() & ~too_old.to_numpy()
        out[f"{col}_null_too_stale"] = too_old.to_numpy()
    return out.drop(columns="as_of")


def macro_checks(raw_macro: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Checks on the raw observations: range sanity and spread consistency."""
    wide = (raw_macro.pivot_table(index="date", columns="series_id", values="value", aggfunc="first")
            .sort_index())
    summary = []
    for sid in config.FRED_SERIES:
        s = wide.get(sid, pd.Series(dtype=float))
        summary.append({
            "series_id": sid,
            "rows": int(s.size), "missing_values": int(s.isna().sum()),
            "first_date": s.dropna().index.min(), "last_date": s.dropna().index.max(),
            # Yields outside [-2%, 20%] are implausible for the US over this window.
            "out_of_range": int(((s < -2) | (s > 20)).sum()),
            "max_gap_days": int(s.dropna().index.to_series().diff().dt.days.max()),
        })
    spread = pd.DataFrame(index=wide.index)
    spread["published"] = wide.get("T10Y2Y")
    spread["computed"] = wide.get("DGS10") - wide.get("DGS2")
    spread["diff"] = spread["published"] - spread["computed"]
    spread = spread.dropna()
    mismatches = spread[spread["diff"].abs() > config.SPREAD_TOLERANCE_PP].reset_index()
    return {"summary": pd.DataFrame(summary), "spread_mismatches": mismatches,
            "spread_compared": pd.DataFrame({"rows_compared": [len(spread)],
                                             "max_abs_diff": [spread["diff"].abs().max()]})}
