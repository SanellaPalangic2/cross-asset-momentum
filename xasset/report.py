"""Markdown data-quality report: reports/data_quality.md."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config

RULES = {
    "flag_duplicate_date": "More than one raw row for (ticker, date). All copies flagged; copies after the first excluded.",
    "flag_off_calendar": "Row dated on a day that is not an NYSE session (weekend, holiday).",
    "flag_missing_price": "Raw row with NULL close or adjusted close.",
    "flag_nonpositive_price": "Close or adjusted close <= 0.",
    "flag_missing_session": "NYSE session inside the ticker's active range with no raw row. Inserted with NULL prices; never filled.",
    "flag_stale_price": f"Close (split-adjusted) unchanged for >= {config.STALE_RUN_SESSIONS} consecutive usable sessions (all rows of the run).",
    "flag_zero_volume": "Reported volume is exactly 0.",
    "flag_extreme_return": (f"|daily return| > {config.EXTREME_ABS_RETURN:.0%}, or > {config.EXTREME_ROBUST_Z:g} robust sigmas "
                            f"(1.4826 x MAD over the prior {config.ROBUST_Z_WINDOW} returns; past data only)."),
    "flag_return_spans_gap": "The previous usable price is older than the previous session, so the 'daily' return covers several days.",
    "flag_partial_coverage_date": "Session on which at least one active ticker has no usable price (cross-asset calendar misalignment).",
}


def _md(df: pd.DataFrame, max_rows: int | None = None, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_None._\n"
    note = ""
    if max_rows is not None and len(df) > max_rows:
        note = f"\n_Showing {max_rows} of {len(df)} rows; query DuckDB for the rest._\n"
        df = df.head(max_rows)
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    return df.to_markdown(index=False, floatfmt=floatfmt) + "\n" + note


def _runs(clean: pd.DataFrame) -> pd.DataFrame:
    s = clean[clean["flag_stale_price"]]
    if s.empty:
        return pd.DataFrame()
    return (s.groupby(["ticker", "stale_run_start"])
            .agg(end=("date", "max"), sessions=("date", "size"), close=("close", "first"),
                 median_volume=("volume", "median"))
            .reset_index().rename(columns={"stale_run_start": "start"}))


def write_report(*, universe, counts, recon, per_ticker, clean, coverage, macro,
                 macro_daily, panel) -> str:
    path = config.REPORTS_DIR / "data_quality.md"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    flag_cols = config.HARD_RULES + config.SOFT_RULES
    n_fail = int((~recon["ok"]).sum())

    extreme = clean.loc[clean["flag_extreme_return"],
                        ["date", "ticker", "asset_class", "daily_return", "close_return", "robust_z",
                         "flag_return_spans_gap"]]
    extreme = extreme.reindex(extreme["daily_return"].abs().sort_values(ascending=False).index)
    # Close is split-adjusted only; a big gap vs Adj Close = a large distribution that day.
    corp = clean.loc[(clean["close_return"] - clean["daily_return"]).abs() > 0.05,
                     ["date", "ticker", "close_return", "daily_return"]]
    missing = (clean[clean["flag_missing_session"]].groupby("ticker")["date"]
               .agg(missing_sessions="size", first="min", last="max").reset_index())
    partial = coverage[coverage["missing_tickers"] > 0].copy()
    partial_by_year = (partial.assign(year=partial["date"].dt.year).groupby("year")
                       .agg(partial_dates=("date", "size"), max_missing=("missing_tickers", "max"))
                       .reset_index())
    who_missing = (clean.loc[clean["flag_partial_coverage_date"] & ~clean["is_usable"]
                             & clean["dup_rank"].eq(1), ["date", "ticker"]]
                   .groupby("date")["ticker"].agg(lambda x: ", ".join(sorted(x))).reset_index(name="tickers_missing"))

    m = config.FRED_SERIES
    macro_fill = pd.DataFrame([{
        "column": c,
        "sessions": len(macro_daily),
        "null": int(macro_daily[c].isna().sum()),
        "forward_filled": int(macro_daily[f"{c}_ffilled"].sum()),
        "null_too_stale": int(macro_daily[f"{c}_null_too_stale"].sum()),
    } for c in m.values()])

    total = counts[flag_cols].sum().rename("rows_flagged").rename_axis("rule").reset_index()
    total["type"] = total["rule"].map(lambda r: "hard (excluded)" if r in config.HARD_RULES else "soft (kept, flagged)")
    total["definition"] = total["rule"].map(RULES)

    status_counts = universe["status"].value_counts().to_dict()
    lines = [
        "# Data Quality Report",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `python -m xasset.pipeline`. "
        "Do not edit by hand; rerun the pipeline.",
        "",
        "## Summary",
        "",
        f"- Window: {config.START_DATE} to {config.END_DATE} (NYSE calendar via `exchange_calendars`).",
        f"- Universe: {len(universe)} tickers requested; " + ", ".join(f"{v} {k}" for k, v in status_counts.items()) + ".",
        f"- Clean rows: {len(clean):,}. Analysis-panel rows: {len(panel):,}.",
        f"- Reconciliation: {len(recon) - n_fail} / {len(recon)} checks passed"
        + (" **(FAILURES BELOW)**" if n_fail else "") + ".",
        "- Policy: rows are flagged, never deleted. Hard rules exclude a row from `analysis.panel`; "
        "soft rules keep it and list the rule in `dq_soft_flags`. Prices are never filled.",
        "",
        "## 1. Universe screen",
        "",
        f"Included if first price <= {config.LATE_START_CUTOFF}; a start after {config.START_DATE} is kept as `included_late_start`.",
        "",
        _md(universe[["ticker", "asset_class", "first_date", "last_date", "raw_rows", "status"]]),
        "## 2. Rules and totals",
        "",
        _md(total[["rule", "type", "rows_flagged", "definition"]]),
        "## 3. Counts per rule per ticker",
        "",
        _md(counts.rename(columns=lambda c: c.removeprefix("flag_"))),
        "## 4. Row-count reconciliation",
        "",
        "### Stage checks",
        "",
        _md(recon[recon["stage"] != "per ticker"]),
        "### Per-ticker lineage",
        "",
        "`cache_rows -> raw_rows -> (+ inserted_missing) clean_rows -> (- excluded_hard) panel_rows`",
        "",
        _md(per_ticker),
        f"Per-ticker checks: {int(recon.loc[recon['stage'] == 'per ticker', 'ok'].sum())} / "
        f"{int((recon['stage'] == 'per ticker').sum())} passed.",
        "",
        "## 5. Extreme returns (inspect each one)",
        "",
        "`daily_return` uses Adj Close (split + dividend adjusted). `close_return` uses yfinance `Close`, which is "
        "split-adjusted but NOT dividend-adjusted, so a large gap between them marks a distribution.",
        "",
        _md(extreme, max_rows=60),
        "### Days where Close and Adj Close returns differ by > 5pp (large distributions)",
        "",
        _md(corp, max_rows=40),
        "## 6. Stale-price runs",
        "",
        "A run with normal volume and a plausibly flat price (e.g. a short-Treasury ETF at zero rates) is "
        "likely genuine; a run with zero or missing volume is likely a dead feed.",
        "",
        _md(_runs(clean), max_rows=40),
        "## 7. Missing NYSE sessions",
        "",
        _md(missing),
        "## 8. Cross-asset calendar alignment",
        "",
        f"{len(partial)} of {len(coverage)} sessions have at least one active ticker without a usable price.",
        "",
        _md(partial_by_year),
        _md(who_missing, max_rows=40),
        "## 9. Macro series (FRED)",
        "",
        _md(macro["summary"]),
        f"Forward-fill rule: macro only, lagged {config.MACRO_PUBLICATION_LAG_SESSIONS} session(s) for "
        f"publication, max {config.MACRO_FFILL_LIMIT_SESSIONS} sessions; older values become NULL.",
        "",
        _md(macro_fill),
        f"Spread consistency: |T10Y2Y - (DGS10 - DGS2)| > {config.SPREAD_TOLERANCE_PP} pp on "
        f"{len(macro['spread_mismatches'])} of {int(macro['spread_compared']['rows_compared'].iloc[0])} days "
        f"(max abs diff {macro['spread_compared']['max_abs_diff'].iloc[0]:.4f}). FRED derives T10Y2Y from "
        "the same H.15 inputs, so this checks internal consistency, not independent sources.",
        "",
        _md(macro["spread_mismatches"], max_rows=20),
        "## 10. Analysis panel",
        "",
        f"`analysis.panel` / `data/curated/panel.parquet`: {len(panel):,} rows, "
        f"{panel['date'].min():%Y-%m-%d} to {panel['date'].max():%Y-%m-%d}, columns: "
        + ", ".join(f"`{c}`" for c in panel.columns) + ".",
        "",
        _md(panel.isna().sum().rename("nulls").rename_axis("column").reset_index()),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
