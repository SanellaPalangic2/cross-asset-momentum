"""Pure-Python helpers shared by the Databricks notebooks and the local pytest suite.

Nothing in this module imports pyspark or Databricks utilities, so it runs (and is
tested) on a laptop. The notebooks import it as a workspace file that sits next to them.

What lives here:
- A mirror of the settings in src/xasset/config.py that the pipeline needs. The
  export manifest carries a snapshot of the original config, and the bronze
  notebook fails if the two disagree, so a threshold changed in one place can't
  silently drift from the other. tests/test_pipeline_lib.py checks them too.
- The catalog of the 10 data-quality rules (name, hard/soft, what it checks, how it flags).
- Ports of the parts of quality.py / curate.py that are row-order or time-series
  logic (per-ticker returns, robust z, stale runs, universe screen, macro alignment).
  They are written in pandas so they stay line-for-line comparable with the
  original, and the local tests assert parity against the original functions.
- SQL builders for the set-based rules and the analysis-panel query (run by Spark).
- Reconciliation and manifest helpers.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# =============================================================================
# Config mirror (must equal src/xasset/config.py; checked by tests and by 01_bronze)
# =============================================================================
START_DATE = "2007-01-01"
END_DATE = "2026-09-30"
LATE_START_CUTOFF = "2007-12-31"

FRED_SERIES = {"DGS10": "dgs10", "DGS2": "dgs2", "T10Y2Y": "t10y2y", "DTB3": "dtb3"}

STALE_RUN_SESSIONS = 5
EXTREME_ABS_RETURN = 0.15
EXTREME_ROBUST_Z = 8.0
ROBUST_Z_WINDOW = 252
ROBUST_Z_MIN_PERIODS = 60

MACRO_PUBLICATION_LAG_SESSIONS = 1
MACRO_FFILL_LIMIT_SESSIONS = 5

HARD_RULES = ["flag_duplicate_date", "flag_off_calendar", "flag_missing_price", "flag_nonpositive_price"]
SOFT_RULES = ["flag_missing_session", "flag_stale_price", "flag_zero_volume", "flag_extreme_return",
              "flag_return_spans_gap", "flag_partial_coverage_date"]
FLAG_COLUMNS = HARD_RULES + SOFT_RULES

CONFIG_KEYS = ("START_DATE", "END_DATE", "LATE_START_CUTOFF", "FRED_SERIES", "STALE_RUN_SESSIONS",
               "EXTREME_ABS_RETURN", "EXTREME_ROBUST_Z", "ROBUST_Z_WINDOW", "ROBUST_Z_MIN_PERIODS",
               "MACRO_PUBLICATION_LAG_SESSIONS", "MACRO_FFILL_LIMIT_SESSIONS", "HARD_RULES", "SOFT_RULES")


def config_snapshot(source=None) -> dict:
    """The CONFIG_KEYS values from `source` (a module, e.g. xasset.config) or from this module."""
    if source is None:
        return {k: globals()[k] for k in CONFIG_KEYS}
    return {k: getattr(source, k) for k in CONFIG_KEYS}


def config_drift(other: dict) -> list[str]:
    """Keys whose value in `other` (e.g. the manifest's snapshot) differs from this module."""
    mine = json.loads(json.dumps(config_snapshot()))       # tuples -> lists, like the manifest
    theirs = json.loads(json.dumps(other))
    return [f"{k}: local={mine.get(k)!r} manifest={theirs.get(k)!r}"
            for k in CONFIG_KEYS if mine.get(k) != theirs.get(k)]


# =============================================================================
# The 10 data-quality rules
# =============================================================================
@dataclass(frozen=True)
class Rule:
    name: str
    severity: str   # "hard": excluded from the gold panel; "soft": kept and listed in dq_soft_flags
    kind: str       # "structural" (row/key level), "time_series" (per ticker), "cross_sectional" (per date)
    checks: str
    flags: str

    @property
    def expectation(self) -> str:
        """SQL constraint for a pipeline expectation: a row passes when the flag is not set."""
        return f"NOT {self.name}"


RULES: list[Rule] = [
    Rule("flag_duplicate_date", "hard", "structural",
         "Raw row whose (ticker, date) appears more than once.",
         "Every copy is flagged; only copies with dup_rank > 1 (ordered by raw_row_id) are excluded."),
    Rule("flag_off_calendar", "hard", "structural",
         "Raw row whose date is not an NYSE session in [START_DATE, END_DATE].",
         "Flagged and excluded."),
    Rule("flag_missing_price", "hard", "structural",
         "Raw row with NULL/NaN adj_close or close.",
         "Flagged and excluded; no return is computed from it."),
    Rule("flag_nonpositive_price", "hard", "structural",
         "Raw row with adj_close <= 0 or close <= 0.",
         "Flagged and excluded; no return is computed from it."),
    Rule("flag_missing_session", "soft", "structural",
         "An NYSE session in the ticker's active range with no raw row.",
         "An explicit row is inserted with NULL prices (never filled) and flagged."),
    Rule("flag_stale_price", "soft", "time_series",
         f"Usable row inside a run of >= {STALE_RUN_SESSIONS} consecutive usable rows with the same close.",
         "Every row of the run is flagged; kept."),
    Rule("flag_zero_volume", "soft", "time_series",
         "Usable row with volume == 0.",
         "Flagged; kept."),
    Rule("flag_extreme_return", "soft", "time_series",
         f"Usable row with |daily_return| > {EXTREME_ABS_RETURN:.0%} or |robust z| > {EXTREME_ROBUST_Z:g} "
         f"(trailing {ROBUST_Z_WINDOW}-row median / 1.4826*MAD of past returns, min {ROBUST_Z_MIN_PERIODS}).",
         "Flagged; kept."),
    Rule("flag_return_spans_gap", "soft", "time_series",
         "Usable row whose previous usable row is not the immediately preceding NYSE session.",
         "Flagged; the return keeps its true multi-day value."),
    Rule("flag_partial_coverage_date", "soft", "cross_sectional",
         "Session row on a date where at least one active ticker has no usable price.",
         "Every session row on that date is flagged; kept."),
]
RULES_BY_NAME = {r.name: r for r in RULES}

# Spark SQL predicates for the structural rules, evaluated on the ticker x session
# grid. Columns they use: row_origin, is_session, n_rows_same_key, adj_close, close.
# isnan() matters: pandas treats NaN as missing, Spark does not.
STRUCTURAL_RULE_SQL = {
    "flag_duplicate_date": "row_origin = 'raw' AND n_rows_same_key > 1",
    "flag_off_calendar": "row_origin = 'raw' AND NOT is_session",
    "flag_missing_session": "row_origin = 'inserted_missing_session'",
    "flag_missing_price": ("row_origin = 'raw' AND (adj_close IS NULL OR isnan(adj_close) "
                           "OR close IS NULL OR isnan(close))"),
    # pandas: NaN <= 0 is False, so (NaN, -1) is still flagged. coalesce keeps that for NULLs.
    "flag_nonpositive_price": "row_origin = 'raw' AND coalesce(adj_close <= 0 OR close <= 0, false)",
}

# A row is usable for returns if no hard rule excludes it (quality.py, "usable").
USABLE_SQL = ("row_origin = 'raw' AND dup_rank = 1 AND NOT flag_off_calendar "
              "AND NOT flag_missing_price AND NOT flag_nonpositive_price")

# The universe screen only counts rows with a non-missing adj_close (quality.screen_universe).
SCREEN_VALID_PRICE_SQL = "adj_close IS NOT NULL AND NOT isnan(adj_close)"


def exclusion_sql(hard_rules: list[str] | None = None) -> str:
    """TRUE for rows the hard rules exclude. Mirrors xasset.warehouse.exclusion_sql()."""
    hard_rules = HARD_RULES if hard_rules is None else hard_rules
    parts = [f"({r} AND dup_rank > 1)" if r == "flag_duplicate_date" else r for r in hard_rules]
    return " OR ".join(parts) if parts else "FALSE"


def analysis_panel_sql(prices_table: str, macro_table: str) -> str:
    """SELECT that builds the gold panel. Mirrors xasset.warehouse.create_analysis_view()."""
    macro_cols = ", ".join(f"m.{c}" for c in FRED_SERIES.values())
    ffilled = " OR ".join(f"m.{c}_ffilled" for c in FRED_SERIES.values())
    soft_list = ", ".join(f"CASE WHEN p.{r} THEN '{r.removeprefix('flag_')}' END" for r in SOFT_RULES)
    return f"""
        SELECT
            p.date, p.ticker, p.asset_class,
            p.adj_close, p.daily_return, p.volume,
            {macro_cols},
            ({ffilled}) AS macro_ffilled,
            concat_ws('|', {soft_list}) AS dq_soft_flags
        FROM {prices_table} p
        LEFT JOIN {macro_table} m ON p.date = m.date
        WHERE NOT ({exclusion_sql()})
        ORDER BY p.date, p.ticker
    """


# =============================================================================
# Ported pandas logic
# =============================================================================
def robust_z(r: pd.Series, window: int = ROBUST_Z_WINDOW, min_periods: int = ROBUST_Z_MIN_PERIODS) -> pd.Series:
    """Return in units of trailing robust sigma (1.4826 * MAD), using only past data.

    Port of xasset.quality._robust_z. Note the MAD is the rolling median of
    |past - med| where med is each row's own rolling median, exactly as the original.
    """
    past = r.shift(1)
    roll = past.rolling(window, min_periods=min_periods)
    med = roll.median()
    mad = (past - med).abs().rolling(window, min_periods=min_periods).median()
    return (r - med) / (1.4826 * mad.replace(0, np.nan))


# Output of ticker_time_series_rules, as a Spark DDL schema for applyInPandas.
TIME_SERIES_SCHEMA = (
    "ticker string, raw_row_id bigint, daily_return double, split_only_return double, "
    "robust_z double, stale_run_length int, stale_run_start timestamp, "
    "flag_return_spans_gap boolean, flag_extreme_return boolean, "
    "flag_stale_price boolean, flag_zero_volume boolean"
)
TIME_SERIES_COLUMNS = [c.strip().split(" ")[0] for c in TIME_SERIES_SCHEMA.split(",")]
TIME_SERIES_FLAGS = ["flag_return_spans_gap", "flag_extreme_return", "flag_stale_price", "flag_zero_volume"]


def ticker_time_series_rules(pdf: pd.DataFrame) -> pd.DataFrame:
    """One ticker's USABLE rows -> returns and the four time-series flags.

    Input columns: ticker, raw_row_id, date, session_pos (index of the date in the
    NYSE session list), adj_close, close, volume. Port of quality.apply_rules, the
    "Return-based rules" block. Spark calls it once per ticker via applyInPandas.
    """
    u = pdf.sort_values(["date", "raw_row_id"]).reset_index(drop=True)
    out = pd.DataFrame({"ticker": u["ticker"], "raw_row_id": u["raw_row_id"].astype("int64")})
    out["daily_return"] = u["adj_close"].pct_change(fill_method=None)
    out["split_only_return"] = u["close"].pct_change(fill_method=None)   # diagnostic only, as in quality.py

    # Gap: the previous usable price is not from the immediately preceding session.
    out["flag_return_spans_gap"] = (u["session_pos"] - u["session_pos"].shift(1)).gt(1)

    out["robust_z"] = robust_z(out["daily_return"])
    out["flag_extreme_return"] = (out["daily_return"].abs().gt(EXTREME_ABS_RETURN)
                                  | out["robust_z"].abs().gt(EXTREME_ROBUST_Z))

    # A run is a maximal stretch of consecutive usable rows with the same close.
    run_id = (~u["close"].diff().eq(0)).cumsum()
    runs = u.groupby(run_id)["date"]
    out["stale_run_length"] = runs.transform("size").astype("int32")
    out["stale_run_start"] = runs.transform("min")
    out["flag_stale_price"] = out["stale_run_length"].ge(STALE_RUN_SESSIONS)
    out["flag_zero_volume"] = u["volume"].eq(0)
    return out[TIME_SERIES_COLUMNS]


def screen_universe(stats: pd.DataFrame, universe: dict[str, str], first_session,
                    cutoff: str = LATE_START_CUTOFF) -> pd.DataFrame:
    """Port of quality.screen_universe.

    stats: one row per ticker with first_date, last_date, raw_rows computed over raw
    rows with a non-missing adj_close. universe: ticker -> asset_class, in config order.
    """
    stats = stats.assign(first_date=pd.to_datetime(stats["first_date"]),
                         last_date=pd.to_datetime(stats["last_date"]))
    u = (stats.set_index("ticker")[["first_date", "last_date", "raw_rows"]]
         .reindex(list(universe)).rename_axis("ticker").reset_index())
    u["asset_class"] = u["ticker"].map(universe)
    u["status"] = np.select(
        [u["first_date"].isna(),
         u["first_date"] > pd.Timestamp(cutoff),
         u["first_date"] > pd.Timestamp(first_session)],
        ["dropped_no_data", "dropped_late_start", "included_late_start"],
        default="included")
    return u


def _ns(values) -> pd.Series:
    """Datetime values (dates, strings, any resolution) -> datetime64[ns]. merge_asof needs one resolution."""
    return pd.Series(pd.to_datetime(values)).astype("datetime64[ns]")


def align_macro(raw_macro: pd.DataFrame, sessions, series: dict[str, str] | None = None,
                lag: int = MACRO_PUBLICATION_LAG_SESSIONS,
                limit: int = MACRO_FFILL_LIMIT_SESSIONS) -> pd.DataFrame:
    """raw_macro (series_id, date, value) -> one row per NYSE session. Port of curate.align_macro.

    For session t, each series takes the latest observation dated on or before
    session t-lag. If that observation is more than `limit` sessions old, the value is NULL.
    """
    series = FRED_SERIES if series is None else series
    sessions = pd.DatetimeIndex(_ns(sessions)).sort_values()
    raw_macro = raw_macro.assign(date=_ns(raw_macro["date"]).to_numpy())
    out = pd.DataFrame({"date": sessions})
    out["as_of"] = out["date"].shift(lag)
    for series_id, col in series.items():
        obs = (raw_macro.loc[raw_macro["series_id"].eq(series_id) & raw_macro["value"].notna(),
                             ["date", "value"]]
               .rename(columns={"date": "obs_date", "value": col}).sort_values("obs_date"))
        m = pd.merge_asof(out[["as_of"]].dropna().sort_values("as_of"), obs,
                          left_on="as_of", right_on="obs_date", direction="backward")
        m = out[["as_of"]].merge(m, on="as_of", how="left")
        pos_asof = sessions.searchsorted(m["as_of"].fillna(sessions[0]), side="right")
        pos_obs = sessions.searchsorted(m["obs_date"].fillna(sessions[0]), side="right")
        stale = pd.Series(pos_asof - pos_obs, dtype="float").where(m["obs_date"].notna())
        too_old = stale.gt(limit)
        out[col] = m[col].where(~too_old).to_numpy()
        out[f"{col}_obs_date"] = m["obs_date"].where(~too_old).to_numpy()
        out[f"{col}_ffilled"] = stale.gt(0).to_numpy() & ~too_old.to_numpy()
        out[f"{col}_null_too_stale"] = too_old.to_numpy()
    return out.drop(columns="as_of")


def macro_daily_schema(series: dict[str, str] | None = None) -> str:
    """Spark DDL schema for align_macro's output."""
    series = FRED_SERIES if series is None else series
    cols = ["date date"]
    for c in series.values():
        cols += [f"{c} double", f"{c}_obs_date date", f"{c}_ffilled boolean", f"{c}_null_too_stale boolean"]
    return ", ".join(cols)


def wide_to_long(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{name: wide (date x ticker) frame} -> one long frame with columns date, ticker, <name>..."""
    out = None
    for name, wide in frames.items():
        long = (wide.rename_axis(index="date", columns="ticker").reset_index()
                .melt(id_vars="date", var_name="ticker", value_name=name))
        out = long if out is None else out.merge(long, on=["date", "ticker"], how="outer")
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def nulls_for_spark(pdf: pd.DataFrame, date_cols=()) -> pd.DataFrame:
    """Make a pandas frame safe for spark.createDataFrame with an explicit schema.

    NaN / NaT become None (Spark would otherwise keep a float NaN, which is not NULL),
    and the named datetime columns become python dates for DATE columns.
    """
    out = pdf.copy()
    for c in date_cols:
        out[c] = pd.to_datetime(out[c]).dt.date
    return out.astype(object).where(out.notna(), None)


# =============================================================================
# Reconciliation and manifest helpers
# =============================================================================
# bronze table -> uploaded file. The expected_* files are not loaded into bronze;
# 04_reconcile reads them straight from the volume.
BRONZE_FILES = {
    "prices": "raw_prices.parquet",
    "macro": "raw_macro.parquet",
    "ref_nyse_sessions": "ref_nyse_sessions.parquet",
    "ref_universe": "ref_universe.parquet",
}
# Feed contract: bronze refuses a file that is missing any of these columns.
REQUIRED_COLUMNS = {
    "prices": {"raw_row_id", "date", "ticker", "open", "high", "low", "close", "adj_close", "volume",
               "source", "downloaded_at"},
    "macro": {"series_id", "date", "value", "source", "downloaded_at"},
    "ref_nyse_sessions": {"date"},
    "ref_universe": {"ticker", "asset_class", "position"},
}
CHECK_SCHEMA = "stage string, check string, expected bigint, actual bigint, ok boolean"

# Gold panel vs the original analysis.panel: numeric columns compared with a
# tolerance, the rest compared exactly (NULL-safe).
PANEL_NUMERIC_COLUMNS = ["adj_close", "daily_return", "volume", "dgs10", "dgs2", "t10y2y", "dtb3"]
PANEL_EXACT_COLUMNS = ["asset_class", "macro_ffilled", "dq_soft_flags"]
VALUE_TOLERANCE = 1e-12   # abs(a - b) <= TOL + TOL * abs(b); the same pandas ops should give identical values


def missing_columns(table: str, columns) -> set[str]:
    return REQUIRED_COLUMNS[table] - set(columns)


class ReconciliationError(RuntimeError):
    pass


def check(stage: str, what: str, expected, actual) -> dict:
    """One reconciliation check as a row (stage, check, expected, actual, ok)."""
    expected, actual = int(expected), int(actual)
    return {"stage": stage, "check": what, "expected": expected, "actual": actual, "ok": expected == actual}


def failed(checks: list[dict]) -> list[dict]:
    return [c for c in checks if not c["ok"]]


def assert_all_passed(checks: list[dict]) -> None:
    bad = failed(checks)
    if bad:
        lines = "\n".join(f"  [{c['stage']}] {c['check']}: expected {c['expected']}, got {c['actual']}"
                          for c in bad)
        raise ReconciliationError(f"{len(bad)} of {len(checks)} reconciliation checks failed:\n{lines}")


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_files(manifest: dict, directory: str) -> list[dict]:
    """Every file in the manifest exists in `directory` and has the recorded SHA-256.

    Returns check rows (expected 1 = present / matching). A partial or corrupted
    upload shows up here before any table is written.
    """
    out = []
    for name, meta in sorted(manifest["files"].items()):
        path = os.path.join(directory, name)
        present = os.path.exists(path)
        out.append(check("feed", f"{name}: file present", 1, int(present)))
        if present:
            out.append(check("feed", f"{name}: sha256 matches manifest", 1,
                             int(sha256_file(path) == meta["sha256"])))
    return out


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_identifier(name: str) -> str:
    """Reject anything but a plain identifier, since names are interpolated into SQL."""
    if not _IDENT.match(name or ""):
        raise ValueError(f"not a plain SQL identifier: {name!r}")
    return name
