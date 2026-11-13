"""Standalone tests for databricks/pipeline_lib.py. They need only pandas and numpy (no Spark, DuckDB, or Databricks)."""
import json

import numpy as np
import pandas as pd
import pytest

import pipeline_lib as pl
from xasset import config   # config.py imports nothing beyond pathlib


# ------------------------------------------------------------------ rule catalog / config

def test_exactly_the_ten_original_rules_in_original_order():
    assert [r.name for r in pl.RULES] == config.HARD_RULES + config.SOFT_RULES
    assert len(pl.RULES) == 10
    assert {r.name for r in pl.RULES if r.severity == "hard"} == set(config.HARD_RULES)
    assert {r.name for r in pl.RULES if r.severity == "soft"} == set(config.SOFT_RULES)


def test_every_structural_rule_has_a_sql_predicate_and_nothing_else_does():
    structural = {r.name for r in pl.RULES if r.kind == "structural"}
    assert set(pl.STRUCTURAL_RULE_SQL) == structural
    time_series = {r.name for r in pl.RULES if r.kind == "time_series"}
    assert set(pl.TIME_SERIES_FLAGS) == time_series
    assert {r.name for r in pl.RULES if r.kind == "cross_sectional"} == {"flag_partial_coverage_date"}


def test_expectations_pass_when_flag_is_not_set():
    assert pl.RULES_BY_NAME["flag_stale_price"].expectation == "NOT flag_stale_price"


def test_config_mirror_matches_src_xasset_config():
    assert pl.config_drift(pl.config_snapshot(config)) == []


def test_config_drift_is_reported():
    snap = json.loads(json.dumps(pl.config_snapshot()))
    snap["STALE_RUN_SESSIONS"] = 6
    assert pl.config_drift(snap) == ["STALE_RUN_SESSIONS: local=5 manifest=6"]


def test_exclusion_sql_matches_original_definition():
    # Same text as xasset.warehouse.exclusion_sql() (that module needs duckdb, see test_parity_with_original.py).
    assert pl.exclusion_sql() == ("(flag_duplicate_date AND dup_rank > 1) OR flag_off_calendar "
                                  "OR flag_missing_price OR flag_nonpositive_price")
    assert pl.exclusion_sql([]) == "FALSE"


def test_panel_sql_lists_every_soft_rule_and_applies_exclusions():
    sql = pl.analysis_panel_sql("c.silver.prices", "c.gold.macro_daily")
    for rule in config.SOFT_RULES:
        assert f"CASE WHEN p.{rule} THEN '{rule.removeprefix('flag_')}' END" in sql
    assert f"WHERE NOT ({pl.exclusion_sql()})" in sql
    assert "close" not in sql.replace("adj_close", "")   # research input is total-return only


@pytest.mark.parametrize("bad", ["", "1abc", "x; DROP TABLE y", "a.b", "a-b"])
def test_safe_identifier_rejects_non_identifiers(bad):
    with pytest.raises(ValueError):
        pl.safe_identifier(bad)


def test_safe_identifier_accepts_plain_names():
    assert pl.safe_identifier("xasset_dev") == "xasset_dev"


# ------------------------------------------------------------------ time-series rules

def make_usable(n=300, seed=0, ticker="AAA"):
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"ticker": ticker, "raw_row_id": np.arange(n), "date": pd.bdate_range("2010-01-04", periods=n),
                         "session_pos": np.arange(n), "adj_close": px, "close": px, "volume": 1e6})


def flags(out):
    return out[pl.TIME_SERIES_FLAGS].sum().to_dict()


def test_clean_series_raises_no_flags_and_returns_are_adj_close_pct_change():
    u = make_usable()
    out = pl.ticker_time_series_rules(u)
    assert sum(flags(out).values()) == 0
    expected = u["adj_close"] / u["adj_close"].shift(1) - 1
    np.testing.assert_allclose(out["daily_return"], expected, rtol=0, atol=0, equal_nan=True)


def test_output_columns_match_the_spark_schema():
    out = pl.ticker_time_series_rules(make_usable())
    schema_cols = [c.strip().split(" ")[0] for c in pl.TIME_SERIES_SCHEMA.split(",")]
    assert list(out.columns) == schema_cols


def test_ex_dividend_day_uses_adj_close_not_close():
    u = make_usable()
    u["adj_close"], u["close"] = 100.0, 100.0
    u.loc[150:, "close"] = 97.0                         # price drops by the dividend...
    u.loc[:149, "adj_close"] = 100.0 * (1 - 3 / 100)    # ...history is back-adjusted
    u.loc[150:, "adj_close"] = 97.0
    out = pl.ticker_time_series_rules(u)
    assert out.loc[150, "daily_return"] == pytest.approx(0.0)
    assert out.loc[150, "split_only_return"] == pytest.approx(-0.03)


@pytest.mark.parametrize("run, expected", [(pl.STALE_RUN_SESSIONS, pl.STALE_RUN_SESSIONS),
                                           (pl.STALE_RUN_SESSIONS - 1, 0)])
def test_stale_run_threshold_flags_every_row_of_the_run(run, expected):
    u = make_usable()
    u.loc[50:50 + run - 1, ["close", "adj_close"]] = u.loc[50, "close"]
    out = pl.ticker_time_series_rules(u)
    assert out["flag_stale_price"].sum() == expected
    if expected:
        assert (out.loc[out["flag_stale_price"], "stale_run_start"] == u.loc[50, "date"]).all()


def test_zero_volume_is_flagged():
    u = make_usable()
    u.loc[[10, 11], "volume"] = 0
    assert flags(pl.ticker_time_series_rules(u))["flag_zero_volume"] == 2


def test_return_after_a_gap_is_flagged_and_keeps_its_true_value():
    u = make_usable().drop(index=100).reset_index(drop=True)   # session 100 missing
    out = pl.ticker_time_series_rules(u)
    assert out["flag_return_spans_gap"].sum() == 1
    i = out.index[out["flag_return_spans_gap"]][0]
    assert out.loc[i, "daily_return"] == pytest.approx(u.loc[i, "adj_close"] / u.loc[i - 1, "adj_close"] - 1)


def test_extreme_return_flags_spike_and_its_reversal():
    u = make_usable()
    u.loc[200, ["close", "adj_close"]] *= 1.20
    out = pl.ticker_time_series_rules(u)
    assert set(out.index[out["flag_extreme_return"]]) == {200, 201}


def test_robust_z_uses_only_past_data():
    base = pl.ticker_time_series_rules(make_usable())
    u = make_usable()
    u.loc[220:225, ["close", "adj_close"]] *= 1.5
    bumped = pl.ticker_time_series_rules(u)
    pd.testing.assert_series_equal(base.loc[:219, "robust_z"], bumped.loc[:219, "robust_z"])


def test_rows_arrive_unsorted_from_spark():
    u = make_usable()
    shuffled = u.sample(frac=1, random_state=1)
    pd.testing.assert_frame_equal(pl.ticker_time_series_rules(shuffled), pl.ticker_time_series_rules(u))


# ------------------------------------------------------------------ universe screen, macro

def test_screen_universe_statuses():
    stats = pd.DataFrame({"ticker": ["AAA", "LATE", "TOO_LATE"],
                          "first_date": pd.to_datetime(["2007-01-03", "2007-04-11", "2008-02-01"]).date,
                          "last_date": pd.to_datetime(["2026-09-30"] * 3).date,
                          "raw_rows": [10, 9, 8]})
    universe = {"AAA": "x", "LATE": "y", "TOO_LATE": "z", "NONE": "w"}
    u = pl.screen_universe(stats, universe, pd.Timestamp("2007-01-03"))
    assert list(u["ticker"]) == list(universe)
    assert dict(zip(u["ticker"], u["status"])) == {"AAA": "included", "LATE": "included_late_start",
                                                   "TOO_LATE": "dropped_late_start", "NONE": "dropped_no_data"}


def test_align_macro_lags_one_session_and_caps_forward_fill():
    sessions = pd.bdate_range("2020-01-01", periods=12)
    raw = pd.DataFrame({"series_id": "DGS10", "date": [sessions[0]], "value": [1.5]})
    out = pl.align_macro(raw, sessions, series={"DGS10": "dgs10"}, lag=1, limit=5)
    assert pd.isna(out.loc[0, "dgs10"])                      # day 0's value isn't published until day 1
    assert out.loc[1, "dgs10"] == 1.5 and not out.loc[1, "dgs10_ffilled"]
    assert out.loc[2:6, "dgs10_ffilled"].all()               # filled for 5 sessions...
    assert out.loc[7:, "dgs10"].isna().all() and out.loc[7:, "dgs10_null_too_stale"].all()   # ...then NULL


def test_align_macro_accepts_python_dates_from_spark():
    sessions = pd.Series(pd.bdate_range("2020-01-01", periods=5).date)
    raw = pd.DataFrame({"series_id": "DGS2", "date": list(sessions), "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = pl.align_macro(raw, sessions, series={"DGS2": "dgs2"})
    assert list(out["dgs2"].iloc[1:]) == [1.0, 2.0, 3.0, 4.0]


def test_macro_daily_schema_matches_align_macro_columns():
    sessions = pd.bdate_range("2020-01-01", periods=3)
    raw = pd.DataFrame({"series_id": list(pl.FRED_SERIES), "date": sessions[0], "value": 1.0})
    cols = list(pl.align_macro(raw, sessions).columns)
    assert cols == [c.split(" ")[0] for c in pl.macro_daily_schema().split(", ")]


# ------------------------------------------------------------------ conversion helpers

def test_wide_to_long():
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="date")
    a = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, np.nan]}, index=idx)
    b = pd.DataFrame({"X": [5.0, 6.0], "Y": [7.0, 8.0]}, index=idx)
    out = pl.wide_to_long({"a": a, "b": b})
    assert list(out.columns) == ["date", "ticker", "a", "b"] and len(out) == 4
    assert pd.isna(out.loc[(out["ticker"] == "Y") & (out["date"] == "2020-01-02"), "a"]).all()


def test_nulls_for_spark_turns_nan_and_nat_into_none():
    pdf = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", None]), "x": [1.5, np.nan], "b": [True, False]})
    out = pl.nulls_for_spark(pdf, date_cols=["d"])
    assert out.loc[0, "d"] == pd.Timestamp("2020-01-01").date() and out.loc[1, "d"] is None
    assert out.loc[1, "x"] is None and out.loc[0, "x"] == 1.5


# ------------------------------------------------------------------ reconciliation and manifest

def test_assert_all_passed_lists_failures():
    checks = [pl.check("a", "ok one", 3, 3), pl.check("b", "bad one", 3, 4)]
    assert [c["check"] for c in pl.failed(checks)] == ["bad one"]
    with pytest.raises(pl.ReconciliationError, match="bad one: expected 3, got 4"):
        pl.assert_all_passed(checks)
    pl.assert_all_passed(checks[:1])


def test_verify_files_catches_missing_and_corrupted_uploads(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"abc")
    (tmp_path / "b.parquet").write_bytes(b"xyz")
    manifest = {"files": {"a.parquet": {"sha256": pl.sha256_file(tmp_path / "a.parquet")},
                          "b.parquet": {"sha256": "0" * 64},
                          "c.parquet": {"sha256": "0" * 64}}}
    bad = {c["check"] for c in pl.failed(pl.verify_files(manifest, str(tmp_path)))}
    assert bad == {"b.parquet: sha256 matches manifest", "c.parquet: file present"}


def test_bronze_contract_covers_every_bronze_file():
    assert set(pl.REQUIRED_COLUMNS) == set(pl.BRONZE_FILES)
    assert pl.missing_columns("macro", ["series_id", "date", "value"]) == {"source", "downloaded_at"}
