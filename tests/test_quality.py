"""Each DQ rule must fire on a deliberately injected defect, and only there."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from xasset import config
from xasset.calendar import nyse_sessions
from xasset.quality import FLAG_COLUMNS, apply_rules, excluded_mask
from xasset.warehouse import exclusion_sql

END = "2007-07-31"


@pytest.fixture(autouse=True)
def small_universe(monkeypatch):
    monkeypatch.setattr(config, "UNIVERSE", {"AAA": "equity_broad", "BBB": "treasury"})


def make_raw(tickers=("AAA", "BBB"), end=END, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sessions = nyse_sessions(config.START_DATE, end)
    frames = []
    for t in tickers:
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(sessions))))
        frames.append(pd.DataFrame({"ticker": t, "date": sessions, "open": px, "high": px, "low": px,
                                    "close": px, "adj_close": px, "volume": 1_000_000.0}))
    return pd.concat(frames, ignore_index=True)


def flags_for(clean, ticker, date):
    row = clean[(clean["ticker"] == ticker) & (clean["date"] == pd.Timestamp(date))]
    return {c for c in FLAG_COLUMNS if row[c].any()}


def test_clean_data_raises_no_flags():
    clean = apply_rules(make_raw(), end=END)["clean"]
    assert clean[FLAG_COLUMNS].sum().sum() == 0
    assert len(clean) == 2 * len(nyse_sessions(config.START_DATE, END))


def test_duplicate_flags_all_copies_and_excludes_only_extras():
    raw = make_raw()
    raw = pd.concat([raw, raw.iloc[[10]]], ignore_index=True)
    clean = apply_rules(raw, end=END)["clean"]
    assert clean["flag_duplicate_date"].sum() == 2
    assert excluded_mask(clean).sum() == 1


def test_missing_session_is_inserted_not_filled_and_next_return_spans_gap():
    raw = make_raw()
    gone = raw[raw["ticker"] == "AAA"].iloc[20]
    raw = raw.drop(index=gone.name)
    clean = apply_rules(raw, end=END)["clean"].set_index(["ticker", "date"])

    inserted = clean.loc[("AAA", gone["date"])]
    assert inserted["flag_missing_session"] and pd.isna(inserted["adj_close"])
    assert not excluded_mask(clean.reset_index()).any()  # soft rule: kept, flagged

    aaa = clean.loc["AAA"]
    nxt = aaa.index[aaa.index.get_loc(gone["date"]) + 1]
    prev = aaa.index[aaa.index.get_loc(gone["date"]) - 1]
    assert aaa.loc[nxt, "flag_return_spans_gap"]
    # The next return is the true two-day return, not 0% from a forward-filled price.
    assert aaa.loc[nxt, "daily_return"] == pytest.approx(aaa.loc[nxt, "adj_close"] / aaa.loc[prev, "adj_close"] - 1)
    # Cross-asset: BBB traded that day, so the date is flagged for both tickers.
    assert clean.loc[("BBB", gone["date"]), "flag_partial_coverage_date"]


def test_off_calendar_row_is_excluded():
    raw = make_raw()
    sat = raw.iloc[[5]].assign(date=pd.Timestamp("2007-01-13"))  # a Saturday
    clean = apply_rules(pd.concat([raw, sat], ignore_index=True), end=END)["clean"]
    assert flags_for(clean, "AAA", "2007-01-13") == {"flag_off_calendar"}
    assert excluded_mask(clean).sum() == 1


def test_nonpositive_and_missing_prices_are_excluded_and_skipped_for_returns():
    raw = make_raw()
    i, j = raw.index[(raw["ticker"] == "AAA")][[30, 40]]
    raw.loc[i, ["close", "adj_close"]] = 0.0
    raw.loc[j, ["close", "adj_close"]] = np.nan
    clean = apply_rules(raw, end=END)["clean"]
    assert clean["flag_nonpositive_price"].sum() == 1
    assert clean["flag_missing_price"].sum() == 1
    assert excluded_mask(clean).sum() == 2
    # No return is computed from a bad price (would be -100% or inf).
    assert clean["daily_return"].abs().max() < 0.2


@pytest.mark.parametrize("run, expected", [(config.STALE_RUN_SESSIONS + 1, config.STALE_RUN_SESSIONS + 1),
                                           (config.STALE_RUN_SESSIONS - 1, 0)])
def test_stale_run_threshold(run, expected):
    raw = make_raw()
    idx = raw.index[raw["ticker"] == "BBB"][50:50 + run]
    raw.loc[idx, ["close", "adj_close"]] = raw.loc[idx[0], "close"]
    clean = apply_rules(raw, end=END)["clean"]
    assert clean["flag_stale_price"].sum() == expected


def test_extreme_return_flags_spike_and_its_reversal_only():
    raw = make_raw()
    k = raw.index[raw["ticker"] == "AAA"][100]
    raw.loc[k, ["close", "adj_close"]] *= 1.20
    clean = apply_rules(raw, end=END)["clean"]
    hits = clean[clean["flag_extreme_return"]]
    assert set(hits["ticker"]) == {"AAA"}
    assert len(hits) == 2  # the +20% day and the reversal the next day


def test_robust_z_uses_only_past_data():
    """Changing a future return must not change today's z-score (no look-ahead)."""
    base = apply_rules(make_raw(), end=END)["clean"]
    raw = make_raw()
    k = raw.index[raw["ticker"] == "AAA"][120]
    raw.loc[k:k + 5, ["close", "adj_close"]] *= 1.5
    bumped = apply_rules(raw, end=END)["clean"]
    sel = (base["ticker"] == "AAA") & (base["date"] < raw.loc[k, "date"])
    pd.testing.assert_series_equal(base.loc[sel, "robust_z"], bumped.loc[sel, "robust_z"])


def test_universe_screen(monkeypatch):
    monkeypatch.setattr(config, "UNIVERSE", {"AAA": "x", "LATE": "x", "TOO_LATE": "x", "NONE": "x"})
    raw = make_raw(("AAA", "LATE", "TOO_LATE"), end="2008-03-31")
    raw = raw[~((raw["ticker"] == "LATE") & (raw["date"] < "2007-04-11"))]
    raw = raw[~((raw["ticker"] == "TOO_LATE") & (raw["date"] < "2008-02-01"))]
    res = apply_rules(raw, end="2008-03-31")
    status = res["universe"].set_index("ticker")["status"].to_dict()
    assert status == {"AAA": "included", "LATE": "included_late_start",
                      "TOO_LATE": "dropped_late_start", "NONE": "dropped_no_data"}
    # A late starter gets no inserted "missing" rows before its first date.
    late = res["clean"][res["clean"]["ticker"] == "LATE"]
    assert late["date"].min() == pd.Timestamp("2007-04-11") and not late["flag_missing_session"].any()


def test_sql_and_pandas_exclusion_agree():
    raw = make_raw()
    raw = pd.concat([raw, raw.iloc[[3]], raw.iloc[[7]].assign(date=pd.Timestamp("2007-01-14"))], ignore_index=True)
    raw.loc[raw.index[50], "adj_close"] = -1
    clean = apply_rules(raw, end=END)["clean"]
    con = duckdb.connect()
    con.register("p", clean)
    n_sql = con.execute(f"SELECT COUNT(*) FROM p WHERE {exclusion_sql()}").fetchall()[0][0]
    assert n_sql == excluded_mask(clean).sum() == 3
