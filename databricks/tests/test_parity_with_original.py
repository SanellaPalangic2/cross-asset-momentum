"""Parity: the ported helpers must give the same answers as the original src/xasset code.

These import the original quality.py / curate.py, which need exchange_calendars,
and warehouse.py, which needs duckdb. Both are in requirements.txt. A test is
skipped where they aren't installed.
"""
import numpy as np
import pandas as pd
import pytest

import pipeline_lib as pl

pytest.importorskip("exchange_calendars")
from xasset import config  # noqa: E402
from xasset.calendar import nyse_sessions  # noqa: E402
from xasset.curate import align_macro as orig_align_macro  # noqa: E402
from xasset.quality import apply_rules, excluded_mask  # noqa: E402

END = "2007-12-31"


@pytest.fixture(autouse=True)
def small_universe(monkeypatch):
    monkeypatch.setattr(config, "UNIVERSE", {"AAA": "equity_broad", "BBB": "treasury", "LATE": "credit"})


def make_raw(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sessions = nyse_sessions(config.START_DATE, END)
    frames = []
    for t in ("AAA", "BBB", "LATE"):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(sessions))))
        frames.append(pd.DataFrame({"ticker": t, "date": sessions, "open": px, "high": px, "low": px,
                                    "close": px, "adj_close": px, "volume": 1_000_000.0}))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[~((raw["ticker"] == "LATE") & (raw["date"] < "2007-04-11"))].reset_index(drop=True)
    return raw


def inject_defects(raw: pd.DataFrame) -> pd.DataFrame:
    """One of every defect the 10 rules look for."""
    aaa = raw.index[raw["ticker"] == "AAA"]
    bbb = raw.index[raw["ticker"] == "BBB"]
    raw = raw.copy()
    raw.loc[aaa[30], ["close", "adj_close"]] = np.nan                      # missing price
    raw.loc[aaa[40], ["close", "adj_close"]] = 0.0                         # non-positive price
    raw.loc[aaa[100], ["close", "adj_close"]] *= 1.25                      # extreme return + reversal
    raw.loc[bbb[60:67], ["close", "adj_close"]] = raw.loc[bbb[60], "close"]  # stale run of 7
    raw.loc[bbb[90], "volume"] = 0.0                                       # zero volume
    raw.loc[aaa[150], "adj_close"] = -1.0                                  # non-positive adj only
    gone = aaa[20]                                                         # missing session -> gap
    extra_dup = raw.loc[[aaa[10]]]                                         # duplicate (ticker, date)
    saturday = raw.loc[[bbb[5]]].assign(date=pd.Timestamp("2007-01-13"))   # off-calendar
    raw = pd.concat([raw.drop(index=gone), extra_dup, saturday], ignore_index=True)
    raw["raw_row_id"] = np.arange(len(raw))
    return raw


def session_positions(dates: pd.Series) -> pd.Series:
    sessions = nyse_sessions(config.START_DATE, END)
    return dates.map(pd.Series(np.arange(len(sessions)), index=sessions))


def test_time_series_rules_match_apply_rules_row_for_row():
    raw = inject_defects(make_raw())
    clean = apply_rules(raw, end=END)["clean"]
    usable = clean[clean["is_usable"]].copy()
    usable["session_pos"] = session_positions(usable["date"])

    # Same shape of call as Spark's applyInPandas: one ticker's usable rows at a time.
    ported = pd.concat([pl.ticker_time_series_rules(g) for _, g in usable.groupby("ticker")]).set_index("raw_row_id")
    orig = usable.set_index("raw_row_id")
    cols = ["daily_return", "split_only_return", "robust_z", "stale_run_length", "stale_run_start",
            *pl.TIME_SERIES_FLAGS]
    for c in cols:
        a, b = ported[c].reindex(orig.index), orig[c]
        if c.startswith("flag_"):
            assert (a.astype(bool) == b.astype(bool)).all(), c
        elif c == "stale_run_start":
            assert (pd.to_datetime(a) == pd.to_datetime(b)).all(), c
        else:
            np.testing.assert_array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), err_msg=c)
    # The injected defects actually fired, so the comparison isn't vacuous.
    for rule in pl.FLAG_COLUMNS:
        assert clean[rule].any(), f"{rule} never fired in the parity fixture"


def test_screen_universe_matches_original():
    raw = inject_defects(make_raw())
    res = apply_rules(raw, end=END)
    valid = raw[raw["adj_close"].notna()]
    stats = valid.groupby("ticker")["date"].agg(first_date="min", last_date="max", raw_rows="size").reset_index()
    first_session = nyse_sessions(config.START_DATE, config.END_DATE)[0]
    ported = pl.screen_universe(stats, config.UNIVERSE, first_session)
    orig = res["universe"]
    assert list(ported["status"]) == list(orig["status"])
    assert list(ported["ticker"]) == list(orig["ticker"])
    assert (ported["first_date"] == orig["first_date"]).all()


def test_align_macro_matches_original(monkeypatch):
    monkeypatch.setattr(config, "END_DATE", END)
    sessions = nyse_sessions(config.START_DATE, END)
    rng = np.random.default_rng(3)
    frames = []
    for sid in config.FRED_SERIES:
        dates = pd.bdate_range(config.START_DATE, END)
        vals = rng.normal(3, 1, len(dates))
        vals[rng.random(len(dates)) < 0.05] = np.nan            # holiday blanks
        df = pd.DataFrame({"series_id": sid, "date": dates, "value": vals})
        if sid == "DTB3":
            df = df[(df["date"] < "2007-06-01") | (df["date"] > "2007-07-15")]   # dead feed -> NULLs
        frames.append(df)
    raw_macro = pd.concat(frames, ignore_index=True)
    raw_macro["date"] = raw_macro["date"].astype("datetime64[ns]")   # as pipeline.load_raw does
    pd.testing.assert_frame_equal(pl.align_macro(raw_macro, sessions), orig_align_macro(raw_macro, end=END),
                                  check_dtype=False)


def test_exclusion_sql_is_the_original_text():
    pytest.importorskip("duckdb")
    from xasset.warehouse import exclusion_sql
    assert pl.exclusion_sql() == exclusion_sql()


def test_panel_sql_gives_the_same_rows_as_the_original_view_in_duckdb():
    duckdb = pytest.importorskip("duckdb")
    from xasset.warehouse import create_analysis_view

    raw = inject_defects(make_raw())
    clean = apply_rules(raw, end=END)["clean"]
    sessions = nyse_sessions(config.START_DATE, END)
    raw_macro = pd.DataFrame({"series_id": np.repeat(list(config.FRED_SERIES), len(sessions)),
                              "date": np.tile(sessions, len(config.FRED_SERIES)), "value": 2.0})
    macro_daily = orig_align_macro(raw_macro, end=END)

    con = duckdb.connect()
    for schema in ("clean", "analysis"):
        con.execute(f"CREATE SCHEMA {schema}")
    con.register("_c", clean)
    con.register("_m", macro_daily)
    con.execute("CREATE TABLE clean.prices AS SELECT * FROM _c")
    con.execute("CREATE TABLE clean.macro_daily AS SELECT * FROM _m")
    create_analysis_view(con)
    orig = con.execute("SELECT * FROM analysis.panel ORDER BY date, ticker").df()
    ported = con.execute(f"SELECT * FROM ({pl.analysis_panel_sql('clean.prices', 'clean.macro_daily')}) "
                         "ORDER BY date, ticker").df()
    pd.testing.assert_frame_equal(ported, orig)
    # dup copy + off-calendar + NULL price + 2 non-positive prices are excluded
    assert len(orig) == len(clean) - int(excluded_mask(clean).sum()) == len(clean) - 5


def test_rules_fire_where_injected():
    """Sanity check on the fixture: the injected defects land on the expected rules."""
    raw = inject_defects(make_raw())
    clean = apply_rules(raw, end=END)["clean"]
    counts = clean[pl.FLAG_COLUMNS].sum()
    assert counts["flag_duplicate_date"] == 2          # both copies flagged
    assert counts["flag_off_calendar"] == 1
    assert counts["flag_missing_price"] == 1
    assert counts["flag_nonpositive_price"] == 2
    assert counts["flag_missing_session"] == 1
    assert counts["flag_zero_volume"] == 1
    assert counts["flag_stale_price"] == 7
