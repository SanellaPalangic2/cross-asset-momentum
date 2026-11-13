"""export_for_databricks.py on a small warehouse built by the ORIGINAL pipeline from synthetic data.

Needs duckdb and exchange_calendars (both in requirements.txt). Skipped without them.
"""
import json

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("exchange_calendars")

import export_for_databricks as exp  # noqa: E402
import pipeline_lib as pl  # noqa: E402
from xasset import config, ingest, pipeline  # noqa: E402
from xasset.calendar import nyse_sessions  # noqa: E402

END = "2007-06-29"
TICKERS = {"AAA": "equity_broad", "BBB": "treasury"}


def fake_yf(ticker, start, end):
    dates = nyse_sessions(start, end)
    px = 100 + np.arange(len(dates), dtype=float) * 0.1
    df = pd.DataFrame({"date": dates, "open": px, "high": px, "low": px, "close": px,
                       "adj_close": px, "volume": 1e6})
    if ticker == "AAA":
        df = pd.concat([df, df.iloc[[5]]])          # duplicate row
        df = df.drop(index=df.index[20])            # missing session
    return df.reset_index(drop=True)


def fake_fred(series_id, start, end):
    return pd.DataFrame({"date": pd.bdate_range(start, end), "value": 4.0})


@pytest.fixture
def warehouse(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UNIVERSE", TICKERS)
    monkeypatch.setattr(config, "END_DATE", END)
    monkeypatch.setattr(config, "RAW_CACHE_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "w.duckdb")
    monkeypatch.setattr(config, "CURATED_DIR", tmp_path / "curated")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setitem(ingest.FETCHERS, "yfinance", fake_yf)
    monkeypatch.setitem(ingest.FETCHERS, "fred", fake_fred)
    monkeypatch.setattr(pl, "END_DATE", END)   # the mirror must follow the patched config
    pipeline.run()
    return tmp_path


def test_export_writes_every_file_with_matching_counts_and_hashes(warehouse):
    out = warehouse / "export"
    m = exp.export(warehouse / "w.duckdb", out)

    assert set(m["files"]) == set(pl.BRONZE_FILES.values()) | {"expected_analysis_panel.parquet",
                                                               "expected_rule_counts.parquet"}
    con = duckdb.connect(str(warehouse / "w.duckdb"), read_only=True)
    for name, meta in m["files"].items():
        assert meta["sha256"] == pl.sha256_file(out / name)
        assert con.execute(f"SELECT COUNT(*) FROM read_parquet('{(out / name).as_posix()}')").fetchall()[0][0] == meta["rows"]
    assert m["files"]["raw_prices.parquet"]["rows"] == m["table_rows"]["raw.prices"]
    assert m["files"]["ref_nyse_sessions.parquet"]["rows"] == len(nyse_sessions(config.START_DATE, END))
    assert m["raw_prices_rows_by_ticker"] == {"AAA": len(nyse_sessions(config.START_DATE, END)),  # +1 dup -1 missing
                                              "BBB": len(nyse_sessions(config.START_DATE, END))}
    assert m["original_rule_totals"]["flag_duplicate_date"] == 2
    assert m["original_rule_totals"]["flag_missing_session"] == 1
    assert pl.config_drift(m["pipeline_config"]) == []
    assert json.loads((out / "manifest.json").read_text()) == m

    # Dates are Parquet DATE (Spark can't read DuckDB's nanosecond timestamps).
    desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{(out / 'raw_prices.parquet').as_posix()}')").fetchall()
    assert {r[0]: r[1] for r in desc}["date"] == "DATE"
    assert not list(out.glob("*.tmp"))


def test_universe_values_sql_keeps_config_order():
    con = duckdb.connect()
    rows = con.execute(f"SELECT * FROM {exp.universe_values_sql({'B': 'x', 'A': 'y'})} ORDER BY position").fetchall()
    assert rows == [("B", "x", 0), ("A", "y", 1)]
