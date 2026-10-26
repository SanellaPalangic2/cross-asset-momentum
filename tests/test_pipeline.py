"""End-to-end pipeline on synthetic, offline data: row counts must reconcile at every stage."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from xasset import config, ingest, pipeline
from xasset.calendar import nyse_sessions

END = "2007-06-29"
TICKERS = {"AAA": "equity_broad", "BBB": "treasury", "LATE": "credit"}


def fake_yf(ticker, start, end):
    dates = nyse_sessions(start, end)
    if ticker == "LATE":
        dates = dates[dates >= "2007-03-01"]
    px = 100 + np.arange(len(dates), dtype=float) * 0.1
    df = pd.DataFrame({"date": dates, "open": px, "high": px, "low": px, "close": px,
                       "adj_close": px, "volume": 1e6})
    if ticker == "AAA":
        df = pd.concat([df, df.iloc[[5]]])          # duplicate row
        df = df.drop(index=df.index[20])            # missing session
        df.loc[df.index[30], "adj_close"] = np.nan  # missing price
    return df.reset_index(drop=True)


def fake_fred(series_id, start, end):
    dates = pd.bdate_range(start, end)
    return pd.DataFrame({"date": dates, "value": 4.0 if series_id != "T10Y2Y" else 0.0})


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UNIVERSE", TICKERS)
    monkeypatch.setattr(config, "END_DATE", END)
    monkeypatch.setattr(config, "RAW_CACHE_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "w.duckdb")
    monkeypatch.setattr(config, "CURATED_DIR", tmp_path / "curated")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setitem(ingest.FETCHERS, "yfinance", fake_yf)
    monkeypatch.setitem(ingest.FETCHERS, "fred", fake_fred)
    return tmp_path


def test_pipeline_reconciles_with_injected_defects(sandbox):
    pipeline.run()
    con = duckdb.connect(str(sandbox / "w.duckdb"), read_only=True)
    recon = con.execute("SELECT * FROM meta.reconciliation").df()
    assert recon["ok"].all() and len(recon) > 5

    n = len(nyse_sessions(config.START_DATE, END))
    late = len(nyse_sessions("2007-03-01", END))
    raw_rows = (n + 1 - 1) + n + late   # AAA: +1 duplicate, -1 missing
    assert con.execute("SELECT COUNT(*) FROM raw.prices").fetchall()[0][0] == raw_rows
    # Clean: raw rows + 1 inserted missing session.
    assert con.execute("SELECT COUNT(*) FROM clean.prices").fetchall()[0][0] == raw_rows + 1
    # Panel: clean minus the extra duplicate copy and the NULL-price row.
    panel = pd.read_parquet(sandbox / "curated" / "panel.parquet")
    assert len(panel) == raw_rows + 1 - 2
    assert not panel.duplicated(["date", "ticker"]).any()
    assert (sandbox / "reports" / "data_quality.md").exists()


def test_rerun_uses_cache(sandbox, monkeypatch):
    pipeline.run()
    def boom(*a, **k):
        raise AssertionError("network called on a cached rerun")
    monkeypatch.setitem(ingest.FETCHERS, "yfinance", boom)
    monkeypatch.setitem(ingest.FETCHERS, "fred", boom)
    pipeline.run()


def test_reconciliation_mismatch_fails_the_run(sandbox, monkeypatch):
    # Make the pandas exclusion logic disagree with the SQL view by one row.
    real = pipeline.excluded_mask
    def off_by_one(df):
        m = real(df).copy()
        m.iloc[0] = not m.iloc[0]
        return m
    monkeypatch.setattr(pipeline, "excluded_mask", off_by_one)
    with pytest.raises(pipeline.ReconciliationError):
        pipeline.run()
