"""Macro alignment rules and ingestion resilience."""
import pandas as pd
import pytest

from xasset import config
from xasset.calendar import nyse_sessions
from xasset.curate import align_macro
from xasset.ingest import EmptyDownloadError, RawCache, download_all, with_retry

END = "2007-03-30"


def macro_obs(dates, values, series_ids=tuple(config.FRED_SERIES)):
    return pd.concat([pd.DataFrame({"series_id": s, "date": pd.to_datetime(dates), "value": values})
                      for s in series_ids], ignore_index=True)


def test_macro_is_lagged_one_session():
    s = nyse_sessions(config.START_DATE, END)
    out = align_macro(macro_obs(s, range(len(s))), end=END).set_index("date")
    # The value observed on session i shows up on session i+1, never on session i.
    assert pd.isna(out["dgs10"].iloc[0])
    assert (out["dgs10"].iloc[1:].to_numpy() == range(len(s) - 1)).all()
    assert (out["dgs10_obs_date"].iloc[1:].to_numpy() == s[:-1].to_numpy()).all()


def test_macro_ffill_is_capped():
    s = nyse_sessions(config.START_DATE, END)
    # A single observation, then the feed dies.
    out = align_macro(macro_obs([s[0]], [4.5]), end=END, lag=1, limit=3)
    filled = out["dgs10"].notna()
    assert filled.sum() == 1 + 3            # the fresh value plus 3 filled sessions
    assert out["dgs10_ffilled"].sum() == 3
    assert out["dgs10_null_too_stale"].sum() == len(s) - 1 - 4


def test_retry_recovers_then_gives_up():
    calls, waits = {"n": 0}, []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise EmptyDownloadError("empty")
        return pd.DataFrame({"x": [1]})

    assert len(with_retry(flaky, "t", max_retries=5, sleep=waits.append)) == 1
    assert len(waits) == 2 and waits[1] > waits[0]  # exponential backoff

    def dead():
        raise ConnectionError("down")
    with pytest.raises(ConnectionError):
        with_retry(dead, "t", max_retries=3, sleep=lambda _: None)


def test_download_resumes_and_isolates_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UNIVERSE", {"AAA": "x", "BAD": "x"})
    monkeypatch.setattr(config, "FRED_SERIES", {"DGS10": "dgs10"})
    monkeypatch.setattr(config, "MAX_RETRIES", 2)
    monkeypatch.setattr("xasset.ingest.time.sleep", lambda _: None)
    hits = []

    def fake(key, start, end):
        hits.append(key)
        if key == "BAD":
            raise ConnectionError("boom")
        return pd.DataFrame({"date": pd.to_datetime(["2007-01-03"]), "value": [1.0]})

    fetchers = {"yfinance": fake, "fred": fake}
    cache = RawCache(tmp_path)
    first = download_all(cache, fetchers=fetchers)
    assert first["failed"] == ["yfinance/BAD"] and len(first["downloaded"]) == 2

    hits.clear()
    second = download_all(RawCache(tmp_path), fetchers=fetchers)  # fresh object: reads manifest from disk
    assert set(second["cached"]) == {"yfinance/AAA", "fred/DGS10"}
    assert set(hits) == {"BAD"}  # only the failed item is re-requested
    assert not list(tmp_path.rglob("*.tmp"))
