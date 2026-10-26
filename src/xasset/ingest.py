"""Raw ingestion: API -> local parquet cache.

Design:
- One file per ticker / series, so a crash mid-run loses at most one item and a
  rerun resumes by skipping files that already exist.
- Files are written to a temp name and then renamed. The rename is atomic, so a
  half-written file never looks like a finished download.
- The cache stores what the API returned, unmodified. All cleaning happens
  downstream, so it can be re-run without touching the network.
- A manifest records when each file was downloaded. yfinance recomputes
  adjusted prices on every pull, so the download time is part of the data.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config

log = logging.getLogger(__name__)


class EmptyDownloadError(RuntimeError):
    """The API returned no rows. yfinance does this instead of raising on many failures."""


def with_retry(fn: Callable[[], pd.DataFrame], label: str,
               max_retries: int | None = None, base: float | None = None,
               cap: float | None = None,
               sleep: Callable[[float], None] | None = None) -> pd.DataFrame:
    """Call fn, retrying with exponential backoff plus jitter.

    Jitter spreads retries out so many requests don't all hit a rate-limited
    API at the same moment. Defaults are read from config at call time, not
    import time.
    """
    max_retries = max_retries or config.MAX_RETRIES
    base = base if base is not None else config.BACKOFF_BASE_SECONDS
    cap = cap if cap is not None else config.BACKOFF_MAX_SECONDS
    sleep = sleep or time.sleep
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # network errors, rate limits, empty payloads
            if attempt == max_retries:
                raise
            wait = min(cap, base * 2 ** (attempt - 1)) * (1 + random.random() * 0.25)
            log.warning("%s: attempt %d/%d failed (%s); retrying in %.1fs",
                        label, attempt, max_retries, exc, wait)
            sleep(wait)
    raise AssertionError("unreachable")


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    # yfinance's `end` is exclusive; add a day so END_DATE itself is included.
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(ticker, start=start, end=end_excl, auto_adjust=False,
                         actions=False, progress=False, threads=False,
                         multi_level_index=False)
    if df is None or df.empty:
        raise EmptyDownloadError(f"no rows for {ticker}")
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                            "Adj Close": "adj_close", "Volume": "volume"})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.rename_axis("date").reset_index()
    return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]


def _fetch_fred(series_id: str, start: str, end: str) -> pd.DataFrame:
    import pandas_datareader.data as web

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = web.DataReader(series_id, "fred", start, end)
    if df is None or df.empty:
        raise EmptyDownloadError(f"no rows for {series_id}")
    # FRED publishes holidays as blank rows; keep them, since raw means faithful.
    out = df.rename(columns={series_id: "value"}).rename_axis("date").reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out[["date", "value"]]


class RawCache:
    def __init__(self, root: Path = config.RAW_CACHE_DIR):
        self.root = root
        self.manifest_path = root / "manifest.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest: dict = (json.loads(self.manifest_path.read_text())
                               if self.manifest_path.exists() else {})

    def path(self, source: str, key: str) -> Path:
        return self.root / source / f"{key}.parquet"

    def has(self, source: str, key: str) -> bool:
        return self.path(source, key).exists() and f"{source}/{key}" in self.manifest

    def read(self, source: str, key: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(source, key))

    def write(self, source: str, key: str, df: pd.DataFrame) -> None:
        path = self.path(source, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)  # atomic on the same filesystem
        self.manifest[f"{source}/{key}"] = {
            "rows": int(len(df)),
            "first_date": str(df["date"].min().date()),
            "last_date": str(df["date"].max().date()),
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        self._save_manifest()

    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
        tmp.replace(self.manifest_path)


FETCHERS = {"yfinance": _fetch_yfinance, "fred": _fetch_fred}


def download_all(cache: RawCache, refresh: bool = False,
                 fetchers: dict | None = None) -> dict[str, list[str]]:
    """Fill the cache. Returns {"downloaded": [...], "cached": [...], "failed": [...]}.

    One failure does not abort the batch. A rerun retries only what is missing.
    """
    fetchers = fetchers or FETCHERS
    jobs = [("yfinance", t) for t in config.UNIVERSE] + [("fred", s) for s in config.FRED_SERIES]
    result: dict[str, list[str]] = {"downloaded": [], "cached": [], "failed": []}
    for source, key in jobs:
        label = f"{source}/{key}"
        if cache.has(source, key) and not refresh:
            result["cached"].append(label)
            continue
        try:
            df = with_retry(lambda: fetchers[source](key, config.START_DATE, config.END_DATE), label)
            cache.write(source, key, df)
            result["downloaded"].append(label)
            log.info("%s: %d rows", label, len(df))
        except Exception as exc:
            log.error("%s: giving up after retries: %s", label, exc)
            result["failed"].append(label)
    return result
