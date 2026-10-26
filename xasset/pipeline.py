"""End-to-end pipeline:  python -m xasset.pipeline [--refresh]

Stages: API -> cache -> raw tables -> clean tables -> analysis view -> parquet export.
Row counts are reconciled between every pair of stages. Any mismatch fails the
run, after the report is written so the mismatch can be inspected.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from . import config
from .curate import align_macro, macro_checks
from .ingest import RawCache, download_all
from .quality import apply_rules, excluded_mask
from .report import write_report
from .warehouse import scalar, connect, create_analysis_view, replace_table, rule_counts

log = logging.getLogger("xasset")


class ReconciliationError(RuntimeError):
    pass


def load_raw(cache: RawCache) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices, macro = [], []
    for ticker in config.UNIVERSE:
        df = cache.read("yfinance", ticker).assign(ticker=ticker, source="yfinance")
        df["downloaded_at"] = cache.manifest[f"yfinance/{ticker}"]["downloaded_at"]
        prices.append(df)
    for sid in config.FRED_SERIES:
        df = cache.read("fred", sid).assign(series_id=sid, source="fred")
        df["downloaded_at"] = cache.manifest[f"fred/{sid}"]["downloaded_at"]
        macro.append(df)
    prices = pd.concat(prices, ignore_index=True)
    macro = pd.concat(macro, ignore_index=True)
    # One datetime resolution everywhere; pandas refuses to merge ns with us keys.
    prices["date"] = prices["date"].astype("datetime64[ns]")
    macro["date"] = macro["date"].astype("datetime64[ns]")
    prices.insert(0, "raw_row_id", range(len(prices)))
    return prices, macro


def run(refresh: bool = False) -> None:
    # Stage 1: ingest
    cache = RawCache()
    dl = download_all(cache, refresh=refresh)
    log.info("downloaded=%d cached=%d failed=%d", len(dl["downloaded"]), len(dl["cached"]), len(dl["failed"]))
    if dl["failed"]:
        raise SystemExit(f"Downloads failed: {dl['failed']}. Rerun to resume; cached items are skipped.")

    recon: list[dict] = []

    def check(stage: str, what: str, expected, actual) -> None:
        recon.append({"stage": stage, "check": what, "expected": int(expected),
                      "actual": int(actual), "ok": int(expected) == int(actual)})

    # Stage 2: cache -> raw tables
    raw_prices, raw_macro = load_raw(cache)
    con = connect()
    replace_table(con, "raw.prices", raw_prices)
    replace_table(con, "raw.macro", raw_macro)
    manifest_price_rows = sum(cache.manifest[f"yfinance/{t}"]["rows"] for t in config.UNIVERSE)
    manifest_macro_rows = sum(cache.manifest[f"fred/{s}"]["rows"] for s in config.FRED_SERIES)
    check("cache -> raw", "price rows (manifest vs raw.prices)", manifest_price_rows,
          scalar(con, "SELECT COUNT(*) FROM raw.prices"))
    check("cache -> raw", "macro rows (manifest vs raw.macro)", manifest_macro_rows,
          scalar(con, "SELECT COUNT(*) FROM raw.macro"))

    # Stage 3: raw -> clean
    res = apply_rules(raw_prices)
    clean, universe, coverage = res["clean"], res["universe"], res["coverage"]
    replace_table(con, "clean.prices", clean)
    replace_table(con, "clean.universe", universe)
    replace_table(con, "clean.calendar_coverage", coverage)

    included = universe.loc[universe["status"].str.startswith("included"), "ticker"]
    raw_in = raw_prices["ticker"].isin(included)
    inserted = int(clean["flag_missing_session"].sum())
    check("raw -> clean", "clean rows = included raw rows + inserted missing sessions",
          int(raw_in.sum()) + inserted, len(clean))

    macro_daily = align_macro(raw_macro)
    replace_table(con, "clean.macro_daily", macro_daily)
    n_sessions = len(macro_daily)
    check("raw -> clean", "macro_daily rows = NYSE sessions", n_sessions,
          scalar(con, "SELECT COUNT(DISTINCT date) FROM clean.macro_daily"))

    # Stage 4: clean -> analysis view
    create_analysis_view(con)
    excl = excluded_mask(clean)
    view_rows = scalar(con, "SELECT COUNT(*) FROM analysis.panel")
    check("clean -> analysis", "view rows = clean rows - hard-rule exclusions (pandas vs SQL)",
          len(clean) - int(excl.sum()), view_rows)
    check("clean -> analysis", "no duplicate (date, ticker) in view", 0,
          scalar(con, "SELECT COUNT(*) - COUNT(DISTINCT (date, ticker)) FROM analysis.panel"))

    # Stage 5: export
    config.CURATED_DIR.mkdir(parents=True, exist_ok=True)
    panel = con.execute("SELECT * FROM analysis.panel").df()
    panel_path = config.CURATED_DIR / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    check("analysis -> export", "parquet rows = view rows", view_rows, len(pd.read_parquet(panel_path)))

    # Per-ticker lineage table
    per_ticker = pd.DataFrame({
        "cache_rows": pd.Series({t: cache.manifest[f"yfinance/{t}"]["rows"] for t in config.UNIVERSE}),
        "raw_rows": raw_prices.groupby("ticker").size(),
        "inserted_missing": clean.groupby("ticker")["flag_missing_session"].sum(),
        "clean_rows": clean.groupby("ticker").size(),
        "excluded_hard": excl.groupby(clean["ticker"]).sum(),
        "panel_rows": panel.groupby("ticker").size(),
    }).reindex(list(config.UNIVERSE)).fillna(0).astype(int).rename_axis("ticker").reset_index()
    for row in per_ticker.itertuples():
        if row.clean_rows:  # included tickers only
            check("per ticker", f"{row.ticker}: raw + inserted = clean",
                  row.raw_rows + row.inserted_missing, row.clean_rows)
            check("per ticker", f"{row.ticker}: clean - excluded = panel",
                  row.clean_rows - row.excluded_hard, row.panel_rows)

    recon_df = pd.DataFrame(recon)
    replace_table(con, "meta.reconciliation", recon_df)
    counts = rule_counts(con)
    replace_table(con, "meta.rule_counts", counts)

    path = write_report(universe=universe, counts=counts, recon=recon_df,
                        per_ticker=per_ticker, clean=clean, coverage=coverage,
                        macro=macro_checks(raw_macro), macro_daily=macro_daily, panel=panel)
    con.close()
    log.info("report: %s", path)
    failed = recon_df[~recon_df["ok"]]
    if not failed.empty:
        raise ReconciliationError(f"{len(failed)} reconciliation checks failed:\n{failed.to_string()}")
    log.info("all %d reconciliation checks passed; panel rows=%d", len(recon_df), len(panel))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-download everything")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(refresh=args.refresh)


if __name__ == "__main__":
    main()
