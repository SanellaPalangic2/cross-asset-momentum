"""Export the raw feeds from the local DuckDB warehouse to Parquet files for Databricks.

    python -m xasset.pipeline                      # first: builds data/warehouse.duckdb
    python databricks/export_for_databricks.py     # then: writes data/databricks_export/

Options: --db PATH (default: config.DB_PATH), --out DIR (default: data/databricks_export).

Files written (upload all of them to the Databricks volume):

  raw_prices.parquet               raw.prices  - inputs to the Databricks pipeline
  raw_macro.parquet                raw.macro
  ref_nyse_sessions.parquet        the NYSE session dates the original pipeline used
  ref_universe.parquet             ticker -> asset_class, in config order
  expected_analysis_panel.parquet  analysis.panel   - only used by 04_reconcile, to compare
  expected_rule_counts.parquet     meta.rule_counts   against the original output
  manifest.json                    row counts per table and file, SHA-256 per file,
                                   and a snapshot of the pipeline config

Notes:
- The database is opened read-only, and the existing pipeline's code and data are not modified.
- Dates are written as Parquet DATE. DuckDB holds them as nanosecond timestamps,
  which Spark can't read. Every value is checked to be midnight first, so the cast loses nothing.
- The NYSE sessions come from clean.macro_daily (one row per session by
  construction), so Databricks uses exactly the calendar the original used and
  doesn't need exchange_calendars installed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))   # the existing xasset package
sys.path.insert(0, str(HERE))                  # pipeline_lib

import duckdb  # noqa: E402

import pipeline_lib  # noqa: E402
from xasset import config  # noqa: E402

# (file name, source, role, query). "input" files feed bronze; "expected" files are
# the original pipeline's outputs, used only for reconciliation.
EXPORTS = [
    ("raw_prices.parquet", "raw.prices", "input",
     "SELECT * REPLACE (CAST(date AS DATE) AS date) FROM raw.prices ORDER BY raw_row_id"),
    ("raw_macro.parquet", "raw.macro", "input",
     "SELECT * REPLACE (CAST(date AS DATE) AS date) FROM raw.macro ORDER BY rowid"),
    ("ref_nyse_sessions.parquet", "clean.macro_daily", "input",
     "SELECT CAST(date AS DATE) AS date FROM clean.macro_daily ORDER BY date"),
    ("ref_universe.parquet", "config.UNIVERSE", "input",
     "SELECT * FROM _universe ORDER BY position"),
    ("expected_analysis_panel.parquet", "analysis.panel", "expected",
     "SELECT * REPLACE (CAST(date AS DATE) AS date) FROM analysis.panel ORDER BY date, ticker"),
    ("expected_rule_counts.parquet", "meta.rule_counts", "expected",
     "SELECT * FROM meta.rule_counts ORDER BY ticker"),
]
TABLES_WITH_DATES = ["raw.prices", "raw.macro", "clean.macro_daily", "analysis.panel"]


def scalar(con, sql: str):
    return con.execute(sql).fetchall()[0][0]


def universe_values_sql(universe: dict[str, str] | None = None) -> str:
    """config.UNIVERSE as an inline VALUES table (ticker, asset_class, position), in config order."""
    universe = config.UNIVERSE if universe is None else universe
    for t, a in universe.items():
        pipeline_lib.safe_identifier(t), pipeline_lib.safe_identifier(a)
    rows = ", ".join(f"('{t}', '{a}', {i})" for i, (t, a) in enumerate(universe.items()))
    return f"(SELECT * FROM (VALUES {rows}) AS u(ticker, asset_class, position))"


def export(db_path: Path, out_dir: Path) -> dict:
    drift = pipeline_lib.config_drift(pipeline_lib.config_snapshot(config))
    if drift:
        raise SystemExit("databricks/pipeline_lib.py no longer matches src/xasset/config.py:\n  "
                         + "\n  ".join(drift))
    if not db_path.exists():
        raise SystemExit(f"{db_path} not found. Run `python -m xasset.pipeline` first.")

    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for t in TABLES_WITH_DATES:
            bad = scalar(con, f"SELECT COUNT(*) FROM {t} "
                              f"WHERE CAST(CAST(date AS DATE) AS TIMESTAMP) <> CAST(date AS TIMESTAMP)")
            if bad:
                raise SystemExit(f"{t}: {bad} dates are not at midnight; refusing to cast to DATE")

        files = {}
        for name, source, role, query in EXPORTS:
            query = query.replace("_universe", universe_values_sql())
            final = out_dir / name
            tmp = out_dir / f"{name}.tmp"
            con.execute(f"COPY ({query}) TO '{tmp.as_posix()}' (FORMAT parquet)")
            tmp.replace(final)   # atomic: a half-written file never has the final name
            rows_written = scalar(con, f"SELECT COUNT(*) FROM read_parquet('{final.as_posix()}')")
            rows_source = (len(config.UNIVERSE) if source == "config.UNIVERSE"
                           else scalar(con, f"SELECT COUNT(*) FROM {source}"))
            if rows_written != rows_source:
                raise SystemExit(f"{name}: wrote {rows_written} rows but {source} has {rows_source}")
            files[name] = {"source": source, "role": role, "rows": int(rows_written),
                           "bytes": final.stat().st_size, "sha256": pipeline_lib.sha256_file(final)}
            print(f"  {name:34s} {rows_written:>8,d} rows  <- {source}")

        table_rows = {t: int(scalar(con, f"SELECT COUNT(*) FROM {t}"))
                      for t in ["raw.prices", "raw.macro", "clean.prices", "clean.macro_daily",
                                "analysis.panel"]}
        by_ticker = dict(con.execute("SELECT ticker, COUNT(*) FROM raw.prices GROUP BY ticker "
                                     "ORDER BY ticker").fetchall())
        by_series = dict(con.execute("SELECT series_id, COUNT(*) FROM raw.macro GROUP BY series_id "
                                     "ORDER BY series_id").fetchall())
        sums = ", ".join(f"SUM({c}::INT)::BIGINT" for c in pipeline_lib.FLAG_COLUMNS)
        totals = con.execute(f"SELECT {sums} FROM clean.prices").fetchall()[0]
        recon = con.execute("SELECT COUNT(*), SUM(ok::INT) FROM meta.reconciliation").fetchall()[0]
    finally:
        con.close()

    manifest = {
        "manifest_version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_db": db_path.name,
        "pipeline_config": pipeline_lib.config_snapshot(config),
        "files": files,
        "table_rows": table_rows,
        "raw_prices_rows_by_ticker": {k: int(v) for k, v in by_ticker.items()},
        "raw_macro_rows_by_series": {k: int(v) for k, v in by_series.items()},
        "original_rule_totals": {c: int(v or 0) for c, v in zip(pipeline_lib.FLAG_COLUMNS, totals)},
        "original_reconciliation": {"checks": int(recon[0]), "passed": int(recon[1] or 0)},
    }
    tmp = out_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(out_dir / "manifest.json")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=config.DB_PATH)
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "databricks_export")
    args = ap.parse_args()
    print(f"Exporting {args.db} -> {args.out}")
    m = export(args.db, args.out)
    print(f"  {'manifest.json':34s} {len(m['files'])} files, table rows: {m['table_rows']}")
    print("Upload every file in that folder to your Databricks volume.")


if __name__ == "__main__":
    main()
