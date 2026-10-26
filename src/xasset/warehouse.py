"""DuckDB storage in three layers: raw -> clean -> analysis.

raw.*      exactly what the cache holds, plus lineage columns
clean.*    full grid with one boolean column per DQ rule; nothing deleted
analysis.* a VIEW over clean, filtered by config.HARD_RULES. It is a view, not a
           copy, so changing a rule needs no data rewrite.
meta.*     row-count reconciliation and per-rule counts
"""
from __future__ import annotations

import duckdb
import pandas as pd

from . import config
from .quality import FLAG_COLUMNS


def connect(path=None) -> duckdb.DuckDBPyConnection:
    path = path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    for schema in ("raw", "clean", "analysis", "meta"):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    return con


def scalar(con: duckdb.DuckDBPyConnection, sql: str):
    """First value of a query. fetchall() fully drains the result (DuckDB 2.0 requires it)."""
    return con.execute(sql).fetchall()[0][0]


def replace_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    con.register("_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
    con.unregister("_df")


def exclusion_sql() -> str:
    """SQL predicate that is TRUE for rows the hard rules exclude. Mirrors quality.excluded_mask."""
    parts = [f"({r} AND dup_rank > 1)" if r == "flag_duplicate_date" else r
             for r in config.HARD_RULES]
    return " OR ".join(parts) if parts else "FALSE"


def create_analysis_view(con: duckdb.DuckDBPyConnection) -> None:
    macro_cols = ", ".join(f"m.{c}" for c in config.FRED_SERIES.values())
    ffilled = " OR ".join(f"m.{c}_ffilled" for c in config.FRED_SERIES.values())
    soft_list = ", ".join(f"CASE WHEN p.{r} THEN '{r.removeprefix('flag_')}' END"
                          for r in config.SOFT_RULES)
    con.execute(f"""
        CREATE OR REPLACE VIEW analysis.panel AS
        SELECT
            p.date, p.ticker, p.asset_class,
            p.adj_close, p.daily_return, p.volume,
            {macro_cols},
            ({ffilled}) AS macro_ffilled,
            concat_ws('|', {soft_list}) AS dq_soft_flags
        FROM clean.prices p
        LEFT JOIN clean.macro_daily m USING (date)
        WHERE NOT ({exclusion_sql()})
        ORDER BY p.date, p.ticker
    """)


def rule_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sums = ", ".join(f"SUM({c}::INT)::BIGINT AS {c}" for c in FLAG_COLUMNS)
    return con.execute(f"""
        SELECT ticker, asset_class, COUNT(*)::BIGINT AS rows, {sums}
        FROM clean.prices GROUP BY ALL ORDER BY asset_class, ticker
    """).df()
