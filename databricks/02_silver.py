# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver: typed rows, one flag column per data-quality rule, nothing deleted
# MAGIC
# MAGIC A port of `src/xasset/quality.py` (`apply_rules`) to Spark.
# MAGIC
# MAGIC | Step | How it runs here | Original |
# MAGIC |---|---|---|
# MAGIC | Cast types | Spark `cast` | pandas dtypes |
# MAGIC | Universe screen (24 rows) | `pipeline_lib.screen_universe` on the driver | `screen_universe` |
# MAGIC | Ticker × NYSE-session grid, inserted missing sessions | Spark cross join + anti join | `_build_grid` |
# MAGIC | 5 structural rules + `dup_rank` | Spark SQL predicates (`pipeline_lib.STRUCTURAL_RULE_SQL`), windows on (ticker, date) | structural block |
# MAGIC | 4 time-series rules (returns, robust z, stale runs, gaps, zero volume) | `applyInPandas`, one ticker per call, running `pipeline_lib.ticker_time_series_rules` | return-based block |
# MAGIC | Partial-coverage rule | Spark `groupBy(date)` | coverage block |
# MAGIC
# MAGIC **Dedupe uses the original's keys.** `dup_rank` is the row number within (ticker, date), ordered by `raw_row_id`. Silver keeps every copy and flags it. Gold drops copies with `dup_rank > 1`, exactly as `analysis.panel` does.
# MAGIC
# MAGIC **Expectations:** a notebook can't declare pipeline expectations. The flags here are the source of truth. `pipelines/silver_dq_expectations.py` re-declares the same 10 rules as Lakeflow expectations, in warn mode so rows are kept (optional, see README).
# MAGIC
# MAGIC Outputs in `<catalog>.silver`: `prices`, `macro`, `universe`, `calendar_coverage`, `data_quality_summary`, `rule_counts_by_ticker`, `dq_rule_catalog`.

# COMMAND ----------

from datetime import datetime, timezone

from pyspark.sql import Window
from pyspark.sql import functions as F

import pipeline_lib as pl

# Ship pipeline_lib's code with the pandas UDF, so the executors don't need to import the workspace file.
try:
    from pyspark import cloudpickle
    cloudpickle.register_pickle_by_value(pl)
except Exception as exc:  # executors can still import it from the notebook folder
    print("register_pickle_by_value unavailable:", exc)

dbutils.widgets.text("catalog", "xasset", "Catalog")
CATALOG = pl.safe_identifier(dbutils.widgets.get("catalog"))
BRONZE, SILVER = f"{CATALOG}.bronze", f"{CATALOG}.silver"
NULL_DOUBLE = F.lit(None).cast("double")


def no_nan(col):
    """pandas treats NaN as missing; Spark doesn't. Make NaN a real NULL."""
    return F.nanvl(F.col(col).cast("double"), NULL_DOUBLE)


def overwrite(df, table):
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SILVER}.{table}")
    print(f"{SILVER}.{table}: {spark.table(f'{SILVER}.{table}').count():,} rows")

# COMMAND ----------

# MAGIC %md ## 1. Cast types

# COMMAND ----------

raw = spark.table(f"{BRONZE}.prices").select(
    F.col("raw_row_id").cast("bigint"),
    F.col("ticker").cast("string"),
    F.col("date").cast("date"),
    *[F.col(c).cast("double").alias(c) for c in ["open", "high", "low", "close", "adj_close", "volume"]],
    F.col("source").cast("string"),
    F.col("downloaded_at").cast("timestamp"),
    "_source_file", "_ingested_at",
)

sessions = spark.table(f"{BRONZE}.ref_nyse_sessions").select(F.col("date").cast("date"))
# 5k rows, so a single-partition window is fine.
sessions_pos = sessions.withColumn("session_pos", F.row_number().over(Window.orderBy("date")) - 1)
first_session = sessions.agg(F.min("date")).first()[0]

macro = spark.table(f"{BRONZE}.macro").select(
    F.col("series_id").cast("string"), F.col("date").cast("date"),
    no_nan("value").alias("value"),          # FRED holiday blanks -> NULL
    F.col("source").cast("string"), F.col("downloaded_at").cast("timestamp"),
    "_source_file", "_ingested_at",
)
overwrite(macro, "macro")

# COMMAND ----------

# MAGIC %md ## 2. Universe screen (included / late start / dropped)

# COMMAND ----------

stats = (raw.where(pl.SCREEN_VALID_PRICE_SQL).groupBy("ticker")
         .agg(F.min("date").alias("first_date"), F.max("date").alias("last_date"),
              F.count(F.lit(1)).alias("raw_rows"))
         .toPandas())
ref_universe = spark.table(f"{BRONZE}.ref_universe").orderBy("position").toPandas()
universe_map = dict(zip(ref_universe["ticker"], ref_universe["asset_class"]))

universe_pdf = pl.screen_universe(stats, universe_map, first_session)
universe_pdf["raw_rows"] = universe_pdf["raw_rows"].astype("Int64")
universe = spark.createDataFrame(
    pl.nulls_for_spark(universe_pdf, date_cols=["first_date", "last_date"]),
    schema="ticker string, first_date date, last_date date, raw_rows bigint, asset_class string, status string")
overwrite(universe, "universe")
universe = spark.table(f"{SILVER}.universe")
display(universe)

# COMMAND ----------

# MAGIC %md ## 3. Ticker × NYSE-session grid: a missing day becomes an explicit row with NULL prices

# COMMAND ----------

included = universe.where(F.col("status").startswith("included")).select("ticker", "first_date")
raw_in = raw.join(included.select("ticker"), "ticker", "left_semi").withColumn("row_origin", F.lit("raw"))

grid = (included.crossJoin(sessions).where(F.col("date") >= F.col("first_date")).select("ticker", "date"))
present = raw_in.select("ticker", "date").distinct()
missing = (grid.join(present, ["ticker", "date"], "left_anti")
           .withColumn("row_origin", F.lit("inserted_missing_session")))

df = (raw_in.unionByName(missing, allowMissingColumns=True)
      .join(sessions_pos, "date", "left")
      .withColumn("is_session", F.col("session_pos").isNotNull()))

# COMMAND ----------

# MAGIC %md ## 4. Structural rules (1–5) and the usable mask

# COMMAND ----------

key = Window.partitionBy("ticker", "date")
df = (df.withColumn("dup_rank", F.row_number().over(key.orderBy(F.col("raw_row_id").asc_nulls_last())))
        .withColumn("n_rows_same_key", F.count(F.lit(1)).over(key)))
for rule, predicate in pl.STRUCTURAL_RULE_SQL.items():
    df = df.withColumn(rule, F.expr(predicate))
df = df.withColumn("is_usable", F.expr(pl.USABLE_SQL))

# COMMAND ----------

# MAGIC %md ## 5. Time-series rules (6–9) on the usable series, one ticker per pandas call

# COMMAND ----------

usable = df.where("is_usable").select(
    "ticker", "raw_row_id", F.col("date").cast("timestamp").alias("date"), "session_pos",
    "adj_close", "close", "volume")
ts = (usable.groupBy("ticker")
      .applyInPandas(pl.ticker_time_series_rules, schema=pl.TIME_SERIES_SCHEMA)
      .drop("ticker")
      .withColumn("stale_run_start", F.col("stale_run_start").cast("date")))
for c in ["daily_return", "split_only_return", "robust_z"]:
    ts = ts.withColumn(c, no_nan(c))

df = df.join(ts, "raw_row_id", "left")   # inserted rows have no raw_row_id -> NULLs, as in the original
for f in pl.TIME_SERIES_FLAGS:
    df = df.withColumn(f, F.coalesce(F.col(f), F.lit(False)))

# COMMAND ----------

# MAGIC %md ## 6. Cross-asset rule (10): did every active ticker print a usable price on this session?

# COMMAND ----------

coverage = (df.where("is_session AND dup_rank = 1").groupBy("date")
            .agg(F.count(F.lit(1)).alias("active_tickers"),
                 F.sum(F.col("is_usable").cast("int")).alias("priced_tickers"))
            .withColumn("missing_tickers", F.col("active_tickers") - F.col("priced_tickers")))
partial = coverage.where("missing_tickers > 0").select("date", F.lit(True).alias("_partial"))

df = (df.join(partial, "date", "left")
        .withColumn("flag_partial_coverage_date", F.col("is_session") & F.coalesce("_partial", F.lit(False)))
        .drop("_partial")
        .join(F.broadcast(universe.select("ticker", "asset_class")), "ticker", "left"))

# COMMAND ----------

# MAGIC %md ## 7. Write silver tables

# COMMAND ----------

SILVER_COLUMNS = [
    "ticker", "date", "asset_class", "row_origin", "raw_row_id",
    "open", "high", "low", "close", "adj_close", "volume",
    "daily_return", "split_only_return", "robust_z", "stale_run_length", "stale_run_start",
    "session_pos", "is_session", "dup_rank", "n_rows_same_key", "is_usable",
    *pl.FLAG_COLUMNS,
    "source", "downloaded_at", "_source_file", "_ingested_at",
]
overwrite(df.select(*SILVER_COLUMNS).orderBy("ticker", "date", "raw_row_id"), "prices")
overwrite(coverage.orderBy("date"), "calendar_coverage")
spark.sql(f"COMMENT ON TABLE {SILVER}.prices IS 'Full ticker x NYSE-session grid. One boolean column "
          "per DQ rule. Nothing deleted; gold applies the hard-rule exclusions.'")

# COMMAND ----------

# MAGIC %md ## 8. Data-quality summary (rows flagged per rule) and per-ticker counts

# COMMAND ----------

prices = spark.table(f"{SILVER}.prices")
computed_at = datetime.now(timezone.utc)

totals = prices.agg(
    F.count(F.lit(1)).alias("rows_total"),
    *[F.sum(F.col(r).cast("int")).alias(r) for r in pl.FLAG_COLUMNS],
    *[F.sum(F.expr(f"CASE WHEN {pl.exclusion_sql([r])} THEN 1 ELSE 0 END")).alias(f"excl_{r}")
      for r in pl.HARD_RULES],
).first()

summary_rows = [{
    "rule_name": r.name, "severity": r.severity, "kind": r.kind,
    "rows_flagged": int(totals[r.name] or 0),
    "rows_excluded_from_gold": int(totals[f"excl_{r.name}"] or 0) if r.severity == "hard" else 0,
    "rows_total": int(totals["rows_total"]),
    "checks": r.checks, "computed_at": computed_at,
} for r in pl.RULES]
summary = spark.createDataFrame(summary_rows, schema=(
    "rule_name string, severity string, kind string, rows_flagged bigint, rows_excluded_from_gold bigint, "
    "rows_total bigint, checks string, computed_at timestamp"))
overwrite(summary, "data_quality_summary")
display(spark.table(f"{SILVER}.data_quality_summary"))

# COMMAND ----------

# Same shape as the original meta.rule_counts, so 04_reconcile can compare cell by cell.
by_ticker = (prices.groupBy("ticker", "asset_class")
             .agg(F.count(F.lit(1)).alias("rows"), *[F.sum(F.col(r).cast("int")).alias(r) for r in pl.FLAG_COLUMNS])
             .orderBy("asset_class", "ticker"))
overwrite(by_ticker, "rule_counts_by_ticker")

catalog_rows = [{"rule_name": r.name, "severity": r.severity, "kind": r.kind, "checks": r.checks,
                 "flags": r.flags, "expectation": r.expectation,
                 "predicate_sql": pl.STRUCTURAL_RULE_SQL.get(r.name)} for r in pl.RULES]
overwrite(spark.createDataFrame(catalog_rows, schema=(
    "rule_name string, severity string, kind string, checks string, flags string, expectation string, "
    "predicate_sql string")), "dq_rule_catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC Rows that `flag_duplicate_date` marks with `dup_rank > 1` are the copies gold drops. Every other flagged row stays and is visible here:

# COMMAND ----------

display(prices.where(" OR ".join(pl.FLAG_COLUMNS))
        .select("ticker", "date", "row_origin", "adj_close", "daily_return", "robust_z", "dup_rank",
                *pl.FLAG_COLUMNS)
        .orderBy("date", "ticker").limit(200))
