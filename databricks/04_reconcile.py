# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Reconcile: every handoff must add up, and the output must match the original pipeline
# MAGIC
# MAGIC | Stage | What is compared |
# MAGIC |---|---|
# MAGIC | manifest → bronze | Rows per table, per ticker, and per FRED series vs. the export manifest |
# MAGIC | bronze → silver | `silver.prices` = included raw rows + inserted missing sessions (total and per ticker); macro rows unchanged |
# MAGIC | silver → gold | `gold.panel` = silver − hard-rule exclusions (total and per ticker); no duplicate (date, ticker); one macro row per NYSE session |
# MAGIC | vs original DuckDB | Row counts of `clean.prices` / `analysis.panel` / `clean.macro_daily`; each rule's total; every (ticker, rule) cell of `meta.rule_counts` |
# MAGIC | vs original values | **Every** row of `gold.panel` joined to the original `analysis.panel` on (date, ticker): returns, prices, volume, macro values, flags. A random sample is shown side by side. |
# MAGIC
# MAGIC Results are **appended** to `<catalog>.ops.reconciliation_runs`, so the history survives across runs. If any check fails, the notebook raises after writing them, and the Job run fails.
# MAGIC
# MAGIC The last section is a short Delta time-travel example.

# COMMAND ----------

import json
import uuid
from datetime import datetime, timezone

from pyspark.sql import functions as F

import pipeline_lib as pl

dbutils.widgets.text("catalog", "xasset", "Catalog")
dbutils.widgets.text("job_run_id", "", "Job run id (set to {{job.run_id}} in the Job)")
CATALOG = pl.safe_identifier(dbutils.widgets.get("catalog"))
JOB_RUN_ID = dbutils.widgets.get("job_run_id") or None
VOLUME_DIR = f"/Volumes/{CATALOG}/landing/files"
BRONZE, SILVER, GOLD, OPS = (f"{CATALOG}.{s}" for s in ("bronze", "silver", "gold", "ops"))

with open(f"{VOLUME_DIR}/manifest.json") as f:
    manifest = json.load(f)

checks: list[dict] = []


def add(stage, what, expected, actual):
    checks.append(pl.check(stage, what, expected, actual))


def rows(table, where=None):
    df = spark.table(table)
    return (df.where(where) if where else df).count()


def counts_by(table, key, where=None):
    df = spark.table(table)
    df = df.where(where) if where else df
    return {r[0]: r[1] for r in df.groupBy(key).count().collect()}

# COMMAND ----------

# MAGIC %md ## A. Export manifest → bronze

# COMMAND ----------

for table, file_name in pl.BRONZE_FILES.items():
    add("manifest -> bronze", f"bronze.{table} rows", manifest["files"][file_name]["rows"], rows(f"{BRONZE}.{table}"))

bronze_by_ticker = counts_by(f"{BRONZE}.prices", "ticker")
for t, n in sorted(manifest["raw_prices_rows_by_ticker"].items()):
    add("manifest -> bronze", f"{t}: bronze.prices rows", n, bronze_by_ticker.get(t, 0))
add("manifest -> bronze", "no tickers in bronze that are absent from the manifest", 0,
    len(set(bronze_by_ticker) - set(manifest["raw_prices_rows_by_ticker"])))

bronze_by_series = counts_by(f"{BRONZE}.macro", "series_id")
for s, n in sorted(manifest["raw_macro_rows_by_series"].items()):
    add("manifest -> bronze", f"{s}: bronze.macro rows", n, bronze_by_series.get(s, 0))

# COMMAND ----------

# MAGIC %md ## B. Bronze → silver

# COMMAND ----------

included = [r.ticker for r in spark.table(f"{SILVER}.universe").where("status LIKE 'included%'").collect()]
silver_by_ticker = counts_by(f"{SILVER}.prices", "ticker")
inserted_by_ticker = counts_by(f"{SILVER}.prices", "ticker", "flag_missing_session")

raw_included = sum(bronze_by_ticker.get(t, 0) for t in included)
inserted = sum(inserted_by_ticker.values())
silver_rows = rows(f"{SILVER}.prices")
add("bronze -> silver", "silver.prices rows = included bronze rows + inserted missing sessions",
    raw_included + inserted, silver_rows)
for t in included:
    add("bronze -> silver", f"{t}: bronze + inserted = silver",
        bronze_by_ticker.get(t, 0) + inserted_by_ticker.get(t, 0), silver_by_ticker.get(t, 0))
add("bronze -> silver", "silver.macro rows = bronze.macro rows", rows(f"{BRONZE}.macro"), rows(f"{SILVER}.macro"))
add("bronze -> silver", "data_quality_summary covers all 10 rules", len(pl.RULES),
    rows(f"{SILVER}.data_quality_summary"))

# COMMAND ----------

# MAGIC %md ## C. Silver → gold

# COMMAND ----------

excl = pl.exclusion_sql()
excluded_by_ticker = counts_by(f"{SILVER}.prices", "ticker", excl)
panel_rows = rows(f"{GOLD}.panel")
panel_by_ticker = counts_by(f"{GOLD}.panel", "ticker")

add("silver -> gold", "gold.panel rows = silver rows - hard-rule exclusions",
    silver_rows - sum(excluded_by_ticker.values()), panel_rows)
for t in included:
    add("silver -> gold", f"{t}: silver - excluded = panel",
        silver_by_ticker.get(t, 0) - excluded_by_ticker.get(t, 0), panel_by_ticker.get(t, 0))
add("silver -> gold", "no duplicate (date, ticker) in gold.panel", 0,
    spark.sql(f"SELECT COUNT(*) - COUNT(DISTINCT date, ticker) FROM {GOLD}.panel").first()[0])
add("silver -> gold", "gold.macro_daily rows = NYSE sessions",
    rows(f"{BRONZE}.ref_nyse_sessions"), rows(f"{GOLD}.macro_daily"))
add("silver -> gold", "gold.daily_returns rows = gold.panel rows", panel_rows, rows(f"{GOLD}.daily_returns"))

# COMMAND ----------

# MAGIC %md ## D. Against the original DuckDB pipeline: row counts and rule counts

# COMMAND ----------

orig = manifest["table_rows"]
add("vs original", "silver.prices rows = original clean.prices", orig["clean.prices"], silver_rows)
add("vs original", "gold.panel rows = original analysis.panel", orig["analysis.panel"], panel_rows)
add("vs original", "gold.macro_daily rows = original clean.macro_daily",
    orig["clean.macro_daily"], rows(f"{GOLD}.macro_daily"))

dq = {r.rule_name: r.rows_flagged for r in spark.table(f"{SILVER}.data_quality_summary").collect()}
for rule in pl.FLAG_COLUMNS:
    add("vs original", f"{rule}: rows flagged", manifest["original_rule_totals"][rule], dq.get(rule, -1))

# Every (ticker, rule) cell of meta.rule_counts, plus the per-ticker row count.
expected_counts = spark.read.parquet(f"{VOLUME_DIR}/expected_rule_counts.parquet")
mine = spark.table(f"{SILVER}.rule_counts_by_ticker")
cells = ["rows", *pl.FLAG_COLUMNS]
cmp = expected_counts.alias("e").join(mine.alias("m"), "ticker", "full_outer")
mismatched_cells = cmp.select(sum(
    F.when(F.col(f"e.{c}").eqNullSafe(F.col(f"m.{c}")), 0).otherwise(1) for c in cells
).alias("n")).agg(F.sum("n")).first()[0] or 0
add("vs original", f"(ticker, rule) cells that differ from meta.rule_counts ({len(cells)} per ticker)",
    0, mismatched_cells)

# COMMAND ----------

# MAGIC %md ## E. Against the original values: every gold.panel row vs. the original analysis.panel

# COMMAND ----------

def num(col, prefix):
    # NaN and NULL both mean "missing" in the original (pandas). Compare them as NULL.
    return F.nanvl(F.col(col).cast("double"), F.lit(None).cast("double")).alias(f"{prefix}_{col}")


gold = spark.table(f"{GOLD}.panel")
ref = spark.read.parquet(f"{VOLUME_DIR}/expected_analysis_panel.parquet").withColumn("date", F.col("date").cast("date"))
g = gold.select("date", "ticker", *[num(c, "g") for c in pl.PANEL_NUMERIC_COLUMNS],
                *[F.col(c).alias(f"g_{c}") for c in pl.PANEL_EXACT_COLUMNS], F.lit(True).alias("in_gold"))
r = ref.select("date", "ticker", *[num(c, "r") for c in pl.PANEL_NUMERIC_COLUMNS],
               *[F.col(c).alias(f"r_{c}") for c in pl.PANEL_EXACT_COLUMNS], F.lit(True).alias("in_original"))
j = g.join(r, ["date", "ticker"], "full_outer")


def differs_numeric(c):
    a, b = F.col(f"g_{c}"), F.col(f"r_{c}")
    tol = F.lit(pl.VALUE_TOLERANCE) * (1 + F.abs(b))
    return (F.when(a.isNull() & b.isNull(), False)
             .when(a.isNull() | b.isNull(), True)
             .otherwise(F.abs(a - b) > tol))


both = F.col("in_gold") & F.col("in_original")
stats = j.agg(
    F.sum(F.col("in_original").isNull().cast("int")).alias("only_in_gold"),
    F.sum(F.col("in_gold").isNull().cast("int")).alias("only_in_original"),
    F.sum(both.cast("int")).alias("rows_compared"),
    *[F.sum((both & differs_numeric(c)).cast("int")).alias(f"diff_{c}") for c in pl.PANEL_NUMERIC_COLUMNS],
    *[F.sum((both & ~F.col(f"g_{c}").eqNullSafe(F.col(f"r_{c}"))).cast("int")).alias(f"diff_{c}")
      for c in pl.PANEL_EXACT_COLUMNS],
    F.max(F.abs(F.col("g_daily_return") - F.col("r_daily_return"))).alias("max_abs_return_diff"),
).first()

add("vs original values", "(date, ticker) only in gold.panel", 0, stats["only_in_gold"] or 0)
add("vs original values", "(date, ticker) only in original analysis.panel", 0, stats["only_in_original"] or 0)
add("vs original values", "rows compared = original analysis.panel rows", orig["analysis.panel"],
    stats["rows_compared"] or 0)
for c in pl.PANEL_NUMERIC_COLUMNS + pl.PANEL_EXACT_COLUMNS:
    add("vs original values", f"rows where {c} differs", 0, stats[f"diff_{c}"] or 0)
print(f"rows compared: {stats['rows_compared'] or 0:,}; max |daily_return diff|: {stats['max_abs_return_diff']}")

# COMMAND ----------

# A random sample of returns side by side, for eyeballing. The checks above cover every row.
display(j.where(both & F.col("g_daily_return").isNotNull())
         .select("date", "ticker", "g_daily_return", "r_daily_return",
                 (F.col("g_daily_return") - F.col("r_daily_return")).alias("diff"),
                 "g_dq_soft_flags", "r_dq_soft_flags")
         .orderBy(F.rand(seed=42)).limit(20))

# COMMAND ----------

# MAGIC %md ## Result: write every check, then fail loudly if any did not pass

# COMMAND ----------

run_id, run_ts = str(uuid.uuid4()), datetime.now(timezone.utc)
results = (spark.createDataFrame(checks, schema=pl.CHECK_SCHEMA)
           .withColumn("run_id", F.lit(run_id))
           .withColumn("job_run_id", F.lit(JOB_RUN_ID).cast("string"))
           .withColumn("run_ts", F.lit(run_ts))
           .withColumn("feed_exported_at", F.lit(manifest["exported_at"])))
results.write.mode("append").option("mergeSchema", "true").saveAsTable(f"{OPS}.reconciliation_runs")

n_failed = len(pl.failed(checks))
print(f"run {run_id}: {len(checks) - n_failed} of {len(checks)} checks passed")
display(results.orderBy("ok", "stage", "check"))
pl.assert_all_passed(checks)   # raises -> the Job task (and run) fails

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta time travel
# MAGIC
# MAGIC **1. What changed in `gold.panel` between the last two pipeline runs?** Every run overwrites the table, which writes a new Delta version. The feed is a pinned snapshot, so a rerun should change **0 rows**, which is a reproducibility check in itself.

# COMMAND ----------

history = spark.sql(f"DESCRIBE HISTORY {GOLD}.panel")
display(history.select("version", "timestamp", "operation", "operationMetrics").orderBy(F.desc("version")))

# Only versions that rewrote the data count as a "run". Metadata-only versions
# (e.g. SET TBLPROPERTIES from a table comment) are skipped.
data_writes = history.where("operation LIKE '%TABLE AS SELECT' OR operation = 'WRITE'")
versions = sorted(r.version for r in data_writes.select("version").collect())
if len(versions) < 2:
    print("Only one version so far. Run the job again to compare two runs.")
else:
    prev, cur = versions[-2], versions[-1]
    before = spark.sql(f"SELECT * FROM {GOLD}.panel VERSION AS OF {prev}")
    after = spark.sql(f"SELECT * FROM {GOLD}.panel VERSION AS OF {cur}")
    print(f"v{prev} -> v{cur}: rows {before.count():,} -> {after.count():,}; "
          f"rows removed {before.exceptAll(after).count():,}; rows added {after.exceptAll(before).count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC **2. A vendor restatement on a scratch table.** Yahoo recomputes `Adj Close` whenever a new dividend is paid (see DECISIONS.md). This simulates that: copy a few TLT rows, change one adjusted close, query the earlier version to see exactly what changed, then `RESTORE`. Production tables are not touched.

# COMMAND ----------

DEMO = f"{OPS}.time_travel_demo"
spark.sql(f"""CREATE OR REPLACE TABLE {DEMO} AS
              SELECT date, ticker, adj_close, daily_return FROM {GOLD}.panel
              WHERE ticker = 'TLT' ORDER BY date DESC LIMIT 5""")


def latest_version(table):
    return spark.sql(f"DESCRIBE HISTORY {table}").agg(F.max("version")).first()[0]


v_before = latest_version(DEMO)
restated_date = spark.sql(f"SELECT MAX(date) FROM {DEMO}").first()[0]
spark.sql(f"UPDATE {DEMO} SET adj_close = adj_close * 1.01 WHERE date = DATE'{restated_date}'")
v_after = latest_version(DEMO)

before = spark.sql(f"SELECT * FROM {DEMO} VERSION AS OF {v_before}")
after = spark.sql(f"SELECT * FROM {DEMO} VERSION AS OF {v_after}")
display(before.exceptAll(after).withColumn("version", F.lit(v_before))
        .unionByName(after.exceptAll(before).withColumn("version", F.lit(v_after))))
display(spark.sql(f"DESCRIBE HISTORY {DEMO}").select("version", "operation", "operationMetrics"))

spark.sql(f"RESTORE TABLE {DEMO} TO VERSION AS OF {v_before}")
assert spark.table(DEMO).exceptAll(before).count() == 0, "RESTORE did not bring back the earlier version"
print(f"Restored {DEMO} to version {v_before}")
