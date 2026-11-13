# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze: land the raw feeds as Delta tables, unchanged
# MAGIC
# MAGIC Input: the files `export_for_databricks.py` wrote, uploaded to `/Volumes/<catalog>/landing/files/`.
# MAGIC
# MAGIC 1. **Check the feed first.** The manifest's config must match `pipeline_lib.py`, every file must be present, and every SHA-256 must match. A partial or corrupted upload stops here, before any table changes.
# MAGIC 2. **Load each file as-is** into `<catalog>.bronze.*`. No cleaning, no casting. The only additions are lineage columns: `_source_file`, `_source_file_modified`, `_ingested_at`, `_feed_exported_at`.
# MAGIC 3. **Check the row counts** of every bronze table against the manifest.
# MAGIC
# MAGIC Each load is a full overwrite of a static snapshot, so reruns are idempotent, and Delta keeps every previous version.

# COMMAND ----------

import json
from datetime import datetime, timezone

from pyspark.sql import functions as F

import pipeline_lib as pl   # workspace file next to this notebook

dbutils.widgets.text("catalog", "xasset", "Catalog")
CATALOG = pl.safe_identifier(dbutils.widgets.get("catalog"))
VOLUME_DIR = f"/Volumes/{CATALOG}/landing/files"
BRONZE = f"{CATALOG}.bronze"

for schema in ("bronze", "silver", "gold", "ops"):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
print("catalog:", CATALOG, "| landing volume:", VOLUME_DIR)

# COMMAND ----------

# MAGIC %md ## 1. Check the feed before loading anything

# COMMAND ----------

with open(f"{VOLUME_DIR}/manifest.json") as f:
    manifest = json.load(f)

drift = pl.config_drift(manifest["pipeline_config"])
if drift:
    raise pl.ReconciliationError("Manifest config differs from pipeline_lib.py:\n" + "\n".join(drift))

feed_checks = pl.verify_files(manifest, VOLUME_DIR)
display(spark.createDataFrame(feed_checks, schema=pl.CHECK_SCHEMA))
pl.assert_all_passed(feed_checks)
print(f"Feed exported at {manifest['exported_at']}: {len(manifest['files'])} files verified")

# COMMAND ----------

# MAGIC %md ## 2. Load each raw file into a bronze Delta table

# COMMAND ----------

ingested_at = datetime.now(timezone.utc)   # one timestamp for the whole batch
load_checks = []

for table, file_name in pl.BRONZE_FILES.items():
    df = (spark.read.parquet(f"{VOLUME_DIR}/{file_name}")
          .select("*",
                  F.col("_metadata.file_path").alias("_source_file"),
                  F.col("_metadata.file_modification_time").alias("_source_file_modified"))
          .withColumn("_ingested_at", F.lit(ingested_at))
          .withColumn("_feed_exported_at", F.lit(manifest["exported_at"])))

    missing = pl.missing_columns(table, df.columns)
    if missing:
        raise pl.ReconciliationError(f"{file_name} is missing required columns {sorted(missing)}")

    target = f"{BRONZE}.{table}"
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(target)
    rows = spark.table(target).count()
    load_checks.append(pl.check("manifest -> bronze", f"{table} rows", manifest["files"][file_name]["rows"], rows))
    print(f"{target:40s} {rows:>9,d} rows")

spark.sql(f"COMMENT ON TABLE {BRONZE}.prices IS "
          "'Raw yfinance daily bars exactly as exported from DuckDB raw.prices, plus lineage columns.'")
spark.sql(f"COMMENT ON TABLE {BRONZE}.macro IS "
          "'Raw FRED observations exactly as exported from DuckDB raw.macro, plus lineage columns.'")

# COMMAND ----------

# MAGIC %md ## 3. Row counts match the manifest

# COMMAND ----------

display(spark.createDataFrame(load_checks, schema=pl.CHECK_SCHEMA))
pl.assert_all_passed(load_checks)

# COMMAND ----------

display(spark.sql(f"""
    SELECT ticker, COUNT(*) AS rows, MIN(date) AS first_date, MAX(date) AS last_date,
           MAX(_ingested_at) AS ingested_at, MAX(_source_file) AS source_file
    FROM {BRONZE}.prices GROUP BY ticker ORDER BY ticker
"""))
