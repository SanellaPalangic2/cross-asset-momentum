# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold: the analysis panel, daily returns, and the backtest's inputs
# MAGIC
# MAGIC | Gold table | Built from | Original |
# MAGIC |---|---|---|
# MAGIC | `macro_daily` | `silver.macro` → `pipeline_lib.align_macro` (lag 1 session, forward-fill capped at 5) | `curate.align_macro` → `clean.macro_daily` |
# MAGIC | `panel` | `silver.prices` ⟕ `macro_daily`, hard-rule rows removed, soft flags listed | the `analysis.panel` view |
# MAGIC | `daily_returns` | `panel` (date, ticker, adj_close, daily_return) | `Market.returns` |
# MAGIC | `risk_free_daily` | `panel` → `xasset.features.daily_rf_from_tbill`, `curve_inverted` | `Market.rf`, `Market.t10y2y` |
# MAGIC | `signal_dates` | `xasset.features.signal_dates` (last session of each month) | `features.signal_dates` |
# MAGIC | `features` | `xasset.features.build_features` (mom 12-1, 1m return, trend 200, vol 60) | `features.build_features` |
# MAGIC
# MAGIC The last three call **the original `src/xasset/features.py` functions directly** (imported from the repo's `src/` folder), so the backtest inputs come from the same code, not a copy.
# MAGIC
# MAGIC The original `analysis.panel` is a *view*, so moving a rule between hard and soft needs no rewrite. Here, the panel is rebuilt on every run (seconds at this size), and Delta history keeps every version.

# COMMAND ----------

import os
import sys

import pandas as pd

import pipeline_lib as pl

dbutils.widgets.text("catalog", "xasset", "Catalog")
CATALOG = pl.safe_identifier(dbutils.widgets.get("catalog"))
BRONZE, SILVER, GOLD = f"{CATALOG}.bronze", f"{CATALOG}.silver", f"{CATALOG}.gold"

# The repo's src/ folder sits next to databricks/ in the Git folder.
SRC = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if not os.path.isdir(os.path.join(SRC, "xasset")):
    raise FileNotFoundError(f"{SRC}/xasset not found. Run these notebooks from a Git folder of the whole repo.")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from xasset import features as xf   # noqa: E402  (needs only numpy + pandas)


def write(pdf, table, schema, date_cols=("date",)):
    sdf = spark.createDataFrame(pl.nulls_for_spark(pdf, date_cols=date_cols), schema=schema)
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD}.{table}")
    print(f"{GOLD}.{table}: {spark.table(f'{GOLD}.{table}').count():,} rows")

# COMMAND ----------

# MAGIC %md ## 1. Macro aligned to NYSE sessions (publication-lagged, capped forward fill)

# COMMAND ----------

session_dates = spark.table(f"{BRONZE}.ref_nyse_sessions").toPandas()["date"]
raw_macro = spark.table(f"{SILVER}.macro").select("series_id", "date", "value").toPandas()
macro_daily = pl.align_macro(raw_macro, session_dates)
obs_cols = [f"{c}_obs_date" for c in pl.FRED_SERIES.values()]
write(macro_daily, "macro_daily", pl.macro_daily_schema(), date_cols=["date", *obs_cols])

# COMMAND ----------

# MAGIC %md ## 2. The analysis panel: hard-rule exclusions applied, soft flags kept as `dq_soft_flags`

# COMMAND ----------

panel_sql = pl.analysis_panel_sql(f"{SILVER}.prices", f"{GOLD}.macro_daily")
print(panel_sql)
spark.sql(f"CREATE OR REPLACE TABLE {GOLD}.panel AS {panel_sql}")
spark.sql(f"COMMENT ON TABLE {GOLD}.panel IS 'Port of analysis.panel: silver.prices minus hard-rule rows "
          "(extra duplicate copies, off-calendar, missing or non-positive price), joined to macro_daily.'")

spark.sql(f"""
    CREATE OR REPLACE TABLE {GOLD}.daily_returns AS
    SELECT date, ticker, asset_class, adj_close, daily_return FROM {GOLD}.panel
""")
display(spark.sql(f"SELECT ticker, COUNT(*) AS rows, COUNT(daily_return) AS returns FROM {GOLD}.daily_returns "
                  "GROUP BY ticker ORDER BY ticker"))

# COMMAND ----------

# MAGIC %md ## 3. Backtest inputs, built with the original `xasset.features` code

# COMMAND ----------

panel = spark.table(f"{GOLD}.panel").toPandas()
panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
mkt = xf.market_from_panel(panel)

dtb3 = panel.groupby("date")["dtb3"].first().reindex(mkt.rf.index)
risk_free = pd.DataFrame({
    "date": mkt.rf.index,
    "dtb3": dtb3.to_numpy(),
    "rf_daily": mkt.rf.to_numpy(),
    "t10y2y": mkt.t10y2y.reindex(mkt.rf.index).to_numpy(),
    "curve_inverted": xf.curve_inverted(mkt.t10y2y).reindex(mkt.rf.index).to_numpy(),
})
write(risk_free, "risk_free_daily",
      "date date, dtb3 double, rf_daily double, t10y2y double, curve_inverted boolean")

write(pd.DataFrame({"signal_date": xf.signal_dates(mkt.prices.index)}), "signal_dates",
      "signal_date date", date_cols=("signal_date",))

feats = xf.build_features(mkt)
features_long = pl.wide_to_long({k: v for k, v in feats.items() if isinstance(v, pd.DataFrame)})
write(features_long, "features",
      "date date, ticker string, mom_12_1 double, ret_1m double, trend_200 double, vol_60 double")

# COMMAND ----------

display(spark.sql(f"""
    SELECT f.date, f.ticker, f.mom_12_1, f.trend_200, f.vol_60, r.daily_return
    FROM {GOLD}.features f JOIN {GOLD}.signal_dates s ON f.date = s.signal_date
    JOIN {GOLD}.daily_returns r ON r.date = f.date AND r.ticker = f.ticker
    ORDER BY f.date DESC, f.ticker LIMIT 48
"""))
