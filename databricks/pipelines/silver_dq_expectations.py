"""OPTIONAL: the 10 silver data-quality rules as Lakeflow pipeline expectations.

This is a pipeline source file, not a notebook. It runs inside a Lakeflow (Spark)
Declarative Pipeline, where `spark` is provided. The README explains how to create the pipeline.

Why it exists alongside the flag columns:
- Five of the rules need window logic (duplicates by key, stale runs, robust z,
  gaps, cross-ticker coverage). An expectation is a per-row SQL boolean, so it
  can't compute them. 02_silver computes every flag, and that stays the source of truth.
- This pipeline re-checks those flags as expectations, so the pipeline UI's
  Data quality tab and event log report passed/failed record counts per rule.
  The failed counts should equal silver.data_quality_summary.rows_flagged.
- @dp.expect_all is WARN mode: failing rows are kept. That matches the project's
  "flag, don't delete" rule. expect_all_or_drop / expect_all_or_fail would break it.

The rules are read from <catalog>.silver.dq_rule_catalog (written by 02_silver),
so the rule list lives in one place (Databricks' "store expectations in a table" pattern).
Set the pipeline configuration key `xasset.catalog` if your catalog is not `xasset`.
"""
import re

from pyspark import pipelines as dp

CATALOG = spark.conf.get("xasset.catalog", "xasset")  # noqa: F821  (spark is provided by the pipeline)
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", CATALOG):
    raise ValueError(f"bad catalog name {CATALOG!r}")

_rules = (spark.read.table(f"{CATALOG}.silver.dq_rule_catalog")  # noqa: F821
          .select("rule_name", "expectation").collect())
EXPECTATIONS = {r["rule_name"]: r["expectation"] for r in _rules}
if len(EXPECTATIONS) != 10:
    raise ValueError(f"expected 10 rules in dq_rule_catalog, found {len(EXPECTATIONS)}")


@dp.materialized_view(
    name="silver_prices_dq_monitor",
    comment="silver.prices re-checked against the 10 DQ rules as warn-only expectations (rows are kept).",
)
@dp.expect_all(EXPECTATIONS)
def silver_prices_dq_monitor():
    return spark.read.table(f"{CATALOG}.silver.prices")  # noqa: F821
