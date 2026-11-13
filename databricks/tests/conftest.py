"""Make databricks/ (pipeline_lib, export_for_databricks) and src/ (the original xasset) importable.

Run from the repo root:  pytest databricks/tests
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1] / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
