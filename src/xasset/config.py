"""Central configuration. Every tunable choice lives here so it can be cited and changed in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CACHE_DIR = DATA_DIR / "raw"          # immutable API downloads (parquet + manifest)
CURATED_DIR = DATA_DIR / "curated"        # exported analysis-ready panel
DB_PATH = DATA_DIR / "warehouse.duckdb"
REPORTS_DIR = ROOT / "reports"

# --- Study window -----------------------------------------------------------
# The end date is pinned (not "today") so reruns are reproducible. yfinance
# re-computes adjusted prices on every download, so the cache plus a fixed
# window is what makes the dataset stable.
START_DATE = "2007-01-01"
END_DATE = "2026-09-30"

# A ticker that starts after START_DATE but on or before this cutoff is KEPT and
# flagged as a late start. One that starts after the cutoff is DROPPED. A 12-month
# lookback means no signal goes live before 2008 anyway, so a 2007 start costs
# at most a few months of signal history.
LATE_START_CUTOFF = "2007-12-31"

# --- Universe ----------------------------------------------------------------
UNIVERSE = {
    # US equity sectors (SPDR Select Sector)
    "XLB": "equity_sector", "XLE": "equity_sector", "XLF": "equity_sector",
    "XLI": "equity_sector", "XLK": "equity_sector", "XLP": "equity_sector",
    "XLU": "equity_sector", "XLV": "equity_sector", "XLY": "equity_sector",
    # Broad and international equity
    "SPY": "equity_broad", "IWM": "equity_broad",
    "EFA": "equity_intl", "EEM": "equity_intl",
    # Treasuries across the curve
    "SHY": "treasury", "IEF": "treasury", "TLT": "treasury",
    # Credit
    "LQD": "credit", "HYG": "credit",
    # Inflation-linked
    "TIP": "tips",
    # Real estate
    "VNQ": "real_estate",
    # Commodities (ETF proxies for futures exposure)
    "GLD": "commodity", "SLV": "commodity", "DBC": "commodity", "USO": "commodity",
}

FRED_SERIES = {
    "DGS10": "dgs10",    # 10Y constant-maturity Treasury yield, %
    "DGS2": "dgs2",      # 2Y constant-maturity Treasury yield, %
    "T10Y2Y": "t10y2y",  # 10Y minus 2Y, percentage points (published by FRED)
}

# --- Download behaviour ------------------------------------------------------
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0   # wait 2, 4, 8, 16 s (+ jitter) between attempts
BACKOFF_MAX_SECONDS = 60.0

# --- Data-quality thresholds --------------------------------------------------
STALE_RUN_SESSIONS = 5            # close unchanged for >= 5 consecutive sessions
EXTREME_ABS_RETURN = 0.15         # |daily return| above 15% is flagged...
EXTREME_ROBUST_Z = 8.0            # ...as is a move > 8 robust sigmas vs the trailing year
ROBUST_Z_WINDOW = 252
ROBUST_Z_MIN_PERIODS = 60
SPREAD_TOLERANCE_PP = 0.015       # |T10Y2Y - (DGS10 - DGS2)| tolerance (percentage points)

# Rules that EXCLUDE a row from the analysis view. Every other flag is
# informational: the row stays in, flagged for inspection. To reverse a
# decision, move the rule between lists and rerun; no data is deleted.
HARD_RULES = [
    "flag_duplicate_date",      # only the non-first copies are excluded (see quality.py)
    "flag_off_calendar",
    "flag_missing_price",
    "flag_nonpositive_price",
]
SOFT_RULES = [
    "flag_missing_session",
    "flag_stale_price",
    "flag_zero_volume",
    "flag_extreme_return",
    "flag_return_spans_gap",
    "flag_partial_coverage_date",
]

# --- Macro alignment ----------------------------------------------------------
# FRED's DGS10 value for day t comes from the Fed H.15 release, published around
# 4:15pm ET on the next business day, after the equity close. A value dated t is
# therefore not knowable at the close of t, so we lag it one session.
MACRO_PUBLICATION_LAG_SESSIONS = 1
# Forward-fill macro at most this many sessions. A longer gap stays NULL, so a
# dead feed cannot silently freeze a value for months.
MACRO_FFILL_LIMIT_SESSIONS = 5
