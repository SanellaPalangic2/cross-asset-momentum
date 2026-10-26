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

# =============================================================================
# Research settings. Everything below is fixed BEFORE looking at results.
# =============================================================================
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252

# Hold-out: the last 3 years of returns are never used for any choice.
HOLDOUT_YEARS = 3
OOS_START = "2023-10-01"   # = END_DATE - 3 years + 1 day; asserted in research.py

# Execution: signal computed at the close of signal date s (last session of the
# month). The trade executes at the close of the NEXT session (s+1), so the new
# position first earns the return of session s+2. One day of delay costs
# little at a monthly horizon and rules out trading on the same close the
# signal was computed from.
EXECUTION_LAG_SESSIONS = 1

COST_BPS_PER_SIDE = 10.0
COST_SENSITIVITY_BPS = [0.0, 10.0, 25.0, 50.0]

# Feature defaults (the spec's values)
MOM_LOOKBACK_MONTHS = 12
MOM_SKIP_MONTHS = 1
REVERSAL_MONTHS = 1
TREND_MA_DAYS = 200
VOL_WINDOW_DAYS = 60

# Cross-sectional momentum: long top third, short bottom third, equal weight.
XS_QUANTILE = 1 / 3

# Time-series trend: each asset is scaled to VOL_TARGET_PER_ASSET using its
# trailing 60-day vol, divided by the number of assets. Per-asset leverage is
# capped (SHY's ~1.5% vol would otherwise get >10x), and so is gross leverage.
VOL_TARGET_PER_ASSET = 0.20
MAX_ASSET_LEVERAGE = 5.0
MAX_GROSS_LEVERAGE = 2.0

# Parameter grids. Selection rule: highest in-sample Sharpe net of
# COST_BPS_PER_SIDE. The full grid is reported (IS and OOS) so the choice is
# visible, but only IS data enters the selection.
XS_MOM_GRID = [(3, 1), (6, 1), (9, 1), (12, 1), (12, 0)]   # (lookback months, skip months)
TREND_MA_GRID = [50, 100, 150, 200, 250]
TREND_BELOW_GRID = ["flat", "short"]

# Stress windows (descriptive only; all are inside the in-sample period).
SUBPERIODS = {
    "2008 crisis": ("2008-09-01", "2009-03-31"),
    "2020 COVID": ("2020-02-19", "2020-04-30"),
    "2022 rate hikes": ("2022-01-01", "2022-12-31"),
}

# Asset-class sleeves for the correlation heatmap.
SLEEVES = {
    "equity": ["equity_sector", "equity_broad", "equity_intl"],
    "rates": ["treasury", "tips"],
    "credit": ["credit"],
    "real_estate": ["real_estate"],
    "commodity": ["commodity"],
}

# ML: hyperparameters are fixed up front and never tuned.
ML_PARAMS = dict(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=40,
                 l2_regularization=1.0, random_state=0)
ML_MIN_TRAIN_MONTHS = 36
