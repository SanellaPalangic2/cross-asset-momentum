"""Every research return must come from Adj Close (total return), never the split-only Close.

yfinance `Close` is adjusted for splits but not dividends. A return computed
from it misses the income, which for bond and credit ETFs is most of the
return. These tests guard against that at four levels:

1. Static: only allowlisted ingestion/DQ modules may reference the `close`
   column. Inside them, any return-like computation on `close` must be the
   labelled diagnostic `split_only_return`. Any future features/backtest/
   benchmark module that touches `close` fails here.
2. Contract: analysis.panel, the only input to research code, exposes no
   split-only price or return column.
3. Behaviour: on a synthetic dividend, daily_return equals the total return,
   not the split-only price change.
4. Data: if the pipeline has run, the real panel's daily_return matches
   adj_close.pct_change() exactly.
"""
import ast
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from xasset import config
from xasset.curate import align_macro
from xasset.quality import apply_rules
from xasset.warehouse import create_analysis_view, replace_table

PKG = Path(__file__).resolve().parents[1] / "src" / "xasset"

# Modules allowed to reference `close`, and why.
CLOSE_ALLOWLIST = {
    "ingest.py",   # column renaming / schema of the raw download
    "quality.py",  # NULL/non-positive checks, stale detection, split_only_return diagnostic
    "report.py",   # displays the diagnostic
}
RETURN_LIKE_CALLS = {"pct_change", "cumprod", "log", "log1p"}
DIAGNOSTIC_NAME = "split_only_return"
SPLIT_ONLY_COLUMNS = {"close", DIAGNOSTIC_NAME}


def _references_close(node: ast.AST) -> bool:
    """True for df["close"] / g["close"] subscripts and .close attribute access."""
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return node.slice.value == "close"
    return isinstance(node, ast.Attribute) and node.attr == "close"


def _is_return_like(stmt: ast.AST) -> bool:
    for n in ast.walk(stmt):
        if isinstance(n, ast.Call):
            fn = n.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name in RETURN_LIKE_CALLS:
                return True
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            return True
    return False


def _assign_targets(stmt: ast.AST) -> set[str]:
    targets = getattr(stmt, "targets", None) or [getattr(stmt, "target", None)]
    names = set()
    for t in filter(None, targets):
        if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant):
            names.add(t.slice.value)
        elif isinstance(t, ast.Name):
            names.add(t.id)
    return names


def find_split_only_return_violations(source: str, filename: str) -> list[str]:
    tree = ast.parse(source)
    # `.close` as a called method (con.close(), file.close()) is not the price column.
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    uses = [n for n in ast.walk(tree) if _references_close(n) and id(n) not in called]
    if not uses:
        return []
    if filename not in CLOSE_ALLOWLIST:
        return [f"{filename}:{n.lineno} references `close`; research code must use adj_close / "
                f"analysis.panel.daily_return" for n in uses]
    out = []
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.stmt) or isinstance(stmt, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            continue
        if any(_references_close(n) for n in ast.walk(stmt)) and _is_return_like(stmt):
            if _assign_targets(stmt) != {DIAGNOSTIC_NAME}:
                out.append(f"{filename}:{stmt.lineno} computes a return from `close` "
                           f"(only `{DIAGNOSTIC_NAME}` may)")
    return out


# ---------------------------------------------------------------- 1. static

def test_no_module_computes_returns_from_split_only_close():
    violations = []
    for path in sorted(PKG.glob("*.py")):
        violations += find_split_only_return_violations(path.read_text(encoding="utf-8"), path.name)
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("snippet, filename", [
    ('r = df["close"].pct_change()', "features.py"),                    # new module touching close
    ('mom = px["close"] / px["close"].shift(252) - 1', "backtest.py"),
    ('bench = np.log(df["close"]).diff()', "benchmarks.py"),
    ('u["daily_return"] = g["close"].pct_change()', "quality.py"),      # allowlisted file, wrong target
    ('df["ret"] = df["close"] / df["close"].shift() - 1', "quality.py"),
    ('r = px.close.pct_change()', "signals.py"),                         # attribute access
])
def test_guard_catches_split_only_returns(snippet, filename):
    """The guard itself must fire, or test 1 proves nothing."""
    assert find_split_only_return_violations(snippet, filename)


@pytest.mark.parametrize("snippet, filename", [
    ('u["split_only_return"] = g["close"].pct_change()', "quality.py"),  # the labelled diagnostic
    ('same = g["close"].diff().eq(0)', "quality.py"),                     # equality check, not a return
    ('r = df["adj_close"].pct_change()', "features.py"),
    ('con.close()', "pipeline.py"),                                       # DB connection, not a price
])
def test_guard_allows_legitimate_uses(snippet, filename):
    assert not find_split_only_return_violations(snippet, filename)


# ---------------------------------------------------------- 2-3. contract/behaviour

@pytest.fixture
def dividend_panel(monkeypatch):
    """One ticker; on session 40 a 3% dividend goes ex: Close drops 3%, total return is 0%."""
    monkeypatch.setattr(config, "UNIVERSE", {"AAA": "credit"})
    end = "2007-04-30"
    from xasset.calendar import nyse_sessions
    s = nyse_sessions(config.START_DATE, end)
    close = np.full(len(s), 100.0)
    close[40:] = 97.0
    # Back-adjust history like Yahoo: pre-ex-date prices scaled by (1 - div / prior close).
    adj = close.copy()
    adj[:40] *= 0.97
    raw = pd.DataFrame({"ticker": "AAA", "date": s, "open": close, "high": close, "low": close,
                        "close": close, "adj_close": adj, "volume": 1e6})
    clean = apply_rules(raw, end=end)["clean"]
    macro = pd.DataFrame({"series_id": list(config.FRED_SERIES), "date": s[0], "value": 1.0})
    con = duckdb.connect()
    for schema in ("clean", "analysis"):
        con.execute(f"CREATE SCHEMA {schema}")
    replace_table(con, "clean.prices", clean)
    replace_table(con, "clean.macro_daily", align_macro(macro, end=end))
    create_analysis_view(con)
    panel = con.execute("SELECT * FROM analysis.panel").df()
    return clean, panel, s[40]


def test_panel_exposes_no_split_only_columns(dividend_panel):
    _, panel, _ = dividend_panel
    assert not SPLIT_ONLY_COLUMNS & set(panel.columns)
    assert {"adj_close", "daily_return"} <= set(panel.columns)


def test_daily_return_is_total_return_on_ex_dividend_day(dividend_panel):
    clean, panel, ex_date = dividend_panel
    on_ex = panel.set_index("date").loc[ex_date]
    assert on_ex["daily_return"] == pytest.approx(0.0)            # holder lost nothing
    split_only = clean.set_index("date").loc[ex_date, "split_only_return"]
    assert split_only == pytest.approx(-0.03)                      # Close says -3%
    # Over the window, compounding daily_return reproduces the adj_close path.
    growth = (1 + panel["daily_return"].fillna(0)).prod()
    assert growth == pytest.approx(panel["adj_close"].iloc[-1] / panel["adj_close"].iloc[0])


# ---------------------------------------------------------------- 4. real data

def test_real_panel_returns_are_adj_close_returns():
    path = config.CURATED_DIR / "panel.parquet"
    if not path.exists():
        pytest.skip("pipeline has not been run; no curated panel")
    p = pd.read_parquet(path).sort_values(["ticker", "date"])
    p = p[p["adj_close"].notna()]
    expected = p.groupby("ticker")["adj_close"].pct_change(fill_method=None)
    pd.testing.assert_series_equal(p["daily_return"], expected, check_names=False, rtol=1e-12)
