"""Hold-out discipline: parameter selection can't see OOS data, and the IS/OOS label split is purged."""
import numpy as np
import pytest
import pandas as pd

from xasset import config
from xasset.backtest import BacktestResult
from xasset.research import OOS, check_holdout_definition, in_sample_only, select_on_is, split_signals


def fake_result(is_mean, oos_mean, seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", "2026-09-30")
    r = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    r[idx < OOS] += is_mean
    r[idx >= OOS] += oos_mean
    zero = pd.Series(0.0, index=idx)
    return BacktestResult(returns=r, gross=r, costs=zero, turnover=pd.Series(1.0, index=idx[::21]),
                          weights=pd.DataFrame(index=idx[::21]))


def test_holdout_is_last_three_years():
    check_holdout_definition()
    assert OOS == pd.Timestamp(config.END_DATE) - pd.DateOffset(years=3) + pd.Timedelta(days=1)


def test_selection_ignores_out_of_sample_performance():
    grid = {"good_is": fake_result(0.001, -0.005, 1), "bad_is": fake_result(0.0, 0.0, 2)}
    assert select_on_is(grid) == "good_is"
    # Make the IS winner's OOS spectacular or terrible: the choice must not move.
    for oos in (-0.05, 0.05):
        grid["good_is"] = fake_result(0.001, oos, 1)
        assert select_on_is(grid) == "good_is"


def test_in_sample_only_drops_everything_from_oos_start():
    cut = in_sample_only(fake_result(0, 0, 3))
    for s in (cut.returns, cut.gross, cut.costs, cut.turnover):
        assert s.index.max() < OOS


def test_signal_split_purges_the_straddling_label():
    sig = pd.DatetimeIndex(["2023-07-31", "2023-08-31", "2023-09-29", "2023-10-31"])
    win = pd.DataFrame({"label_start": pd.to_datetime(["2023-08-01", "2023-09-01", "2023-10-02", "2023-11-01"]),
                        "label_end": pd.to_datetime(["2023-09-01", "2023-10-02", "2023-11-01", "2023-12-01"])},
                       index=sig)
    is_, oos = split_signals(win)
    assert list(is_) == [sig[0]]
    assert list(oos) == [sig[2], sig[3]]
    assert sig[1] not in is_ and sig[1] not in oos  # its label crosses OOS_START


def test_ic_tstat_and_ml_pvalue_are_separately_computed():
    """Momentum's IC t-stat and the ML-vs-momentum p-value are different statistics.

    Recompute both from the monthly ICs and check them against the stored results.
    """
    import json

    from scipy import stats

    res = config.RESULTS_DIR
    summary = json.loads((res / "summary.json").read_text())
    ic = pd.read_csv(res / "ic_monthly.csv", index_col="date", parse_dates=True)
    oos_ic = ic[ic.index >= summary["first_oos_signal"]]

    mom = oos_ic["mom_12_1"].dropna()
    t_mom = mom.mean() / (mom.std(ddof=1) / np.sqrt(len(mom)))   # one-sample t on the IC itself
    stored_t = pd.read_csv(res / "ic_summary.csv").set_index(["signal", "period"]).loc[
        ("mom_12_1", "out-of-sample"), "ic_t"]
    assert t_mom == pytest.approx(stored_t, rel=1e-9)

    both = oos_ic[["ml_pred", "mom_12_1"]].dropna()
    t_diff, p_diff = stats.ttest_1samp(both["ml_pred"] - both["mom_12_1"], 0.0)   # paired test on the difference
    stored = pd.read_csv(res / "ml_vs_momentum.csv").set_index("period").loc["out-of-sample"]
    assert p_diff == pytest.approx(stored["diff_p"], rel=1e-9)
    assert t_diff == pytest.approx(stored["diff_t"], rel=1e-9)

    # Different quantities: a t-stat for "mean IC != 0" vs a p-value for "ML IC != momentum IC".
    assert t_mom != pytest.approx(p_diff, abs=1e-6)
    assert t_diff != pytest.approx(t_mom, abs=1e-3)
