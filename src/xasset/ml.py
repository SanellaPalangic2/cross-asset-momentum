"""Walk-forward gradient boosting: predict next-month cross-sectional return rank.

Honesty rules:
- Hyperparameters are fixed in config.ML_PARAMS and never tuned.
- The model is retrained once a year (expanding window). At the first signal
  date of year Y, it may only train on rows whose label window had already
  ENDED (label_end <= that date). Labels that straddle the retraining date are
  purged. This is asserted at runtime.
- Inputs are cross-sectional ranks, so the model learns relative ordering,
  which is the same task the momentum signal performs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config

RANKED = ["mom_12_1", "ret_1m", "trend_200", "vol_60"]
FEATURES = [f"rank_{c}" for c in RANKED] + ["curve_inverted"]


def build_dataset(features: dict, fwd: pd.DataFrame, windows: pd.DataFrame,
                  dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Long table (date, ticker): ranked features at the signal date plus the forward-rank label.

    Signal dates without a complete label are kept (label NaN), so the latest
    month can still be predicted.
    """
    parts = {f"rank_{c}": features[c].loc[dates].rank(axis=1, pct=True) for c in RANKED}
    ds = pd.concat({k: v.stack() for k, v in parts.items()}, axis=1)
    ds.index.names = ["date", "ticker"]
    ds = ds.reset_index()
    ds["curve_inverted"] = ds["date"].map(features["curve_inverted"]).astype(float)
    label = fwd.rank(axis=1, pct=True).stack().rename("label").reset_index()
    label.columns = ["date", "ticker", "label"]
    ds = ds.merge(label, on=["date", "ticker"], how="left")
    ds = ds.merge(windows, left_on="date", right_index=True, how="left")
    return ds.dropna(subset=FEATURES[:-1]).reset_index(drop=True)


def walk_forward(ds: pd.DataFrame, min_train_months: int = config.ML_MIN_TRAIN_MONTHS,
                 params: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Yearly expanding-window fits. Returns (predictions date x ticker, training log)."""
    params = params or config.ML_PARAMS
    preds, log = [], []
    for year in sorted(ds["date"].dt.year.unique()):
        test = ds[ds["date"].dt.year == year]
        cutoff = test["date"].min()
        train = ds[ds["label_end"].notna() & (ds["label_end"] <= cutoff)]
        if train["date"].nunique() < min_train_months:
            continue
        assert train["label_end"].max() <= cutoff, "training label not yet known at fit time"
        assert train["date"].max() < cutoff
        model = HistGradientBoostingRegressor(**params).fit(train[FEATURES], train["label"])
        p = test[["date", "ticker"]].assign(pred=model.predict(test[FEATURES]))
        preds.append(p)
        log.append({"test_year": int(year), "fit_date": cutoff.date().isoformat(),
                    "train_months": int(train["date"].nunique()), "train_rows": int(len(train)),
                    "last_train_label_end": train["label_end"].max().date().isoformat(),
                    "test_months": int(test["date"].nunique())})
    pred = pd.concat(preds).pivot(index="date", columns="ticker", values="pred")
    return pred, pd.DataFrame(log)
