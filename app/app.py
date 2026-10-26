"""Streamlit dashboard. Reads only the pre-aggregated files in results/.

    streamlit run app/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RES = Path(__file__).resolve().parents[1] / "results"
REPO = "https://github.com/SanellaPalangic2/cross-asset-momentum"

NAMES = {"xs_mom": "XS momentum", "ts_trend": "TS trend", "ew_rebal": "Equal weight (rebalanced)",
         "ew_hold": "Equal weight (buy & hold)", "ml_xs": "ML rank L/S"}
# Fixed entity -> color mapping (colorblind-validated order); filtering never repaints survivors.
COLORS = {"XS momentum": "#2a78d6", "TS trend": "#eb6834", "Equal weight (rebalanced)": "#1baf7a",
          "Equal weight (buy & hold)": "#eda100", "ML rank L/S": "#e87ba4"}
COLOR_SCALE = alt.Scale(domain=list(COLORS), range=list(COLORS.values()))

st.set_page_config(page_title="Cross-Asset Momentum & Trend", layout="wide")


@st.cache_data
def load():
    daily = pd.read_parquet(RES / "daily_returns.parquet")
    return {
        "summary": json.loads((RES / "summary.json").read_text()),
        "daily": daily,
        "perf": pd.read_csv(RES / "performance.csv"),
        "ic": pd.read_csv(RES / "ic_summary.csv"),
        "ic_monthly": pd.read_csv(RES / "ic_monthly.csv", parse_dates=["date"]),
        "grid": pd.read_csv(RES / "param_grid.csv"),
        "cost": pd.read_csv(RES / "cost_sensitivity.csv"),
        "corr": pd.read_csv(RES / "strategy_correlation.csv", index_col=0),
        "ml": pd.read_csv(RES / "ml_vs_momentum.csv"),
    }


d = load()
s = d["summary"]
oos = pd.Timestamp(s["oos_start"])

st.title("Cross-asset momentum & trend")
st.caption(f"{s['n_assets']} ETFs, {s['data_start']} to {s['data_end']}. Out-of-sample hold-out from {s['oos_start']} "
           f"(never used for any choice). Costs {s['cost_bps']:g} bp per side. Sharpe is on returns in excess of the "
           f"3-month T-bill; idle cash earns it and leverage pays it. "
           f"[Code and methodology]({REPO})")

perf = d["perf"].assign(name=d["perf"]["strategy"].map(NAMES))
oos_perf = perf[perf["period"] == "out-of-sample"].set_index("strategy")
is_perf = perf[perf["period"] == "in-sample"].set_index("strategy")
cols = st.columns(4)
for col, k in zip(cols, ["xs_mom", "ts_trend", "ew_rebal", "ml_xs"]):
    col.metric(f"{NAMES[k]}: OOS Sharpe (excess)", f"{oos_perf.loc[k, 'sharpe']:.2f}",
               delta=f"{oos_perf.loc[k, 'sharpe'] - is_perf.loc[k, 'sharpe']:+.2f} vs in-sample",
               delta_color="off")

picked = st.multiselect("Strategies", list(NAMES.values()), default=list(NAMES.values())[:3])
keys = [k for k, v in NAMES.items() if v in picked]

tab_perf, tab_ic, tab_robust, tab_corr = st.tabs(["Performance", "Signal quality (IC)", "Robustness", "Correlation"])

with tab_perf:
    daily = d["daily"][keys]
    nav = (1 + daily.fillna(0)).cumprod().where(daily.notna().cumsum() > 0)
    weekly = nav.resample("W-FRI").last().reset_index().melt("date", var_name="k", value_name="growth").dropna()
    weekly["strategy"] = weekly["k"].map(NAMES)
    band = alt.Chart(pd.DataFrame({"start": [oos], "end": [nav.index.max()]})).mark_rect(
        color="#e1e0d9", opacity=0.6).encode(x="start:T", x2="end:T")
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    lines = alt.Chart(weekly).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("growth:Q", scale=alt.Scale(type="log"), title="Growth of $1 (log)"),
        color=alt.Color("strategy:N", scale=COLOR_SCALE, legend=alt.Legend(orient="top", title=None)))
    rule = alt.Chart(weekly).mark_rule(color="#898781").encode(
        x="date:T", opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("strategy:N"), alt.Tooltip("growth:Q", format=".2f")]
    ).add_params(hover)
    st.subheader("Cumulative growth, net of costs (shaded = out-of-sample)")
    st.altair_chart((band + lines + rule).properties(height=380), width="stretch")

    period = st.radio("Period", ["out-of-sample", "in-sample", "full"], horizontal=True)
    t = perf[(perf["period"] == period) & perf["strategy"].isin(keys)]
    st.dataframe(pd.DataFrame({
        "Strategy": t["name"], "From": t["start"], "To": t["end"],
        "Ann. return": t["ann_return"].map("{:.1%}".format), "Ann. vol": t["ann_vol"].map("{:.1%}".format),
        "Sharpe (excess)": t["sharpe"].map("{:.2f}".format),
        "Sharpe (0% rf, old)": t["sharpe_rf0"].map("{:.2f}".format),
        "Max DD": t["max_drawdown"].map("{:.0%}".format),
        "Turnover/yr": t["turnover_per_year"].map("{:.1f}".format),
        "Hit rate": t["hit_rate_monthly"].map("{:.0%}".format)}), hide_index=True, width="stretch")

with tab_ic:
    st.subheader("Information coefficient: Spearman(signal, next-month return)")
    st.dataframe(d["ic"].round(3), hide_index=True, width="stretch")
    roll = d["ic_monthly"].set_index("date")[["ml_pred", "mom_12_1"]].rolling(12, min_periods=12).mean()
    roll = roll.rename(columns={"ml_pred": "ML prediction", "mom_12_1": "12-1 momentum"}).reset_index().melt(
        "date", var_name="signal", value_name="ic").dropna()
    ic_chart = alt.Chart(roll).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None), y=alt.Y("ic:Q", title="12-month mean IC"),
        color=alt.Color("signal:N", scale=alt.Scale(domain=["ML prediction", "12-1 momentum"],
                                                    range=["#2a78d6", "#eb6834"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("date:T"), "signal:N", alt.Tooltip("ic:Q", format=".3f")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#c3c2b7").encode(y="y:Q")
    st.altair_chart((band.properties() + zero + ic_chart).properties(height=320), width="stretch")
    st.markdown("**ML vs momentum on identical months (paired t-test on the IC difference)**")
    st.dataframe(d["ml"].round(3), hide_index=True, width="stretch")

with tab_robust:
    st.subheader("Excess-return Sharpe vs transaction cost")
    c = d["cost"].assign(strategy=d["cost"]["strategy"].map(NAMES))
    c = c[c["strategy"].isin(picked)]
    st.altair_chart(alt.Chart(c).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2).encode(
        x=alt.X("cost_bps:Q", title="cost per side (bp)"), y=alt.Y("sharpe:Q", title="Sharpe (excess)"),
        color=alt.Color("strategy:N", scale=COLOR_SCALE, legend=alt.Legend(orient="top", title=None)),
        column=alt.Column("period:N", title=None),
        tooltip=["strategy:N", "period:N", "cost_bps:Q", alt.Tooltip("sharpe:Q", format=".2f"),
                 alt.Tooltip("turnover_per_year:Q", format=".1f")]).properties(width=380, height=280))
    st.subheader("Parameter sensitivity (selection used in-sample Sharpe only)")
    st.dataframe(d["grid"].round(3), hide_index=True, width="stretch")

with tab_corr:
    st.subheader("Correlation of monthly returns")
    corr = d["corr"].reset_index().melt("index", var_name="b", value_name="rho").rename(columns={"index": "a"})
    order = list(d["corr"].index)
    heat = alt.Chart(corr).mark_rect(stroke="#fcfcfb", strokeWidth=2).encode(
        x=alt.X("b:N", sort=order, title=None), y=alt.Y("a:N", sort=order, title=None),
        color=alt.Color("rho:Q", scale=alt.Scale(domain=[-1, 0, 1], range=["#184f95", "#f0efec", "#a8302f"]),
                        title="corr"),
        tooltip=["a:N", "b:N", alt.Tooltip("rho:Q", format=".2f")])
    text = heat.mark_text(fontSize=11).encode(text=alt.Text("rho:Q", format=".2f"),
                                              color=alt.value("#0b0b0b"))
    st.altair_chart((heat + text).properties(height=460), width="stretch")

st.caption("ETF data is not futures data; small universe; one vendor; costs are estimates; the hold-out is short. "
           "Past performance does not predict future results.")
