"""Static charts for the README, built only from files in results/.

    python analysis/charts.py
"""
from __future__ import annotations

import json

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from xasset import config  # noqa: E402

RES, FIG = config.RESULTS_DIR, config.FIGURES_DIR

# Reference palette (light mode). Categorical slots are assigned in fixed order.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"xs_mom": "#2a78d6", "ts_trend": "#eb6834", "ew_rebal": "#1baf7a"}
SECOND_PAIR = {"ml_pred": "#2a78d6", "mom_12_1": "#eb6834"}
LABELS = {"xs_mom": "XS momentum", "ts_trend": "TS trend", "ew_rebal": "Equal weight",
          "ml_pred": "ML prediction", "mom_12_1": "12-1 momentum"}
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#3987e5", "#f0efec", "#e66767", "#a8302f"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "legend.labelcolor": INK2,
})

OOS = pd.Timestamp(config.OOS_START)


def shade_oos(ax, end, y=1.0, va="top"):
    ax.axvspan(OOS, end, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(OOS, y, "  out-of-sample", transform=ax.get_xaxis_transform(), va=va, ha="left",
            color=INK2, fontsize=9)


def end_labels(ax, items, min_gap_px=14):
    """Label line ends at the right edge, nudged apart so they never overlap.

    items: [(x, y, text)] in data coordinates.
    """
    fig = ax.figure
    fig.canvas.draw()
    to_px, to_data = ax.transData, ax.transData.inverted()
    num = matplotlib.dates.date2num
    pts = sorted(((to_px.transform((num(x), y))[1], x, text) for x, y, text in items))
    placed = []
    for ypx, x, text in pts:
        if placed and ypx - placed[-1][0] < min_gap_px:
            ypx = placed[-1][0] + min_gap_px
        placed.append((ypx, x, text))
    for ypx, x, text in placed:
        xpx = to_px.transform((num(x), 0))[0]
        y = to_data.transform((xpx, ypx))[1]
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=9, annotation_clip=False)


def cumulative(daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ends = []
    for k, c in SERIES.items():
        nav = (1 + daily[k].dropna()).cumprod()
        ax.plot(nav.index, nav, color=c, label=LABELS[k])
        ends.append((nav.index[-1], nav.iloc[-1], f"{LABELS[k]} {nav.iloc[-1]:.2f}x"))
    shade_oos(ax, daily.index[-1])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 0.75, 1, 1.5, 2, 3, 4, 6]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    end_labels(ax, ends)
    ax.set_title("Growth of $1, net of 10 bp costs (log scale)")
    ax.legend(loc="upper left")
    fig.subplots_adjust(right=0.84)
    fig.savefig(FIG / "cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def drawdowns(daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, sharey=True)
    for ax, (k, c) in zip(axes, SERIES.items()):
        nav = (1 + daily[k].dropna()).cumprod()
        dd = nav / nav.cummax() - 1
        ax.fill_between(dd.index, dd, 0, color=c, alpha=0.35, lw=0)
        ax.plot(dd.index, dd, color=c, lw=1.2)
        ax.set_title(f"{LABELS[k]}: max drawdown {dd.min():.0%}", fontsize=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        shade_oos(ax, daily.index[-1], y=0.03, va="bottom")
    fig.suptitle("Drawdowns from previous peak", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "drawdowns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def rolling_ic(ic: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ends = []
    for k, c in SECOND_PAIR.items():
        s = ic[k].dropna().rolling(12, min_periods=12).mean().dropna()
        ax.plot(s.index, s, color=c, label=f"{LABELS[k]} (12-month mean IC)")
        ends.append((s.index[-1], s.iloc[-1], LABELS[k]))
    ax.axhline(0, color=AXIS, lw=1)
    shade_oos(ax, ic.index[-1])
    end_labels(ax, ends)
    ax.set_title("Rolling information coefficient (Spearman, signal vs next-month return)")
    ax.legend(loc="lower left")
    fig.subplots_adjust(right=0.84)
    fig.savefig(FIG / "rolling_ic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def cost_sensitivity(cs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, period in zip(axes, ["in-sample", "out-of-sample"]):
        for k, c in SERIES.items():
            d = cs[(cs["strategy"] == k) & (cs["period"] == period)]
            ax.plot(d["cost_bps"], d["sharpe"], color=c, marker="o", ms=5, label=LABELS[k])
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_title(period.capitalize(), fontsize=11)
        ax.set_xlabel("cost per side (bp)")
        ax.set_xticks(config.COST_SENSITIVITY_BPS)
    axes[0].set_ylabel("Sharpe ratio")
    axes[1].legend(loc="center right")
    fig.suptitle("Sharpe ratio vs transaction cost", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "cost_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def correlation(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    ax.set_title("Correlation of monthly returns: strategies and trend by asset class")
    fig.tight_layout()
    fig.savefig(FIG / "strategy_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(RES / "daily_returns.parquet")
    ic = pd.read_csv(RES / "ic_monthly.csv", index_col="date", parse_dates=True)
    cumulative(daily)
    drawdowns(daily)
    rolling_ic(ic)
    cost_sensitivity(pd.read_csv(RES / "cost_sensitivity.csv"))
    correlation(pd.read_csv(RES / "strategy_correlation.csv", index_col=0))
    print("figures written to", FIG, json.dumps(sorted(p.name for p in FIG.glob("*.png"))))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
