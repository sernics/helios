"""Generate publication-quality figures for the thesis.

Outputs land under ``results/figures/`` as PNG (300 dpi).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
ABL      = ROOT / "results" / "ablation"
FIG_DIR  = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":        110,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches":0.15,
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Colour palette — colour-blind friendly
COLORS = {
    "B1_BuyHold":      "#1f77b4",
    "B2_TechThresh":   "#aec7e8",
    "B3_SentThresh":   "#2ca02c",
    "B4v2_EarlyFusion":"#ff7f0e",
    "Pv2_LateFusion":  "#d62728",
}
LABELS = {
    "B1_BuyHold":      "B1  Buy & Hold",
    "B2_TechThresh":   "B2  Tech threshold",
    "B3_SentThresh":   "B3  Sentiment threshold",
    "B4v2_EarlyFusion":"B4  Early fusion",
    "Pv2_LateFusion":  "P   Late fusion",
}


# ── Figure 1: Equity curves ─────────────────────────────────────────────────

def fig_equity_curves() -> None:
    df = pd.read_csv(ABL / "equity_curves_v2.csv", parse_dates=["date"])
    df = df.set_index("date").sort_index()

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for cfg in ["B1_BuyHold", "B2_TechThresh", "B3_SentThresh",
                "B4v2_EarlyFusion", "Pv2_LateFusion"]:
        is_late = (cfg == "Pv2_LateFusion")
        ax.plot(df.index, df[cfg],
                color=COLORS[cfg], label=LABELS[cfg],
                lw=2.3 if is_late else 1.4,
                zorder=3 if is_late else 2,
                alpha=1.0 if is_late else 0.85)

    ax.axhline(1.0, color="black", lw=0.6, alpha=0.5)
    ax.set_ylabel("Portfolio value (start = 1.0)")
    ax.set_xlabel("Date")
    ax.set_title("Equity curves — five configurations on test split (2024–2025)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.2f}×"))
    fig.savefig(FIG_DIR / "fig_equity_curves.png")
    plt.close(fig)


# ── Figure 2: Drawdown curves ───────────────────────────────────────────────

def fig_drawdown_curves() -> None:
    df = pd.read_csv(ABL / "equity_curves_v2.csv", parse_dates=["date"])
    df = df.set_index("date").sort_index()

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for cfg in ["B1_BuyHold", "B3_SentThresh", "B4v2_EarlyFusion",
                "Pv2_LateFusion"]:
        pv = df[cfg]
        dd = (pv - pv.cummax()) / pv.cummax() * 100
        is_late = (cfg == "Pv2_LateFusion")
        ax.fill_between(df.index, dd, 0,
                        color=COLORS[cfg],
                        alpha=0.35 if is_late else 0.18,
                        label=LABELS[cfg])
        ax.plot(df.index, dd, color=COLORS[cfg],
                lw=2.0 if is_late else 1.1,
                alpha=1.0 if is_late else 0.7)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.set_title("Drawdown profiles — late fusion controls tail risk")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    fig.savefig(FIG_DIR / "fig_drawdowns.png")
    plt.close(fig)


# ── Figure 3: Per-asset cum return + max DD bars ────────────────────────────

def fig_per_asset() -> None:
    df = pd.read_csv(ABL / "per_asset_v2.csv").set_index("ticker")
    tickers = df.index.tolist()
    x = np.arange(len(tickers))
    w = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    # Cumulative return
    ax = axes[0]
    ax.bar(x - w/2, df["bh_cumret"] * 100, w,
           color=COLORS["B1_BuyHold"], label="Buy & Hold", alpha=0.85)
    ax.bar(x + w/2, df["lf_cumret"] * 100, w,
           color=COLORS["Pv2_LateFusion"], label="Late fusion")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Per-asset: late fusion gives up upside…")
    ax.legend(loc="upper left", frameon=True)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    # Max DD
    ax = axes[1]
    ax.bar(x - w/2, df["bh_maxdd"] * 100, w,
           color=COLORS["B1_BuyHold"], label="Buy & Hold", alpha=0.85)
    ax.bar(x + w/2, df["lf_maxdd"] * 100, w,
           color=COLORS["Pv2_LateFusion"], label="Late fusion")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Max drawdown (%)")
    ax.set_title("…in exchange for systematically smaller drawdowns "
                 "on every ticker")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_per_asset.png")
    plt.close(fig)


# ── Figure 4: Cost sensitivity ──────────────────────────────────────────────

def fig_cost_sensitivity() -> None:
    df = pd.read_csv(ABL / "cost_sensitivity_v2.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    ax1.plot(df["lambda_cost_bps"], df["sharpe"],
             marker="o", lw=2.2, color=COLORS["Pv2_LateFusion"],
             label="Sharpe (LF)")
    ax1.axhline(1.731, ls="--", color=COLORS["B1_BuyHold"],
                lw=1.4, label="B1 Sharpe (1.73)")
    ax1.axhline(0, color="black", lw=0.5)
    ax1.set_xlabel("Transaction cost (bps per flip)")
    ax1.set_ylabel("Sharpe ratio")
    ax1.set_title("Sharpe degrades with cost")
    ax1.legend(loc="lower left", frameon=True)

    ax2.plot(df["lambda_cost_bps"], df["calmar"],
             marker="o", lw=2.2, color=COLORS["Pv2_LateFusion"],
             label="Calmar (LF)")
    ax2.axhline(1.957, ls="--", color=COLORS["B1_BuyHold"],
                lw=1.4, label="B1 Calmar (1.96)")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("Transaction cost (bps per flip)")
    ax2.set_ylabel("Calmar ratio")
    ax2.set_title("Calmar is robust up to ~20 bps")
    ax2.legend(loc="lower left", frameon=True)

    fig.suptitle("Late-fusion sensitivity to transaction cost",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cost_sensitivity.png")
    plt.close(fig)


# ── Figure 5: Per-regime LF vs BH ───────────────────────────────────────────

def fig_per_regime() -> None:
    df = pd.read_csv(ABL / "per_regime_returns_v2.csv").set_index("regime")
    order = ["high_news", "low_news", "high_vol", "normal"]
    df = df.loc[order]
    x = np.arange(len(df))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ax.bar(x - w/2, df["bh_mean_ret_bps"], w,
           color=COLORS["B1_BuyHold"], label="Buy & Hold", alpha=0.85)
    ax.bar(x + w/2, df["lf_mean_ret_bps"], w,
           color=COLORS["Pv2_LateFusion"], label="Late fusion")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in df.index])
    ax.set_ylabel("Mean daily P&L (bps)")
    ax.set_title("Late fusion wins only in high-vol regime")
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1]
    ax.bar(x, df["lf_share_long_%"],
           color=COLORS["Pv2_LateFusion"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in df.index])
    ax.set_ylabel("Long share (%)")
    ax.set_title("Agent is 2.2× more long on high-vol days")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    for i, v in enumerate(df["lf_share_long_%"]):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

    fig.suptitle("Dynamic routing: regime-conditional behaviour",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_regime.png")
    plt.close(fig)


# ── Figure 6: Yearly returns by strategy ────────────────────────────────────

def fig_yearly_returns() -> None:
    eq = pd.read_csv(ABL / "equity_curves_v2.csv", parse_dates=["date"])
    eq = eq.set_index("date").sort_index()
    rets = eq.pct_change().fillna(0.0)
    cfgs = ["B1_BuyHold", "B2_TechThresh", "B3_SentThresh",
            "B4v2_EarlyFusion", "Pv2_LateFusion"]
    years = sorted(set(rets.index.year))

    data = {y: [] for y in years}
    for y in years:
        mask = rets.index.year == y
        for cfg in cfgs:
            cum = (1 + rets.loc[mask, cfg]).prod() - 1
            data[y].append(cum * 100)

    x = np.arange(len(cfgs))
    w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w/2, data[years[0]], w, label=str(years[0]),
           color="#4c72b0", alpha=0.92)
    ax.bar(x + w/2, data[years[1]], w, label=str(years[1]),
           color="#dd8452", alpha=0.92)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c].split()[0] for c in cfgs])
    ax.set_ylabel("Annual return (%)")
    ax.set_title("Year-by-year breakdown — buy-hold dominates the bull "
                 "2024, late fusion is more stable")
    ax.legend(loc="upper right", frameon=True)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    for i, c in enumerate(cfgs):
        ax.text(i - w/2, data[years[0]][i] + 1.2,
                f"{data[years[0]][i]:+.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, data[years[1]][i] + 1.2,
                f"{data[years[1]][i]:+.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_yearly.png")
    plt.close(fig)


# ── Figure 7: Monthly LF vs BH scatter / histogram ──────────────────────────

def fig_monthly() -> None:
    """Monthly LF vs BH scatter with diagonal — much clearer than 24 bars."""
    eq = pd.read_csv(ABL / "equity_curves_v2.csv", parse_dates=["date"])
    eq = eq.set_index("date").sort_index()
    rets = eq.pct_change().fillna(0.0)
    monthly = rets.resample("ME").apply(lambda r: (1 + r).prod() - 1) * 100
    bh = monthly["B1_BuyHold"]
    lf = monthly["Pv2_LateFusion"]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    lim_lo = min(bh.min(), lf.min()) - 2
    lim_hi = max(bh.max(), lf.max()) + 2

    # Quadrant shading: green = LF beats BH, red = LF loses
    ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi], lim_hi,
                    color="#a8e6b0", alpha=0.18, label="LF beats BH")
    ax.fill_between([lim_lo, lim_hi], lim_lo, [lim_lo, lim_hi],
                    color="#ffb3b3", alpha=0.18, label="BH beats LF")

    # Diagonal
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.2, alpha=0.6,
            label="LF = BH")
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.axvline(0, color="black", lw=0.6, alpha=0.5)

    # Points coloured by quadrant: who lost?
    win = lf > bh
    ax.scatter(bh[win], lf[win], s=85, c=COLORS["Pv2_LateFusion"],
               edgecolor="black", lw=0.7, zorder=4,
               label=f"LF wins ({win.sum()}/{len(monthly)} months)")
    ax.scatter(bh[~win], lf[~win], s=85, c="#b0b0b0",
               edgecolor="black", lw=0.7, zorder=4,
               label=f"BH wins ({(~win).sum()}/{len(monthly)} months)")

    # Annotate worst BH and best BH months
    for label, idx in [("worst BH", bh.idxmin()), ("best BH", bh.idxmax())]:
        ax.annotate(f"{label}\n{idx.strftime('%Y-%m')}",
                    xy=(bh[idx], lf[idx]),
                    xytext=(15, -25 if "worst" in label else 25),
                    textcoords="offset points", fontsize=9,
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Buy & Hold monthly return (%)")
    ax.set_ylabel("Late Fusion monthly return (%)")
    ax.set_title("Monthly return scatter — LF compresses both tails")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_monthly.png")
    plt.close(fig)


# ── Figure 8: Action footprint per ticker ───────────────────────────────────

def fig_action_footprint() -> None:
    sw = pd.read_csv(ABL / "position_switches_v2.csv").set_index("ticker")
    sw = sw.sort_values("pct_long")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.barh(sw.index, sw["pct_long"], color=COLORS["Pv2_LateFusion"], alpha=0.9)
    for i, v in enumerate(sw["pct_long"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Long share over test period (%)")
    ax.set_title("Agent is mostly in cash — selectively long on volatile names")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    ax = axes[1]
    ax.barh(sw.index, sw["switches"], color="#7570b3", alpha=0.85)
    for i, v in enumerate(sw["switches"]):
        ax.text(v + 1.5, i, f"{int(v)}", va="center", fontsize=9)
    ax.set_xlabel("Position switches over 2-year test")
    ax.set_title("Trade activity per ticker")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_action_footprint.png")
    plt.close(fig)


# ── Architecture diagram (schematic) ────────────────────────────────────────

def fig_architecture() -> None:
    """Cleaner layout: linear flow left-to-right, no crossing arrows."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    from matplotlib.patches import FancyBboxPatch
    def box(x, y, w, h, text, color, fontsize=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                     boxstyle="round,pad=0.04",
                                     facecolor=color, edgecolor="black",
                                     lw=1.3, alpha=0.92))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, weight="bold")

    def arrow(x1, y1, x2, y2, color="black", lw=1.6):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                     mutation_scale=18))

    # Column headers (subtle)
    for x, lbl in [(1.2, "INPUTS"), (4.0, "MODULES"),
                   (7.2, "STATE VECTOR"), (10.6, "POLICY")]:
        ax.text(x, 6.2, lbl, ha="center", fontsize=9,
                color="#666", style="italic")

    # Inputs
    box(0.3, 4.5, 1.9, 0.9, "News\narticles", "#ffd9b3")
    box(0.3, 0.7, 1.9, 0.9, "OHLCV\nprices", "#b3d9ff")

    # Modules
    box(3.0, 4.5, 2.0, 0.9, "FinBERT\n(Transformer encoder, 110M)", "#ff9966", 9)
    box(3.0, 0.7, 2.0, 0.9, "XGBoost\n(15 indicators)", "#6699ff", 9)

    # State vector — single conceptual box with all 7 dimensions
    box(5.8, 0.4, 3.0, 5.4,
        "STATE  s_t ∈ [−1, 1]⁷\n\n"
        "s_sent, c_sent  ← FinBERT\n\n"
        "s_tech, c_tech  ← XGBoost\n\n"
        "r̃_{t−1}, r̃_{t−2}, r̃_{t−3}\n(lagged returns)",
        "#fff2cc", 9)

    # PPO policy
    box(9.6, 2.5, 2.6, 2.1, "PPO  policy π\n(MLP 64×64)\n+ Sortino reward",
        "#cc6666", 10)

    # Action
    box(9.6, 0.4, 2.6, 1.1, "Action a_t\n0 = cash   1 = long",
        "#99ff99", 10)

    # Arrows (no crossings)
    arrow(2.2, 4.95, 3.0, 4.95)    # news → FinBERT
    arrow(2.2, 1.15, 3.0, 1.15)    # prices → XGB
    arrow(5.0, 4.95, 5.8, 4.6)     # FinBERT → state
    arrow(5.0, 1.15, 5.8, 1.6)     # XGBoost → state
    arrow(8.8, 3.4, 9.6, 3.4)      # state → PPO
    arrow(10.9, 2.5, 10.9, 1.5)    # PPO → action

    ax.set_title("Helios system architecture — late fusion via (score, confidence)",
                 fontsize=13, pad=14)
    fig.savefig(FIG_DIR / "fig_architecture.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_equity_curves()
    fig_drawdown_curves()
    fig_per_asset()
    fig_cost_sensitivity()
    fig_per_regime()
    fig_yearly_returns()
    fig_monthly()
    fig_action_footprint()
    print(f"All figures saved to {FIG_DIR.relative_to(ROOT)}/")
