"""Real-money backtest study.

Translates the unit-normalised equity curves from
``results/ablation/equity_curves_v2.csv`` into dollar P&L for a range of
starting capitals, and decomposes returns by year, computing drawdown
in dollar terms.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ABL  = ROOT / "results" / "ablation"

CAPITALS = [10_000, 100_000, 1_000_000]
CONFIGS  = ["B1_BuyHold", "B2_TechThresh", "B3_SentThresh",
            "B4v2_EarlyFusion", "Pv2_LateFusion"]


def _load_equity() -> pd.DataFrame:
    df = pd.read_csv(ABL / "equity_curves_v2.csv", parse_dates=["date"])
    return df.set_index("date").sort_index()


def _annualised(rets: pd.Series) -> float:
    n_years = len(rets) / 252
    cum     = (1 + rets).prod()
    return float(cum ** (1 / n_years) - 1) if n_years > 0 else 0.0


def _max_dd_dollars(pv: pd.Series, capital: float) -> tuple[float, float]:
    dollars  = pv * capital
    roll_max = dollars.cummax()
    dd       = dollars - roll_max
    i_min    = dd.idxmin()
    return float(dd.min()), float(roll_max.loc[i_min])


def per_year_breakdown(equity: pd.DataFrame) -> pd.DataFrame:
    """Return year × config table of yearly returns (%)."""
    rets   = equity.pct_change().fillna(0.0)
    years  = rets.index.year
    out    = {}
    for cfg in CONFIGS:
        per_y = {}
        for y in sorted(set(years)):
            mask = years == y
            cum  = float((1 + rets.loc[mask, cfg]).prod() - 1)
            per_y[y] = cum * 100.0
        out[cfg] = per_y
    return pd.DataFrame(out)


def dollar_study(equity: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Full per-config dollar summary for a given starting capital."""
    rows = []
    for cfg in CONFIGS:
        pv      = equity[cfg]
        rets    = pv.pct_change().dropna()
        final   = float(pv.iloc[-1]) * capital
        profit  = final - capital
        cagr    = _annualised(rets)
        max_dd_dollars, peak = _max_dd_dollars(pv, capital)
        sharpe  = float(np.sqrt(252) * rets.mean() / rets.std(ddof=1)) if rets.std() > 0 else 0.0
        rows.append({
            "config":           cfg,
            "final_value_$":    final,
            "profit_$":         profit,
            "total_return_%":   (final / capital - 1) * 100,
            "cagr_%":           cagr * 100,
            "max_dd_$":         max_dd_dollars,
            "max_dd_%":         max_dd_dollars / peak * 100,
            "peak_$":           peak,
            "sharpe":           sharpe,
            "profit_per_year_$": profit / (len(rets) / 252),
        })
    return pd.DataFrame(rows).set_index("config")


def fusion_vs_buyhold_episodes(equity: pd.DataFrame) -> dict:
    """How often does late fusion beat buy-and-hold on a rolling basis?"""
    rets = equity.pct_change().fillna(0.0)
    bh   = rets["B1_BuyHold"]
    lf   = rets["Pv2_LateFusion"]
    monthly = (
        pd.DataFrame({"bh": bh, "lf": lf})
        .resample("ME").apply(lambda r: (1 + r).prod() - 1)
    )
    n_months         = len(monthly)
    lf_wins_monthly  = int((monthly["lf"] > monthly["bh"]).sum())
    lf_neg_months    = int((monthly["lf"] < 0).sum())
    bh_neg_months    = int((monthly["bh"] < 0).sum())
    return {
        "n_months":          n_months,
        "lf_beats_bh_months": lf_wins_monthly,
        "lf_negative_months": lf_neg_months,
        "bh_negative_months": bh_neg_months,
        "lf_avg_monthly_%":  float(monthly["lf"].mean() * 100),
        "bh_avg_monthly_%":  float(monthly["bh"].mean() * 100),
        "lf_worst_month_%":  float(monthly["lf"].min() * 100),
        "bh_worst_month_%":  float(monthly["bh"].min() * 100),
        "lf_best_month_%":   float(monthly["lf"].max() * 100),
        "bh_best_month_%":   float(monthly["bh"].max() * 100),
    }


def _print_dollar_table(summary: pd.DataFrame, capital: float) -> None:
    sep = "=" * 102
    print(f"\n{sep}")
    print(f"  REAL-MONEY STUDY  —  starting capital = ${capital:,.0f}  "
          f"(period {summary.attrs.get('period', '?')})")
    print(sep)
    print(f"  {'Config':<22}  {'Final $':>12}  {'Profit $':>12}  "
          f"{'Total %':>8}  {'CAGR %':>8}  {'MaxDD $':>12}  {'MaxDD %':>8}  {'Sharpe':>7}")
    print("  " + "-" * 98)
    for cfg, row in summary.iterrows():
        print(
            f"  {cfg:<22}  ${row['final_value_$']:>11,.0f}  "
            f"${row['profit_$']:>11,.0f}  "
            f"{row['total_return_%']:>7.2f}%  {row['cagr_%']:>7.2f}%  "
            f"${row['max_dd_$']:>11,.0f}  {row['max_dd_%']:>7.2f}%  "
            f"{row['sharpe']:>7.3f}"
        )
    print(sep)


def main() -> None:
    equity = _load_equity()
    period = f"{equity.index[0].date()} → {equity.index[-1].date()}  ({len(equity)} trading days)"
    print(f"\nLoaded equity curves: {period}")

    # ── Dollar tables ─────────────────────────────────────────────────────────
    all_summaries = {}
    for cap in CAPITALS:
        summary = dollar_study(equity, capital=cap)
        summary.attrs["period"] = period
        _print_dollar_table(summary, cap)
        all_summaries[cap] = summary

    # ── Yearly breakdown ──────────────────────────────────────────────────────
    yearly = per_year_breakdown(equity)
    print("\n" + "=" * 78)
    print("  YEAR-BY-YEAR RETURNS (%)")
    print("=" * 78)
    print(yearly.round(2).to_string())

    # ── Late-fusion vs buy-and-hold robustness ────────────────────────────────
    ep = fusion_vs_buyhold_episodes(equity)
    print("\n" + "=" * 78)
    print("  LATE-FUSION vs BUY-AND-HOLD — robustness")
    print("=" * 78)
    print(f"  Months observed                                  {ep['n_months']:>4d}")
    print(f"  Late-fusion beats buy-hold (monthly return)      {ep['lf_beats_bh_months']:>4d} / {ep['n_months']}")
    print(f"  Negative months — late fusion                    {ep['lf_negative_months']:>4d} / {ep['n_months']}")
    print(f"  Negative months — buy-hold                       {ep['bh_negative_months']:>4d} / {ep['n_months']}")
    print(f"  Avg monthly return  LF: {ep['lf_avg_monthly_%']:>+6.2f}%   BH: {ep['bh_avg_monthly_%']:>+6.2f}%")
    print(f"  Best monthly        LF: {ep['lf_best_month_%']:>+6.2f}%   BH: {ep['bh_best_month_%']:>+6.2f}%")
    print(f"  Worst monthly       LF: {ep['lf_worst_month_%']:>+6.2f}%   BH: {ep['bh_worst_month_%']:>+6.2f}%")
    print("=" * 78)

    # ── Save artefacts ────────────────────────────────────────────────────────
    out_dir = ABL
    for cap, summary in all_summaries.items():
        summary.to_csv(out_dir / f"money_study_{cap}.csv")
    yearly.to_csv(out_dir / "money_study_yearly.csv")
    print(f"\nSaved per-capital summaries → results/ablation/money_study_<capital>.csv")
    print(f"Saved yearly breakdown      → results/ablation/money_study_yearly.csv")


if __name__ == "__main__":
    main()
