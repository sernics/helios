"""Extra analyses requested for the thesis: per-asset, per-regime, and
transaction-cost sensitivity."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config
from src.environment.trading_env import TradingEnv
from src.fusion.baselines import _load_prices, _portfolio_value, compute_metrics

ROOT       = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "splits"
FUSION_DIR = ROOT / "results" / "fusion"
ABL        = ROOT / "results" / "ablation"
SPLIT      = "test"


def _run_late_fusion(lambda_cost: float | None = None) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Run Pv2 late fusion on test. Return portfolio value, positions per (ticker,date), and per-ticker pv."""
    model = PPO.load(str(FUSION_DIR / "ppo_late_fusion_v2.zip"))
    lc = config.LAMBDA_COST if lambda_cost is None else lambda_cost
    env = TradingEnv(split=SPLIT, fusion_mode="late", lambda_cost=lc)
    prices = _load_prices(SPLIT)

    pos_records: list[dict] = []
    for (ticker, year) in sorted(env._episodes.keys()):
        obs, _ = env.reset(options={"ticker": ticker, "year": year})
        ep_snap = env._ep_df.copy()
        step, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            date = ep_snap.iloc[step]["date"]
            pos_records.append({"ticker": ticker, "date": date,
                                 "position": float(int(action))})
            step += 1
            done = terminated or truncated

    pos_df = pd.DataFrame(pos_records)
    pv = _portfolio_value(prices, pos_df, lambda_cost=lc)

    # Per-ticker portfolio value
    per_ticker = {}
    for tkr, grp in prices.groupby("ticker"):
        tkr_pos = pos_df[pos_df["ticker"] == tkr]
        per_ticker[tkr] = _portfolio_value(grp, tkr_pos, lambda_cost=lc)
    per_ticker_pv = pd.DataFrame(per_ticker).sort_index()

    return pv, pos_df, per_ticker_pv


def per_asset_analysis() -> pd.DataFrame:
    print("\n=== Per-asset analysis ===")
    _, pos_df, per_ticker_pv = _run_late_fusion()
    prices = _load_prices(SPLIT)

    rows = []
    for tkr in sorted(prices["ticker"].unique()):
        # Late-fusion returns for this ticker
        lf_pv = per_ticker_pv[tkr].dropna()
        lf = compute_metrics(lf_pv)

        # Buy-and-hold for this ticker
        tkr_prices = prices[prices["ticker"] == tkr].sort_values("date")
        bh_pv = (tkr_prices["close"] / tkr_prices["close"].iloc[0]).reset_index(drop=True)
        bh_pv.index = tkr_prices["date"].values
        bh = compute_metrics(bh_pv)

        # Position breakdown
        positions = pos_df[pos_df["ticker"] == tkr]["position"]
        pct_long = (positions == 1.0).mean() * 100

        rows.append({
            "ticker":      tkr,
            "lf_sharpe":   lf["sharpe"],
            "lf_cumret":   lf["cumulative_return"],
            "lf_maxdd":    lf["max_drawdown"],
            "bh_sharpe":   bh["sharpe"],
            "bh_cumret":   bh["cumulative_return"],
            "bh_maxdd":    bh["max_drawdown"],
            "pct_long_%":  pct_long,
        })
    df = pd.DataFrame(rows).set_index("ticker")
    print(df.round(3).to_string())
    df.to_csv(ABL / "per_asset_v2.csv")
    return df


def per_regime_returns() -> pd.DataFrame:
    """Compute LF and BH average daily return per regime (high_news/low_news/high_vol/normal)."""
    print("\n=== Per-regime returns ===")
    pv, pos_df, _ = _run_late_fusion()
    prices = _load_prices(SPLIT)

    # Regime labels (mirror ablation.py)
    train = pd.read_parquet(SPLITS_DIR / "train.parquet",
                             columns=["ticker", "date", "close", "news_count"])
    train["date"] = pd.to_datetime(train["date"])
    p25 = float(train["news_count"].quantile(0.25))
    p75 = float(train["news_count"].quantile(0.75))
    train["lr"] = train.groupby("ticker")["close"].transform(lambda s: np.log(s / s.shift(1)))
    vol_th = float(train.groupby("date")["lr"].mean().rolling(20).std().dropna().mean() * 1.5)

    test = pd.read_parquet(SPLITS_DIR / "test.parquet",
                            columns=["ticker", "date", "close", "news_count"])
    test["date"] = pd.to_datetime(test["date"])
    test["lr"] = test.groupby("ticker")["close"].transform(lambda s: np.log(s / s.shift(1)))
    port_vol = (test.groupby("date")["lr"].mean().rolling(20).std()
                .rename("vol_20d").reset_index())
    test = test.merge(port_vol, on="date", how="left")
    test["regime"] = "normal"
    test.loc[test["news_count"] > p75, "regime"] = "high_news"
    test.loc[test["news_count"] <= p25, "regime"] = "low_news"
    test.loc[test["vol_20d"] > vol_th, "regime"] = "high_vol"

    # Merge positions and prices to compute per-day per-ticker LF return
    merged = prices.merge(pos_df, on=["ticker", "date"], how="left")
    merged["position"] = merged["position"].fillna(0.0)
    merged["ret"] = merged.groupby("ticker")["close"].pct_change().fillna(0.0)
    merged["pos_prev"] = merged.groupby("ticker")["position"].shift(1).fillna(0.0)
    merged["lf_pnl"] = merged["pos_prev"] * merged["ret"]
    merged["bh_pnl"] = merged["ret"]
    merged = merged.merge(test[["ticker", "date", "regime"]], on=["ticker", "date"], how="left")

    rows = []
    for regime, sub in merged.groupby("regime"):
        rows.append({
            "regime":           regime,
            "n_obs":            len(sub),
            "lf_mean_ret_bps":  sub["lf_pnl"].mean() * 10000,
            "bh_mean_ret_bps":  sub["bh_pnl"].mean() * 10000,
            "lf_win_rate_%":    (sub["lf_pnl"] > 0).mean() * 100,
            "lf_share_long_%":  (sub["pos_prev"] == 1.0).mean() * 100,
        })
    df = pd.DataFrame(rows).set_index("regime")
    print(df.round(2).to_string())
    df.to_csv(ABL / "per_regime_returns_v2.csv")
    return df


def transaction_cost_sensitivity() -> pd.DataFrame:
    print("\n=== Transaction cost sensitivity (Pv2 inference-time) ===")
    rows = []
    for lc in [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]:
        pv, _, _ = _run_late_fusion(lambda_cost=lc)
        m = compute_metrics(pv)
        rows.append({"lambda_cost_bps": lc * 10000, **m})
    df = pd.DataFrame(rows).set_index("lambda_cost_bps")
    print(df.round(4).to_string())
    df.to_csv(ABL / "cost_sensitivity_v2.csv")
    return df


def action_distribution() -> dict:
    print("\n=== Action distribution and switches ===")
    _, pos_df, _ = _run_late_fusion()
    n_total = len(pos_df)
    n_long  = int((pos_df["position"] == 1.0).sum())
    n_flat  = int((pos_df["position"] == 0.0).sum())
    # Switches per ticker
    switches = []
    for tkr, grp in pos_df.sort_values("date").groupby("ticker"):
        s = (grp["position"].diff().abs() > 0).sum()
        switches.append({"ticker": tkr, "switches": int(s),
                         "pct_long": float((grp["position"] == 1.0).mean() * 100)})
    sw_df = pd.DataFrame(switches).set_index("ticker")
    summary = {"n_total": n_total, "n_long": n_long, "n_flat": n_flat,
               "pct_long": n_long / n_total * 100,
               "mean_switches_per_ticker": sw_df["switches"].mean()}
    print(f"Total decisions: {n_total}  long: {n_long} ({n_long/n_total*100:.1f}%)  flat: {n_flat} ({n_flat/n_total*100:.1f}%)")
    print(f"Mean position switches per ticker over the 2-year test: {sw_df['switches'].mean():.1f}")
    print(sw_df.to_string())
    sw_df.to_csv(ABL / "position_switches_v2.csv")
    return summary


if __name__ == "__main__":
    per_asset_analysis()
    per_regime_returns()
    transaction_cost_sensitivity()
    action_distribution()
    print("\nAll extra studies saved under results/ablation/")
